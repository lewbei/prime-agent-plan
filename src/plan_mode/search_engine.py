"""Advanced plan-space search for plan_mode (v0.7.0: + evaluator-driven recombination).

Implements literature-grounded search over plan versions:

- MCTS with UCB1 selection and backprop (LATS 2310.04406; SYMPHONY
  2601.22623 multi-candidate expansion).
- LLM-driven expansion: a proposer model (DeepSeek) generates critique-
  addressing revisions of a node's plan (self-refinement, 2508.15501).
- Deterministic rule-based mutations as offline fallback (rubric-grounded
  template edits).
- Rollout = rubric score + structural verify + STRIPS simulation
  (SymPlanner 2505.01479, PlanBench 2409.13373).
- Optional external-judge evaluation at leaves (2510.03469).
- Transposition table (GATS 2607.08894): identical plans share a node.
- Cost/budget tracking (2505.14656); confidence-gated pruning of hopeless
  leaves (2602.08948).
- Beam search mode (best-of-N per level, 2601.17942 / 2509.00084).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

try:  # httpx is only needed for LLM expansion (online mode)
    import httpx
except ImportError:  # offline/rule-based search works without it
    httpx = None

# resolved lazily at call time to avoid circular imports
def _engine():
    from . import _load_rubric, _score, simulate, verify
    return _load_rubric, _score, simulate, verify


_BASE = "https://api.deepseek.com/anthropic"
_DEFAULT_MODEL = "deepseek-v4-flash"

ROLLOUT_WEIGHTS = (0.6, 0.2, 0.2)  # rubric, verify, simulation


def _resolve_api_key() -> str:
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        auth = json.loads((Path.home() / ".prime" / "agent" / "auth.json").read_text())
        cred = auth.get("deepseek") if isinstance(auth, dict) else None
        if isinstance(cred, dict) and cred.get("type") == "api_key":
            v = str(cred.get("key") or "").strip()
            if v and not v.startswith("!"):
                return os.environ.get(v) or v
    except (OSError, ValueError):
        pass
    return ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _hash(text: str) -> str:
    return hashlib.md5(_norm(text).encode("utf-8")).hexdigest()[:16]


def _rollout(plan_text: str, rubric: dict[str, Any]) -> dict[str, Any]:
    _load_rubric, _score, simulate, verify = _engine()
    res = _score(plan_text, rubric)
    v = verify(plan_text)
    sim = simulate(plan_text)
    value = (ROLLOUT_WEIGHTS[0] * res["score"] / 100.0
             + ROLLOUT_WEIGHTS[1] * (1.0 if v["ok"] else 0.0)
             + ROLLOUT_WEIGHTS[2] * (1.0 if sim["executable_plan"] else 0.0))
    return {"score": res["score"], "value": value,
            "verify_ok": v["ok"], "sim_ok": sim["executable_plan"],
            "critiques": res["critiques"]}




def feedback_penalty(plan_text: str, feedback: list[dict[str, Any]] | None) -> float:
    """FlowScout-style scalar penalty for execution-feedback disagreement."""
    if not feedback:
        return 0.0
    from .causal_validator import PlanParser
    try:
        ast = PlanParser.parse_plan(plan_text)
    except Exception:
        return 0.0
    actions = {a.id: a for a in ast.actions}
    penalty = 0.0
    for item in feedback:
        if not isinstance(item, dict):
            continue
        task_id = int(item.get("task_id") or item.get("task") or 0)
        action = actions.get(task_id)
        if action is None:
            continue
        missing = [str(x) for x in (item.get("missing_outputs") or [])]
        if missing and any(out not in action.outputs for out in missing):
            penalty += 0.10
        detail = str(item.get("detail") or "")
        if detail and "repair" not in action.name.lower():
            penalty += 0.05
    return min(0.30, penalty)


def _feedback_critiques(feedback: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not feedback:
        return []
    out: list[dict[str, str]] = []
    for item in feedback:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id") or item.get("task") or "?"
        detail = item.get("detail") or item.get("expected") or "execution feedback mismatch"
        out.append({"id": f"feedback:{task_id}", "section": "execution",
                    "hint": f"[execution feedback task {task_id}] {detail}"})
    return out


def _recombine(plan_a: str, plan_b: str) -> str | None:
    """Evaluator-driven recombination (Mind Evolution 2501.09891 via digest;
    diversity maintenance 2509.22613): join the first half of one plan with
    the second half of the other at a section boundary. Returns None when no
    safe splice point exists."""
    def half(text: str) -> tuple[list[str], list[str]]:
        lines = text.split("\n")
        k = len(lines) // 2
        for idx in range(k, min(len(lines) - 1, k + 40)):
            if lines[idx].startswith("#"):
                return lines[:idx], lines[idx:]
        return lines[:k], lines[k:]
    a_head, _ = half(plan_a)
    _, b_tail = half(plan_b)
    if len(a_head) < 4 or len(b_tail) < 4:
        return None
    cand = "\n".join(a_head + b_tail)
    return cand if _norm(cand) != _norm(plan_a) and _norm(cand) != _norm(plan_b) else None

# --- deterministic mutation library (offline expansion) --------------------

_MUTATIONS = [
    ("add-fallback", lambda t: t + "\n\n## Risks\n- Risk: step fails; type: execution-defect; recovery scope: local fix; mitigation: retry with backoff; fallback path: report to the user instead of failing silently. Rollback: revert the last change."),
    ("add-replan-budget", lambda t: t + "\n\n## Replan\nTriggers: T1 step fails -> replan that step only; scoped repair: patch the failing step, reuse the valid prefix. Acceptance threshold: stop when score >= 90. Replan budget: cap 3 revisions."),
    ("add-verification-machine", lambda t: t + "\n\n## Verification machine\nInvariant: every output file exists with expected size. Checked by a deterministic non-LLM script (ls + wc -c). External checker: pytest."),
    ("add-grounding", lambda t: t + "\n\n## Grounding\nHow we detect step success/failure: each step writes a named file; success = file exists; failure = missing/empty file. Restate the plan state after each step. Silent-failure detection: empty output = failed step, re-run once."),
    ("add-uncertainty", lambda t: t + "\n\n## Uncertainty\nUncertainty per step: low for file ops, medium for external services. Conservative switch: if > 5 failures, stop and report instead of proceeding."),
    ("add-escalation", lambda t: t + "\n\n## Escalation\nCheapest first: attempt the cheapest viable action first. Escalate to expensive steps only when needed."),
    ("add-memory", lambda t: t + "\n\n## Memory\nLessons learned from plan history: cite the evidence of each prior failure. Revision strategy: address every critique id explicitly. Reusable core: the assess-revise loop transfers to any objective."),
    ("add-constraints", lambda t: t + "\n\n## Constraints\nPreconditions per step, re-anchored at each step. Dual-correction: logical consistency AND physical feasibility of every step."),
    ("add-structure", lambda t: t + "\n\n## Structure\nLayered: top-level named sub-plans with a detail block each. Pseudocode: step1(); step2(); step3()."),
]


_SECTION_TEMPLATES = {
    "risks": "\n\n## Risks\n- Risk list: (name the top failure); failure classification: local fix vs global replan; mitigation: (one concrete action); fallback path: if the step fails, (the alternative). Rollback: revert the change.",
    "replan": "\n\n## Replan\nTriggers: event-driven, threshold-driven: a gate fails -> replan. Scoped repair: patch the failing step, reuse the valid plan prefix. Acceptance threshold: stop when (criterion). Replan budget: cap 3 revisions.",
    "verification": "\n\n## Verification\nEvaluation function: score each step (rubric 0-100). Dense per-step feedback: re-score immediately after each task. Verification step per milestone. Revision loop iterates on verified failures only.",
    "grounding": "\n\n## Grounding\nDetect step success by the output file plus a count or exit code (how we know each step succeeded). Restate state after each step. Silent-failure detection: empty output = failed step.",
    "constraints": "\n\n## Constraints\nPreconditions per step, re-anchored at the step where they apply. Dual correction: logical consistency AND physical feasibility. Multi-agent coordination: planner, executor, and an independent judge; handoffs are the output files.",
    "memory": "\n\n## Memory\nLessons learned from past failures; revision strategy: address every critique id; best-of-N via assess_candidates; evidence-traced revisions; fold/prune history.",
    "escalation": "\n\n## Escalation\nCheapest viable action first; escalate only if needed. Isolate the root cause of a failing gate before symptom edits.",
    "objective": "\n\n## Objective\nGoal restated in one line. In scope: (what we will deliver). Out of scope (non-goals): (what we will not do).",
    "success": "\n\n## Success criteria\n- S1: (numeric or pass/fail criterion); verifiable by (command/file). Deadline: within 1 day. Acceptance criteria: pass/fail on S1.",
    "resources": "\n\n## Resources\nTime estimates: 20 min per task. Budget: (cost/tokens). Cost reconciliation: cumulative total checked against the budget.",
    "executability": "\n\n## Executability\nFirst action today: (the named first step). Stop when (exit criterion). Refusal policy: abort if (infeasibility condition).",
    "milestones": "\n\n## Milestones\n- M1 after task 1: measurable signal: (what to check); decision rule: halt/revise; go/no-go on failure. Rollback: revert the last change.",

    "hardening": "\n\n## Hardening\nExecution feedback updates the world model, closing the loop. Detect the error immediately at each step. Avoid revisiting the same failed state.",
}


def _mutations(plan_text: str, width: int,
                critiques: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Critique-aware mutations (2512.09629, 2606.21740): when a rollout
    carries critique ids, mutate toward the sections those critiques name
    instead of firing blind boilerplate."""
    rng = random.Random(hash(_norm(plan_text)) & 0xFFFFFFFF)  # deterministic per plan
    targeted: list[tuple[str, Any]] = []
    if critiques:
        # critique ids are "<section>:<hint>"; map section -> template
        seen: set[str] = set()
        for c in critiques:
            sec = c.get("id", "").split(":", 1)[0] if isinstance(c, dict) else str(c).split(":", 1)[0]
            for key, tmpl in _SECTION_TEMPLATES.items():
                if key in sec and key not in seen:
                    seen.add(key)
                    targeted.append((f"target-{key}", lambda t, tmpl=tmpl: t + tmpl))
    if targeted:
        chosen = rng.sample(targeted, min(width, len(targeted)))
        return [{"text": f(plan_text), "note": name} for name, f in chosen]
    chosen = rng.sample(_MUTATIONS, min(width, len(_MUTATIONS)))
    return [{"text": f(plan_text), "note": name} for name, f in chosen]


# --- LLM proposer (expansion via a generator model) ------------------------

PROPOSER_SYSTEM = (
    "You are the expansion policy of a planning search engine. Given a plan "
    "and its critiques, produce distinct improved revisions. Each revision "
    "must be a COMPLETE plan (not a diff) and must fix at least one critique "
    "without breaking the plan's task chain. Do not remove task outputs. "
    "Respond ONLY with a JSON object: {\"variants\": [\"<full plan>\", ...]}."
)


async def _propose(plan_text: str, critiques: list[dict[str, str]], width: int,
                   api_key: str, model: str, timeout: int = 180) -> tuple[list[str], int]:
    crits = "\n".join(f"- {c['hint']}" for c in critiques[:12]) or "(none)"
    user = f"PLAN:\n{plan_text}\n\nCRITIQUES:\n{crits}\n\nProduce {width} distinct improved revisions as JSON."
    body = {
        "model": model,
        "max_tokens": 8192,
        "thinking": {"type": "disabled"},
        "system": PROPOSER_SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{_BASE}/v1/messages", json=body, headers=headers)
        if resp.status_code != 200:
            return [], 0
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text").strip()
        if not text:
            return [], 0
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        parsed = json.loads(text)
        variants = parsed.get("variants", [])
        usage = (data.get("usage") or {}).get("output_tokens", 0) + (data.get("usage") or {}).get("input_tokens", 0)
        return [v for v in variants if isinstance(v, str) and v.strip()][:width], usage
    except Exception:
        return [], 0


# --- tree state ------------------------------------------------------------

def _tree(session: dict[str, Any]) -> dict[str, Any]:
    return session.setdefault("search_tree",
                              {"nodes": {}, "transposition": {}, "root": None,
                               "best_node": None, "best_value": 0.0,
                               "expansions": 0, "rollouts": 0,
                               "tokens_used": 0, "pruned": []})


def _new_node(t: dict[str, Any], plan_text: str, parent: str | None,
              depth: int, note: str | None, rollout: dict[str, Any]) -> str:
    h = _hash(plan_text)
    if h in t["transposition"]:
        node_id = t["transposition"][h]
        t["nodes"][node_id]["visits"] += 1
        if parent and node_id not in t["nodes"][parent]["children"]:
            t["nodes"][parent]["children"].append(node_id)
        return node_id
    nid = f"n{len(t['nodes']) + 1}"
    t["nodes"][nid] = {
        "id": nid, "parent": parent, "depth": depth, "note": note,
        "plan_text": plan_text, "score": rollout["score"],
        "value": rollout["value"], "q": rollout["value"], "visits": 1,
        "verify_ok": rollout["verify_ok"], "sim_ok": rollout["sim_ok"],
        "critiques": rollout["critiques"], "children": [],
    }
    t["transposition"][h] = nid
    if parent:
        t["nodes"][parent]["children"].append(nid)
    return nid


def _backprop(t: dict[str, Any], node_id: str, value: float) -> None:
    nodes = t["nodes"]
    while node_id:
        n = nodes[node_id]
        n["visits"] += 1
        n["q"] = (n["q"] * (n["visits"] - 1) + value) / n["visits"]
        node_id = n.get("parent")
    best = max(nodes.values(), key=lambda x: (x["value"], x["q"]))
    t["best_node"] = best["id"]
    t["best_value"] = best["value"]


def _ucb(t: dict[str, Any], node: dict[str, Any], exploration: float,
         cost_penalty: float) -> float:
    N = max(1, sum(n["visits"] for n in t["nodes"].values()))
    return (node["q"]
            + exploration * (__import__("math").log(N) / max(1, node["visits"])) ** 0.5
            - cost_penalty * node["visits"])


def _select(t: dict[str, Any], exploration: float, cost_penalty: float) -> str:
    cur = t["root"]
    while t["nodes"][cur]["children"]:
        kids = [c for c in t["nodes"][cur]["children"] if c in t["nodes"]]
        if not kids:
            break
        cur = max(kids, key=lambda c: _ucb(t, t["nodes"][c], exploration, cost_penalty))
    return cur


def _prune(t: dict[str, Any], margin: float) -> None:
    nodes = t["nodes"]
    if not nodes:
        return
    best_q = max(n["q"] for n in nodes.values())
    for nid, n in list(nodes.items()):
        if n["visits"] > 1 and not n["children"] and n["q"] < best_q - margin:
            t["pruned"].append(nid)
            parent = n.get("parent")
            if parent and parent in nodes and nid in nodes[parent]["children"]:
                nodes[parent]["children"].remove(nid)
            del nodes[nid]


# --- public search ---------------------------------------------------------

async def search(session: dict[str, Any] | str, *,
                 iterations: int = 4, width: int = 2,
                 exploration: float = 1.4, cost_penalty: float = 0.0,
                 mode: str = "mcts",            # "mcts" | "beam" | "ast"
                 expansion: str = "llm",        # "llm" | "rules"
                 judge_evals: bool = False,
                 max_nodes: int = 64,
                 depth: int = 2, beam_width: int = 3,
                 prune_margin: float | None = 0.15,
                 model: str | None = None,
                 root_plan: str | None = None,
                 cwd: str | Path | None = None,
                 checkpoint_before: bool = False,
                 execution_feedback: list[dict[str, Any]] | None = None,
                 plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Run a search over plan space and return the best plan found.

    mcts mode: UCB1-selected node is expanded each iteration; rollouts are
    rubric+verify+simulation; values back-propagated. beam mode: level-wise
    expansion keeping top beam_width plans per level.
    """
    from . import DEFAULT_PLANS_DIR, _load_session, _save_session
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    rubric = s.get("rubric_snapshot") or _engine()[0]()
    t = _tree(s)
    nodes = t["nodes"]

    if root_plan is None:
        rounds = s.get("rounds", [])
        root_plan = rounds[-1]["plan_text"] if rounds else None
        if root_plan is None and nodes:
            root_plan = nodes[t["root"]]["plan_text"]
    if root_plan is None:
        raise ValueError("search needs a root plan (pass root_plan or run assess first)")

    if t["root"] is None or nodes.get(t["root"], {}).get("plan_text") != root_plan:
        ro = _rollout(root_plan, rubric)
        root_id = _new_node(t, root_plan, None, 0, "root", ro)
        t["root"] = root_id
        t["best_node"] = root_id
        t["best_value"] = ro["value"]

    if checkpoint_before:
        from . import checkpoint as session_checkpoint
        session_checkpoint(s, plans_dir=plans_dir, note=f"search:{mode}:pre-expansion")

    # seed pool (2605.21902, 2605.06957): reuse validated plans from prior
    # finished sessions in this plans_dir as extra root variants (cap 3)
    if t.get("seeded") is None:
        t["seeded"] = True
        candidates: list[tuple[int, str]] = []
        try:
            cur_tokens = set(re.findall(r"[a-z]{4,}", s.get("objective", "").lower()))
            for p in sorted(Path(plans_dir).glob("*.json")):
                try:
                    s2 = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                if s2.get("status") == "finished" and s2.get("best_version") and s2.get("rounds"):
                    pt = s2["rounds"][s2["best_version"] - 1].get("plan_text", "")
                    if pt and pt != root_plan and len(pt) > 200:
                        # family match (2605.21902): prefer seeds whose
                        # objective shares tokens with the current objective
                        fam = set(re.findall(r"[a-z]{4,}", s2.get("objective", "").lower()))
                        candidates.append((len(cur_tokens & fam), pt))
        except OSError:
            pass
        candidates.sort(key=lambda x: -x[0])
        seeds = [pt for _, pt in candidates[:3]]
        for pt in seeds:
            ro = _rollout(pt, rubric)
            sid = _new_node(t, pt, t["root"], 1, "seed", ro)
            t["rollouts"] += 1

    api_key = _resolve_api_key()
    # reactive coeffect (Cordis notify): the llm_expansion dependency spec, not
    # a hard-coded key check, decides whether LLM expansion can activate
    try:
        from . import deps_check
        if deps_check()["status"].get("llm_expansion") != "satisfied":
            api_key = ""
    except Exception:
        pass
    model = model or os.environ.get("PLAN_SEARCH_MODEL", "").strip() or _DEFAULT_MODEL
    t0 = time.time()

    if mode in ("ast", "evolutionary"):
        from .ast_search import ASTSearchEngine, PlanParser, apply_execution_feedback
        from . import ground_check
        gc = ground_check(root_plan, cwd=cwd or Path.cwd())
        initial_state = set(gc.get("verified", []))
        engine = ASTSearchEngine(objective=s.get("objective", ""), initial_state=initial_state, source_plan_text=root_plan)
        root_ast = PlanParser.parse_plan(root_plan, objective=s.get("objective", ""))
        root_member = engine.evaluate_ast(root_ast, lambda pt: _rollout(pt, rubric)["score"], source_plan_text=root_plan)
        engine.population = [root_member]
        if execution_feedback:
            repaired_ast = apply_execution_feedback(root_ast, execution_feedback)
            repaired_member = engine.evaluate_ast(repaired_ast, lambda pt: _rollout(pt, rubric)["score"], source_plan_text=root_plan)
            engine.population.append(repaired_member)

        for iter_idx in range(iterations):
            evolved = engine.evolve_step(population_size=max(2, width), rubric_score_fn=lambda pt: _rollout(pt, rubric)["score"])
            for member in evolved:
                if len(nodes) >= max_nodes:
                    break
                ro = _rollout(member.plan_text, rubric)
                nid = _new_node(t, member.plan_text, t["root"], 1, f"ast_evolve_iter_{iter_idx}", ro)
                t["rollouts"] += 1
                if ro["value"] > t["best_value"]:
                    t["best_value"] = ro["value"]
                    t["best_node"] = nid

    elif mode == "beam":
        frontier = [t["root"]]
        for level in range(depth):
            next_frontier: list[str] = []
            for nid in frontier:
                if len(nodes) >= max_nodes:
                    break
                plan = nodes[nid]["plan_text"]
                crits = nodes[nid]["critiques"] + _feedback_critiques(execution_feedback)
                if expansion == "llm" and api_key:
                    variants, tok = await _propose(plan, crits, width, api_key, model)
                    t["tokens_used"] += tok
                    variants = variants or [m["text"] for m in _mutations(plan, width, crits)]
                else:
                    variants = [m["text"] for m in _mutations(plan, width, crits)]
                for v in variants:
                    ro = _rollout(v, rubric)
                    ro["value"] = max(0.0, ro["value"] - feedback_penalty(v, execution_feedback))
                    ro["feedback_penalty"] = feedback_penalty(v, execution_feedback)
                    cid = _new_node(t, v, nid, level + 1, "beam", ro)
                    next_frontier.append(cid)
                    t["rollouts"] += 1
            if not next_frontier:
                break
            next_frontier.sort(key=lambda c: -nodes[c]["value"])
            frontier = next_frontier[:beam_width]
            t["expansions"] += 1
        best = max(nodes.values(), key=lambda n: (n["value"], n["q"]))
        t["best_node"], t["best_value"] = best["id"], best["value"]
    else:  # mcts
        prev_best = t["best_value"]
        plateau = 0
        start_width = width
        for it in range(iterations):
            if len(nodes) >= max_nodes:
                break
            sel = _select(t, exploration, cost_penalty)
            plan = nodes[sel]["plan_text"]
            crits = nodes[sel]["critiques"] + _feedback_critiques(execution_feedback)
            if expansion == "llm" and api_key:
                variants, tok = await _propose(plan, crits, width, api_key, model)
                t["tokens_used"] += tok
                if not variants:
                    variants = [m["text"] for m in _mutations(plan, width, crits)]
            else:
                variants = [m["text"] for m in _mutations(plan, width, crits)]
            # recombination: when a sibling's value is close to the selected
            # node's, add a crossover variant (evaluator-driven, 2509.22613)
            sib_vals = [(n["value"], n["plan_text"]) for n in nodes.values()
                        if n.get("parent") == nodes[sel].get("parent") and n["id"] != sel]
            if sib_vals:
                close = [txt for val, txt in sib_vals if abs(val - nodes[sel]["value"]) < 0.05]
                if close:
                    xo = _recombine(plan, close[0])
                    if xo:
                        variants = variants[:width] + [xo]
            values = []
            for v in variants:
                ro = _rollout(v, rubric)
                ro["value"] = max(0.0, ro["value"] - feedback_penalty(v, execution_feedback))
                ro["feedback_penalty"] = feedback_penalty(v, execution_feedback)
                cid = _new_node(t, v, sel, nodes[sel]["depth"] + 1, f"mcts it{it}", ro)
                t["rollouts"] += 1
                if judge_evals and api_key:
                    from . import judge
                    jr = await judge(v, s.get("objective", ""), model=model)
                    if jr.get("ok") and jr.get("verdict") == "go":
                        ro["value"] = min(1.0, ro["value"] + 0.05)
                # nested rollout (2511.21706): deepen a variant that clearly
                # beats its parent; stop deepening when the gain saturates
                gain = ro["value"] - nodes[sel]["value"]
                if gain >= 0.01 and nodes[sel]["depth"] + 2 <= depth + 1:
                    ro2 = _rollout(v, rubric)
                    if ro2["value"] >= ro["value"] - 0.005:
                        ro = ro2
                        t["rollouts"] += 1
                values.append((cid, ro["value"]))
                _backprop(t, cid, ro["value"])
            if prune_margin is not None:
                _prune(t, prune_margin)
            t["expansions"] += 1
            # adaptive search effort (LFS 2506.05213): escalate the expansion
            # width when the best value plateaus for 2 consecutive expansions
            if abs(t["best_value"] - prev_best) < 0.01:
                plateau += 1
                if plateau >= 2 and width < start_width + 2:
                    width += 1
                    plateau = 0
                    t.setdefault("escalations", []).append(
                        {"iteration": it, "width": width,
                         "best_value": round(t["best_value"], 3)})
            else:
                plateau = 0
            prev_best = t["best_value"]

    _save_session(plans_dir, s)
    best_node = nodes[t["best_node"]]
    s.setdefault("search_log", []).append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode, "iterations": iterations, "width": width,
        "nodes": len(nodes), "rollouts": t["rollouts"],
        "tokens_used": t["tokens_used"], "seconds": round(time.time() - t0, 1),
        "best_score": best_node["score"], "best_value": round(best_node["value"], 3),
        "escalations": len(t.get("escalations", [])),
    })

    # Automatically commit the best plan found by search to the session history
    from . import assess
    if best_node["score"] > (s.get("best_score") or 0) or best_node["plan_text"] != root_plan:
        assess(s, best_node["plan_text"], note=f"search:{mode}:score_{best_node['score']}", plans_dir=plans_dir)
    else:
        _save_session(plans_dir, s)
    return {
        "best_plan": best_node["plan_text"],
        "best_score": best_node["score"],
        "best_value": round(best_node["value"], 3),
        "best_node": best_node["id"],
        "nodes": len(nodes), "rollouts": t["rollouts"],
        "tokens_used": t["tokens_used"],
        "mode": mode, "escalations": t.get("escalations", []),
        "tree": t,
        "report": _report(t),
    }


def _report(t: dict[str, Any]) -> dict[str, Any]:
    nodes = t["nodes"]
    depths = {nid: n["depth"] for nid, n in nodes.items()}
    return {"nodes": len(nodes), "expansions": t["expansions"],
            "root": t["root"], "best_node": t["best_node"],
            "best_value": round(t["best_value"], 3),
            "max_depth": max(depths.values()) if depths else 0,
            "leaves": [nid for nid, n in nodes.items() if not n["children"]],
            "pruned": t["pruned"]}
