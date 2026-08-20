"""Bounded plan-space search for plan_mode.

The public default is deliberately local and deterministic. Online LLM
expansion is opt-in because network/provider calls are unbounded relative to
local search and must inherit the active implementation model/thinking profile
rather than silently switching providers.

Supported search modes:
- AST/evolutionary search (default)
- MCTS with UCB1 selection/backprop
- Beam search

Search rollouts combine rubric score, structural verification, and simulation.
Execution feedback can repair/penalize candidates. LLM expansion remains
available only when explicitly requested and supplied a runtime proposer (or
an explicitly enabled legacy DeepSeek proposer).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # online legacy expansion only
    import httpx
except ImportError:  # pragma: no cover - local search does not need httpx
    httpx = None


def _engine():
    from . import _load_rubric, _score, simulate, verify
    return _load_rubric, _score, simulate, verify


_LEGACY_DEEPSEEK_BASE = "https://api.deepseek.com/anthropic"
_LEGACY_DEEPSEEK_MODEL = "deepseek-v4-flash"
ROLLOUT_WEIGHTS = (0.6, 0.2, 0.2)


class SearchTimeoutError(TimeoutError):
    """A bounded search/proposal exceeded its explicit wall-clock budget."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _hash(text: str) -> str:
    return hashlib.md5(_norm(text).encode("utf-8")).hexdigest()[:16]


def _rollout(plan_text: str, rubric: dict[str, Any]) -> dict[str, Any]:
    _load_rubric, _score, simulate, verify = _engine()
    res = _score(plan_text, rubric)
    v = verify(plan_text)
    sim = simulate(plan_text)
    value = (
        ROLLOUT_WEIGHTS[0] * res["score"] / 100.0
        + ROLLOUT_WEIGHTS[1] * (1.0 if v["ok"] else 0.0)
        + ROLLOUT_WEIGHTS[2] * (1.0 if sim["executable_plan"] else 0.0)
    )
    return {
        "score": res["score"],
        "value": value,
        "verify_ok": v["ok"],
        "sim_ok": sim["executable_plan"],
        "critiques": res["critiques"],
    }


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
        out.append({
            "id": f"feedback:{task_id}",
            "section": "execution",
            "hint": f"[execution feedback task {task_id}] {detail}",
        })
    return out


def _recombine(plan_a: str, plan_b: str) -> str | None:
    """Join compatible halves of two sibling plans at a section boundary."""
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
    return cand if _norm(cand) not in {_norm(plan_a), _norm(plan_b)} else None


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
    """Critique-aware deterministic mutation library."""
    rng = random.Random(hash(_norm(plan_text)) & 0xFFFFFFFF)
    targeted: list[tuple[str, Any]] = []
    if critiques:
        seen: set[str] = set()
        for c in critiques:
            sec = c.get("id", "").split(":", 1)[0] if isinstance(c, dict) else str(c).split(":", 1)[0]
            for key, tmpl in _SECTION_TEMPLATES.items():
                if key in sec and key not in seen:
                    seen.add(key)
                    targeted.append((f"target-{key}", lambda t, tmpl=tmpl: t + tmpl))
    if targeted:
        chosen = rng.sample(targeted, min(width, len(targeted)))
        return [{"text": fn(plan_text), "note": name} for name, fn in chosen]
    chosen = rng.sample(_MUTATIONS, min(width, len(_MUTATIONS)))
    return [{"text": fn(plan_text), "note": name} for name, fn in chosen]


PROPOSER_SYSTEM = (
    "You are the expansion policy of a planning search engine. Given a plan "
    "and its critiques, produce distinct improved revisions. Each revision "
    "must be a COMPLETE plan (not a diff) and must fix at least one critique "
    "without breaking the plan's task chain. Do not remove task outputs. "
    "Respond ONLY with a JSON object: {\"variants\": [\"<full plan>\", ...]}."
)


def _resolve_deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    try:
        auth = json.loads((Path.home() / ".prime" / "agent" / "auth.json").read_text())
        cred = auth.get("deepseek") if isinstance(auth, dict) else None
        if isinstance(cred, dict) and cred.get("type") == "api_key":
            value = str(cred.get("key") or "").strip()
            if value and not value.startswith("!"):
                return os.environ.get(value) or value
    except (OSError, ValueError):
        pass
    return ""


def _deepseek_thinking_body(profile: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(profile.get("mode", "default"))
    if mode == "default":
        return {}
    budget = profile.get("thinking_budget")
    if budget is not None:
        raise ValueError("legacy DeepSeek proposer cannot faithfully reproduce an exact thinking_budget")
    level = profile.get("reasoning_effort", profile.get("thinking_level"))
    if level is None:
        raise ValueError("thinking profile has no enforceable DeepSeek effort")
    level = str(level).lower()
    if level in {"off", "disabled", "none"}:
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "enabled"}, "reasoning_effort": level}


async def _legacy_deepseek_propose(
    plan_text: str,
    critiques: list[dict[str, str]],
    width: int,
    *,
    api_key: str,
    model: str,
    thinking_profile: Mapping[str, Any],
    request_timeout_seconds: float,
) -> tuple[list[str], int]:
    """Legacy hosted DeepSeek proposer, available only by explicit opt-in."""
    if httpx is None:
        raise RuntimeError("httpx is required for legacy DeepSeek LLM expansion")
    if not model.lower().startswith("deepseek"):
        raise ValueError("legacy DeepSeek proposer cannot be used for a non-DeepSeek implementation model")
    crits = "\n".join(f"- {c['hint']}" for c in critiques[:12]) or "(none)"
    user = f"PLAN:\n{plan_text}\n\nCRITIQUES:\n{crits}\n\nProduce {width} distinct improved revisions as JSON."
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "system": PROPOSER_SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }
    body.update(_deepseek_thinking_body(thinking_profile))
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    timeout = httpx.Timeout(request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{_LEGACY_DEEPSEEK_BASE}/v1/messages", json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    text = "".join(
        block.get("text", "") for block in data.get("content", [])
        if block.get("type") == "text"
    ).strip()
    if not text:
        raise RuntimeError("LLM proposer returned an empty response")
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    parsed = json.loads(text)
    variants = [v for v in parsed.get("variants", []) if isinstance(v, str) and v.strip()][:width]
    if not variants:
        raise RuntimeError("LLM proposer returned no valid variants")
    usage = (data.get("usage") or {})
    tokens = int(usage.get("output_tokens", 0) or 0) + int(usage.get("input_tokens", 0) or 0)
    return variants, tokens


async def _invoke_runtime_proposer(
    proposer: Callable[..., Any],
    *,
    plan_text: str,
    critiques: list[dict[str, str]],
    width: int,
    model: str,
    thinking_profile: Mapping[str, Any],
) -> tuple[list[str], int]:
    result = proposer(
        plan_text=plan_text,
        critiques=critiques,
        width=width,
        model=model,
        thinking_profile=dict(thinking_profile),
    )
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, tuple) and len(result) == 2:
        variants, tokens = result
    else:
        variants, tokens = result, 0
    variants = [v for v in list(variants or []) if isinstance(v, str) and v.strip()][:width]
    if not variants:
        raise RuntimeError("runtime LLM proposer returned no valid variants")
    return variants, int(tokens or 0)


def _fresh_tree() -> dict[str, Any]:
    return {
        "nodes": {},
        "transposition": {},
        "root": None,
        "best_node": None,
        "best_value": 0.0,
        "expansions": 0,
        "rollouts": 0,
        "tokens_used": 0,
        "pruned": [],
        "warnings": [],
    }


def _tree(session: dict[str, Any]) -> dict[str, Any]:
    return session.setdefault("search_tree", _fresh_tree())


def _new_node(t: dict[str, Any], plan_text: str, parent: str | None,
              depth: int, note: str | None, rollout: dict[str, Any]) -> str:
    h = _hash(plan_text)
    if h in t["transposition"] and t["transposition"][h] in t["nodes"]:
        node_id = t["transposition"][h]
        t["nodes"][node_id]["visits"] += 1
        if parent and node_id not in t["nodes"][parent]["children"]:
            t["nodes"][parent]["children"].append(node_id)
        return node_id
    nid = f"n{len(t['nodes']) + 1}"
    t["nodes"][nid] = {
        "id": nid,
        "parent": parent,
        "depth": depth,
        "note": note,
        "plan_text": plan_text,
        "score": rollout["score"],
        "value": rollout["value"],
        "q": rollout["value"],
        "visits": 1,
        "verify_ok": rollout["verify_ok"],
        "sim_ok": rollout["sim_ok"],
        "critiques": rollout["critiques"],
        "children": [],
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
    return (
        node["q"]
        + exploration * (__import__("math").log(N) / max(1, node["visits"])) ** 0.5
        - cost_penalty * node["visits"]
    )


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


def _report(t: dict[str, Any]) -> dict[str, Any]:
    nodes = t["nodes"]
    depths = {nid: n["depth"] for nid, n in nodes.items()}
    return {
        "nodes": len(nodes),
        "expansions": t["expansions"],
        "root": t["root"],
        "best_node": t["best_node"],
        "best_value": round(t["best_value"], 3),
        "max_depth": max(depths.values()) if depths else 0,
        "leaves": [nid for nid, n in nodes.items() if not n["children"]],
        "pruned": t["pruned"],
        "warnings": list(t.get("warnings", [])),
    }


def _emit_progress(callback: Callable[[dict[str, Any]], Any] | None, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        return


async def search(
    session: dict[str, Any] | str,
    *,
    iterations: int = 4,
    width: int = 2,
    exploration: float = 1.4,
    cost_penalty: float = 0.0,
    mode: str = "ast",                 # safe default: local deterministic search
    expansion: str = "rules",          # online LLM expansion requires explicit opt-in
    judge_evals: bool = False,
    max_nodes: int = 64,
    depth: int = 2,
    beam_width: int = 3,
    prune_margin: float | None = 0.15,
    model: str | None = None,           # backward-compatible alias for implementation_model
    implementation_model: str | None = None,
    implementation_thinking: Any = None,
    llm_proposer: Callable[..., Any] | None = None,
    allow_legacy_deepseek_expansion: bool = False,
    proposal_timeout_seconds: float = 60.0,
    search_timeout_seconds: float = 120.0,
    skip_if_converged: bool = True,
    convergence_score: float = 99.0,
    root_plan: str | None = None,
    cwd: str | Path | None = None,
    checkpoint_before: bool = False,
    execution_feedback: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    plans_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a bounded search and return the best plan found.

    A bare call is intentionally safe: AST search with deterministic rule
    expansion, no network. Set ``mode='mcts'``/``'beam'`` explicitly for those
    algorithms. Set ``expansion='llm'`` only when a runtime proposer is
    supplied; that proposer receives the same implementation model/thinking
    profile recorded by Prime.

    ``search_timeout_seconds`` is a hard wall-clock budget between search
    steps. Each online proposal is additionally wrapped in ``asyncio.wait_for``
    with ``proposal_timeout_seconds`` (or the remaining total budget,
    whichever is smaller). On a proposal error/timeout, Prime records a warning
    and falls back to deterministic mutations rather than hanging silently.
    """
    if iterations < 0 or width < 1 or max_nodes < 1:
        raise ValueError("iterations must be >= 0; width/max_nodes must be >= 1")
    if mode not in {"ast", "evolutionary", "mcts", "beam"}:
        raise ValueError(f"unsupported search mode: {mode}")
    if expansion not in {"rules", "llm"}:
        raise ValueError(f"unsupported expansion mode: {expansion}")
    if proposal_timeout_seconds <= 0 or search_timeout_seconds <= 0:
        raise ValueError("search/proposal timeouts must be > 0")

    from . import DEFAULT_PLANS_DIR, _load_session, _save_session

    started = time.monotonic()
    deadline = started + float(search_timeout_seconds)

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    def expired() -> bool:
        return remaining() <= 0.0

    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)

    rubric = s.get("rubric_snapshot") or _engine()[0]()
    if root_plan is None:
        rounds = s.get("rounds", [])
        root_plan = rounds[-1]["plan_text"] if rounds else None
    if root_plan is None:
        raise ValueError("search needs a root plan (pass root_plan or run assess first)")

    root_rollout = _rollout(root_plan, rubric)

    # A changed root invalidates the persisted transposition tree. Keeping the
    # old tree caused stale-node selection across repeated search calls.
    old_tree = s.get("search_tree")
    old_root_text = None
    if isinstance(old_tree, dict):
        old_root_id = old_tree.get("root")
        old_root_text = (old_tree.get("nodes") or {}).get(old_root_id, {}).get("plan_text")
    if old_root_text != root_plan:
        s["search_tree"] = _fresh_tree()
    t = _tree(s)
    nodes = t["nodes"]
    if t["root"] is None:
        root_id = _new_node(t, root_plan, None, 0, "root", root_rollout)
        t["root"] = root_id
        t["best_node"] = root_id
        t["best_value"] = root_rollout["value"]

    # Do not spend test-time compute on a plan that is already essentially
    # perfect and passes both deterministic structural checks and simulation.
    if (
        skip_if_converged
        and root_rollout["score"] >= float(convergence_score)
        and root_rollout["verify_ok"]
        and root_rollout["sim_ok"]
    ):
        elapsed = round(time.monotonic() - started, 3)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode,
            "iterations": 0,
            "width": width,
            "nodes": len(nodes),
            "rollouts": t["rollouts"],
            "tokens_used": t["tokens_used"],
            "seconds": elapsed,
            "best_score": root_rollout["score"],
            "best_value": round(root_rollout["value"], 3),
            "termination_reason": "already-converged",
        }
        s.setdefault("search_log", []).append(entry)
        _save_session(plans_dir, s)
        return {
            "best_plan": root_plan,
            "best_score": root_rollout["score"],
            "best_value": round(root_rollout["value"], 3),
            "best_node": t["root"],
            "nodes": len(nodes),
            "rollouts": t["rollouts"],
            "tokens_used": t["tokens_used"],
            "mode": mode,
            "expansion": expansion,
            "termination_reason": "already-converged",
            "timed_out": False,
            "escalations": t.get("escalations", []),
            "tree": t,
            "report": _report(t),
        }

    if checkpoint_before:
        from . import checkpoint as session_checkpoint
        session_checkpoint(s, plans_dir=plans_dir, note=f"search:{mode}:pre-expansion")

    # Seed pool is local but bounded by the same wall-clock budget.
    if t.get("seeded") is None and not expired():
        t["seeded"] = True
        candidates: list[tuple[int, str]] = []
        try:
            cur_tokens = set(re.findall(r"[a-z]{4,}", s.get("objective", "").lower()))
            for p in sorted(Path(plans_dir).glob("*.json")):
                if expired():
                    break
                try:
                    s2 = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                if s2.get("status") == "finished" and s2.get("best_version") and s2.get("rounds"):
                    pt = s2["rounds"][s2["best_version"] - 1].get("plan_text", "")
                    if pt and pt != root_plan and len(pt) > 200:
                        fam = set(re.findall(r"[a-z]{4,}", s2.get("objective", "").lower()))
                        candidates.append((len(cur_tokens & fam), pt))
        except OSError:
            pass
        candidates.sort(key=lambda x: -x[0])
        for _, pt in candidates[:3]:
            if expired():
                break
            ro = _rollout(pt, rubric)
            _new_node(t, pt, t["root"], 1, "seed", ro)
            t["rollouts"] += 1

    active_model: str | None = None
    thinking_profile: dict[str, Any] = {"mode": "default"}
    legacy_key = ""
    if expansion == "llm":
        from .self_verification import resolve_implementation_model, resolve_implementation_thinking

        active_model = resolve_implementation_model(
            implementation_model or model,
            session=s,
        )
        thinking_profile = resolve_implementation_thinking(
            implementation_thinking,
            session=s,
        )
        if not active_model:
            raise ValueError(
                "LLM search expansion requires the active implementation model; "
                "Prime will not silently choose a verifier/search model"
            )
        if llm_proposer is None:
            if not allow_legacy_deepseek_expansion:
                raise ValueError(
                    "LLM search expansion requires an explicit runtime llm_proposer. "
                    "This prevents plan.search() from silently switching to DeepSeek."
                )
            if not active_model.lower().startswith("deepseek"):
                raise ValueError(
                    "legacy DeepSeek expansion cannot preserve a non-DeepSeek implementation model"
                )
            legacy_key = _resolve_deepseek_api_key()
            if not legacy_key:
                raise ValueError("legacy DeepSeek expansion explicitly enabled but DEEPSEEK_API_KEY is unavailable")

    async def variants_for(plan: str, crits: list[dict[str, str]], current_width: int) -> list[str]:
        if expansion != "llm":
            return [m["text"] for m in _mutations(plan, current_width, crits)]
        wait_budget = min(float(proposal_timeout_seconds), remaining())
        if wait_budget <= 0:
            raise SearchTimeoutError("search wall-clock budget exhausted before LLM proposal")
        _emit_progress(progress_callback, {
            "event": "proposal-start",
            "model": active_model,
            "width": current_width,
            "remaining_seconds": round(remaining(), 2),
        })
        try:
            if llm_proposer is not None:
                call = _invoke_runtime_proposer(
                    llm_proposer,
                    plan_text=plan,
                    critiques=crits,
                    width=current_width,
                    model=active_model or "",
                    thinking_profile=thinking_profile,
                )
            else:
                call = _legacy_deepseek_propose(
                    plan,
                    crits,
                    current_width,
                    api_key=legacy_key,
                    model=active_model or _LEGACY_DEEPSEEK_MODEL,
                    thinking_profile=thinking_profile,
                    request_timeout_seconds=wait_budget,
                )
            variants, tok = await asyncio.wait_for(call, timeout=wait_budget)
            t["tokens_used"] += tok
            return variants
        except (asyncio.TimeoutError, TimeoutError) as exc:
            warning = f"LLM proposal timed out after {wait_budget:.1f}s; used deterministic fallback"
            t.setdefault("warnings", []).append(warning)
            _emit_progress(progress_callback, {"event": "proposal-timeout", "detail": warning})
            if expired():
                raise SearchTimeoutError(warning) from exc
            return [m["text"] for m in _mutations(plan, current_width, crits)]
        except Exception as exc:
            warning = f"LLM proposal failed ({type(exc).__name__}: {exc}); used deterministic fallback"
            t.setdefault("warnings", []).append(warning)
            _emit_progress(progress_callback, {"event": "proposal-error", "detail": warning})
            return [m["text"] for m in _mutations(plan, current_width, crits)]

    timed_out = False
    termination_reason = "completed"

    if mode in {"ast", "evolutionary"}:
        from .ast_search import ASTSearchEngine, PlanParser, apply_execution_feedback
        from . import ground_check

        gc = ground_check(root_plan, cwd=cwd or Path.cwd())
        initial_state = set(gc.get("verified", []))
        engine = ASTSearchEngine(
            objective=s.get("objective", ""),
            initial_state=initial_state,
            source_plan_text=root_plan,
        )
        root_ast = PlanParser.parse_plan(root_plan, objective=s.get("objective", ""))
        root_member = engine.evaluate_ast(
            root_ast,
            lambda pt: _rollout(pt, rubric)["score"],
            source_plan_text=root_plan,
        )
        engine.population = [root_member]
        if execution_feedback:
            repaired_ast = apply_execution_feedback(root_ast, execution_feedback)
            repaired_member = engine.evaluate_ast(
                repaired_ast,
                lambda pt: _rollout(pt, rubric)["score"],
                source_plan_text=root_plan,
            )
            engine.population.append(repaired_member)

        for iter_idx in range(iterations):
            if expired():
                timed_out, termination_reason = True, "search-timeout"
                break
            _emit_progress(progress_callback, {"event": "iteration", "mode": "ast", "iteration": iter_idx + 1, "total": iterations})
            evolved = engine.evolve_step(
                population_size=max(2, width),
                rubric_score_fn=lambda pt: _rollout(pt, rubric)["score"],
            )
            for member in evolved:
                if expired() or len(nodes) >= max_nodes:
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
            if expired():
                timed_out, termination_reason = True, "search-timeout"
                break
            next_frontier: list[str] = []
            for nid in frontier:
                if expired() or len(nodes) >= max_nodes:
                    break
                plan = nodes[nid]["plan_text"]
                crits = nodes[nid]["critiques"] + _feedback_critiques(execution_feedback)
                try:
                    variants = await variants_for(plan, crits, width)
                except SearchTimeoutError:
                    timed_out, termination_reason = True, "search-timeout"
                    break
                for variant in variants:
                    if expired():
                        timed_out, termination_reason = True, "search-timeout"
                        break
                    ro = _rollout(variant, rubric)
                    penalty = feedback_penalty(variant, execution_feedback)
                    ro["value"] = max(0.0, ro["value"] - penalty)
                    cid = _new_node(t, variant, nid, level + 1, "beam", ro)
                    next_frontier.append(cid)
                    t["rollouts"] += 1
            if timed_out or not next_frontier:
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
            if expired():
                timed_out, termination_reason = True, "search-timeout"
                break
            if len(nodes) >= max_nodes:
                termination_reason = "max-nodes"
                break
            _emit_progress(progress_callback, {"event": "iteration", "mode": "mcts", "iteration": it + 1, "total": iterations, "width": width})
            sel = _select(t, exploration, cost_penalty)
            plan = nodes[sel]["plan_text"]
            crits = nodes[sel]["critiques"] + _feedback_critiques(execution_feedback)
            try:
                variants = await variants_for(plan, crits, width)
            except SearchTimeoutError:
                timed_out, termination_reason = True, "search-timeout"
                break

            sib_vals = [
                (n["value"], n["plan_text"])
                for n in nodes.values()
                if n.get("parent") == nodes[sel].get("parent") and n["id"] != sel
            ]
            if sib_vals:
                close = [txt for val, txt in sib_vals if abs(val - nodes[sel]["value"]) < 0.05]
                if close:
                    xo = _recombine(plan, close[0])
                    if xo:
                        variants = variants[:width] + [xo]

            for variant in variants:
                if expired():
                    timed_out, termination_reason = True, "search-timeout"
                    break
                ro = _rollout(variant, rubric)
                penalty = feedback_penalty(variant, execution_feedback)
                ro["value"] = max(0.0, ro["value"] - penalty)
                cid = _new_node(t, variant, sel, nodes[sel]["depth"] + 1, f"mcts it{it}", ro)
                t["rollouts"] += 1

                if judge_evals:
                    from . import judge
                    try:
                        jr = await asyncio.wait_for(
                            judge(variant, s.get("objective", ""), model=active_model or model),
                            timeout=min(float(proposal_timeout_seconds), remaining()),
                        )
                        if jr.get("ok") and jr.get("verdict") == "go":
                            ro["value"] = min(1.0, ro["value"] + 0.05)
                    except Exception as exc:
                        t.setdefault("warnings", []).append(
                            f"judge evaluation skipped after {type(exc).__name__}: {exc}"
                        )

                gain = ro["value"] - nodes[sel]["value"]
                if gain >= 0.01 and nodes[sel]["depth"] + 2 <= depth + 1:
                    ro2 = _rollout(variant, rubric)
                    if ro2["value"] >= ro["value"] - 0.005:
                        ro = ro2
                        t["rollouts"] += 1
                _backprop(t, cid, ro["value"])

            if prune_margin is not None:
                _prune(t, prune_margin)
            t["expansions"] += 1

            best_score_now = nodes[t["best_node"]]["score"] if t.get("best_node") in nodes else 0.0
            # Never widen an already-converged search. The previous behavior
            # spent more compute precisely when the plan was already ~perfect.
            if best_score_now < float(convergence_score):
                if abs(t["best_value"] - prev_best) < 0.01:
                    plateau += 1
                    if plateau >= 2 and width < start_width + 2:
                        width += 1
                        plateau = 0
                        t.setdefault("escalations", []).append({
                            "iteration": it,
                            "width": width,
                            "best_value": round(t["best_value"], 3),
                        })
                else:
                    plateau = 0
            prev_best = t["best_value"]

    best_node = nodes[t["best_node"]]
    elapsed = round(time.monotonic() - started, 3)
    s.setdefault("search_log", []).append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "expansion": expansion,
        "iterations": iterations,
        "width": width,
        "nodes": len(nodes),
        "rollouts": t["rollouts"],
        "tokens_used": t["tokens_used"],
        "seconds": elapsed,
        "best_score": best_node["score"],
        "best_value": round(best_node["value"], 3),
        "escalations": len(t.get("escalations", [])),
        "termination_reason": termination_reason,
        "warnings": list(t.get("warnings", [])),
    })
    _save_session(plans_dir, s)

    # Preserve historical behavior: if search really found a changed/better
    # plan, assess it as a new round. Already-converged early exit never reaches
    # this mutation point.
    from . import assess
    if best_node["score"] > (s.get("best_score") or 0) or best_node["plan_text"] != root_plan:
        assess(
            s,
            best_node["plan_text"],
            note=f"search:{mode}:score_{best_node['score']}",
            plans_dir=plans_dir,
        )
    else:
        _save_session(plans_dir, s)

    return {
        "best_plan": best_node["plan_text"],
        "best_score": best_node["score"],
        "best_value": round(best_node["value"], 3),
        "best_node": best_node["id"],
        "nodes": len(nodes),
        "rollouts": t["rollouts"],
        "tokens_used": t["tokens_used"],
        "mode": mode,
        "expansion": expansion,
        "implementation_model": active_model,
        "implementation_thinking": thinking_profile if expansion == "llm" else None,
        "termination_reason": termination_reason,
        "timed_out": timed_out,
        "warnings": list(t.get("warnings", [])),
        "escalations": t.get("escalations", []),
        "tree": t,
        "report": _report(t),
    }
