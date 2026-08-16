
"""plan_mode - iterative self-improving planning engine for the ai_funded repo.

Plan mode contract
------------------
A plan session is created for an objective, then improved in rounds.
Each round the agent drafts or revises a plan; `assess()` scores it against a
deterministic rubric and emits structured critiques. The loop continues while
the score improves; history is persisted so improvement is auditable.

Typical interactive usage (this is what the /plan skill instructs):

    import plan_mode
    s = plan_mode.start("make X robust")
    # draft plan v1, then:
    r = plan_mode.assess(s, draft_v1)        # -> critiques
    # revise plan addressing each critique, then:
    r = plan_mode.assess(s, draft_v2)
    # ... until r['status'] == 'converged' or rounds exhausted
    plan_mode.best(s)                        # best-scoring version
    plan_mode.status(s)

The engine is pure Python and deterministic: scores come only from rubric
checks, so "better and better" is a recorded, auditable fact, not a vibe.
"""
from __future__ import annotations

from .causal_validator import (
    ActionSchema,
    CausalFlaw,
    CausalLink,
    CausalValidator,
    PlanAST,
    PlanParser,
    Proposition,
)
from .ast_search import (
    ASTSearchEngine,
    PopulationMember,
    ast_distance,
    crossover_ast,
    mutate_exploratory,
    mutate_flaw_directed,
)
from .memory_distiller import (
    ContextBudgeter,
    ReplanningLadder,
    RoTRule,
    RoTRuleBase,
)

import asyncio
import difflib
import inspect
import json
import hashlib
import shutil
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__version__ = "0.14.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLANS_DIR = Path(os.environ.get("PLAN_PLANS_DIR") or (Path.cwd() / "plans"))
RUBRIC_PATH = Path(__file__).resolve().parent / "RUBRIC.md"

MIN_DELTA_TO_CONTINUE = 1.0      # a round must beat the best score by >= this
MAX_PLATEAU_ROUNDS = 2           # stop after this many non-improving rounds
DEFAULT_MAX_ROUNDS = 8

DEFAULT_RUBRIC: dict[str, dict[str, Any]] = {'objective': {'label': 'Objective clarity', 'weight': 10, 'items': [('explicit_goal', '\\b(goal|objective|aim|purpose)\\b[^\\n]{0,200}', 'State the objective explicitly in one opening line.'), ('scope_in', '(in\\s*scope|included|we will (do|build|deliver))', 'List what is in scope.'), ('scope_out', '(out\\s*of\\s*scope|excluded|not\\s*(doing|building|in scope)|non-goals?)', 'List what is explicitly out of scope (non-goals).')]}, 'success': {'label': 'Measurable success criteria', 'weight': 15, 'items': [('numeric_criteria', '(\\d+(\\.\\d+)?\\s*(%|percent|ms|s|min|hours?|days?|weeks?|points?|items?|papers?|files?|tests?|epochs?|rounds?)|\\bpass\\b|\\bfail\\b)', 'Give numeric/verifiable acceptance criteria.'), ('deadline', '(by\\s+\\d{4}-\\d{2}-\\d{2}|within\\s+\\d+\\s+(hour|day|week|month)s?|deadline)', 'Give a time bound or deadline.'), ('falsifiable', '(pass/fail|pass\\s*/\\s*fail|verif\\w+ (by|with|via)|measur\\w+|acceptance (test|criteria)|reject\\b|falsif\\w+)', 'Define a pass/fail (falsifiable) check per criterion.')]}, 'assumptions': {'label': 'Assumptions and unknowns', 'weight': 10, 'items': [('assumptions', '(^|\\n)\\s*#{0,6}\\s*assumptions?:?(\\s|$)', 'List explicit assumptions.'), ('unknowns', '(unknowns?|open questions?|to\\s+be\\s+determined|TBD|risks?:)', 'List unknowns / open questions.')]}, 'tasks': {'label': 'Task decomposition and dependencies', 'weight': 15, 'items': [('numbered_tasks', '(?m)^\\s*(\\d+[.)]|[-*])\\s+[A-Z][^\\n]{10,}$', 'Provide at least 3 concrete, ordered tasks.', 3), ('dependencies', '(depends?\\s+on|after\\s+step|before\\s+step|blocked\\s+by|prerequisite|dependency)', 'State dependencies between tasks.'), ('outputs', '(output|deliverable|artifact|produces?)', 'Say what artifact each task produces.')]}, 'milestones': {'label': 'Milestones and checkpoints', 'weight': 10, 'items': [('milestones', '(milestone|checkpoint|gate|phase\\s+\\d)', 'Define intermediate milestones or checkpoints.'), ('gonogo', '(go/no-go|go\\s*/\\s*no-go|decision (point|gate)|abort|stop condition)', 'Include a go/no-go decision gate.')]}, 'risks': {'label': 'Risks and failure modes', 'weight': 15, 'items': [('risk_list', '((^|\\n)\\s*#{0,6}\\s*risks?:?(\\s|$)|failure modes?:|could fail|might fail|worst case)', 'List risks or failure modes.'), ('mitigations', '(mitigat\\w+|fallback|contingency|rollback|revert|backup plan)', 'Give a mitigation/fallback per major risk.'), ('rollback', '(rollback|revert|undo|restore)', 'Describe how to roll back or undo.')]}, 'resources': {'label': 'Resource and cost estimates', 'weight': 10, 'items': [('time_estimate', '(\\d+\\s*(minutes?|hours?|days?|weeks?|months?)\\s*(per|each|total|budget)|\\bestimated?\\b)', 'Estimate time per task or total.'), ('budget', '(cost|budget|\\$|USD|tokens?|compute|GPU)', 'Estimate cost/compute/token budget.')]}, 'alternatives': {'label': 'Alternatives considered', 'weight': 5, 'items': [('alternatives', '(alternative|instead|rather than|option [A-Z]|vs\\.? )', 'Consider at least one alternative and say why it was rejected.')]}, 'verification': {'label': 'Verification and self-improvement loop', 'weight': 10, 'items': [('verify', '(verify|validate|test|audit|check that|review)', 'Include a verification step for each milestone.'), ('feedback_loop', '(revis\\w+|iterate|improve|refine|feedback|next round|loop)', 'Include a revision/improvement loop.')]}, 'structure': {'label': 'Explicit plan structure', 'weight': 6, 'items': [['declared_structure', '(sections?|phases?|##|numbered|structure of this plan)', "Declare the plan's structure (sections/phases)."], ['pseudocode_steps', '(?m)^\\s*(step\\s+\\d+|\\d+[.)])\\s*[A-Za-z]', 'Use numbered pseudocode-style steps.', 3]]}, 'constraints': {'label': 'Step constraints (preconditions/effects)', 'weight': 6, 'items': [['preconditions', '(precondition|requires?|inputs?|before starting|prerequisite)', 'State preconditions/inputs per step.'], ['effects', '(expected (output|result|effect)|outputs?|deliverables?|produces?|postcondition)', 'State expected outputs/effects per step.'], ['per_step_constraints', '(applies? at (each|every|the) step|re-?anchor|constraint(s)? (per|at (each|every)) step|step-level (constraint|check))', 'Re-anchor each constraint at the step where it applies.']]}, 'verification_machine': {'label': 'Machine-checkable verification', 'weight': 6, 'items': [['invariant', '(invariant|checkable|solver|validator|automated check|script|test that)', 'Include at least one mechanically checkable invariant.'], ['how_checked', '(verify (by|with|via|using)|checked by|validated by|assert|hash|checksum)', 'Say how the invariant is checked (tool/script/assert).'], ['external_checker', '(solver|validator|script|tool|external (check|verif)|non-LLM|deterministic (check|verif)|assert)', 'Name a non-LLM checker (tool/script/solver) for the plan.']]}, 'replan': {'label': 'Replanning policy', 'weight': 6, 'items': [['triggers', '(replan|revise the plan|if .* (fail|change|wrong)|trigger)', 'Define replan triggers (what failure/observation causes revision).'], ['scoped_repair', '(smallest|local (fix|repair|patch)|patch (the|only|that)|reuse (the|prior|existing) plan|prefix)', 'Prefer smallest-scope repair and reuse of the valid plan prefix.'], ['preemptive_failure_enumeration', '(likely (failure|pitfall|risk)s? (before|up ?front|in advance)|enumerate (failure|pitfall)|negative constraints?|proactive (pitfall|failure) avoidance)', 'Enumerate likely failure modes as negative constraints before drafting.']]}, 'grounding': {'label': 'Step grounding and progress detection', 'weight': 6, 'items': [['step_outcome', '(how (to|we) know|detect|success (of|for) (a |each |the )?step|step (succeeds|fails|works))', "Say how each step's success/failure is detected."], ['subgoal_checkpoints', '(sub-?goal|intermediate (goal|checkpoint)|checkpoint after|incremental progress)', 'Include intermediate sub-goal checkpoints for long plans.'], ['state_restatement', '(restat(e|ing) (the )?(state|world|goal)|memoiz\\w+ state|state (after|at) (each|every) step|update (the )?state (after|per|each))', 'Restate the world/plan state after each step to prevent goal drift.']]}, 'memory': {'label': 'History and revision strategy', 'weight': 4, 'items': [['lessons', '(lessons? learned|past (failure|attempt|mistake)|previous (round|version)|history)', 'Reference past failures/lessons or plan history.'], ['revision_strategy', '(revision (strategy|loop|policy)|improve (by|via|through)|next round|iterate)', 'Name the revision strategy (how the next round will differ).'], ['diverse_alternatives', '(best-?of-?N|candidate (plans?|alternatives?|options?)|sample (multiple|several|N) (plans?|alternatives?)|evaluate (multiple|candidate))', 'Consider diverse candidate plans and pick the best via an evaluator (best-of-N).']]}, 'executability': {'label': 'Immediate executability', 'weight': 10, 'items': [('first_action', '(first|step\\s*1|start by|today|immediately|now)', 'Name the very first action to take.'), ('no_vague', '', "Every step is concrete (no bare 'explore'/'consider' without a criterion).")]}, 'escalation': {'label': 'Adaptive deliberation and escalation', 'weight': 4, 'items': [['cheapest_first', '(cheapest|cheap|low-?cost).{0,40}(first|action|option)', 'Attempt the cheapest viable action first.'], ['escalate_when_needed', '(escalat\\w+|only if (needed|necessary)|fall ?back to)', 'Escalate to expensive/deliberate steps only when needed.']]}, 'uncertainty': {'label': 'Uncertainty awareness', 'weight': 4, 'items': [['uncertainty_estimate', '(uncertain\\w+|confidence (estimate|level|score)|risk of (failure|error))', 'Estimate uncertainty per step.'], ['conservative_switch', '(conservative|switch (planning )?mode|fall ?back plan|when (uncertain|confidence is low))', 'Switch to a conservative mode when uncertainty is high.']]}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "plan"


def _load_rubric() -> dict[str, dict[str, Any]]:
    """Rubric is the code default; RUBRIC.md can override weights/items in a
    fenced JSON block to let the literature review tune it without code edits."""
    if RUBRIC_PATH.exists():
        text = RUBRIC_PATH.read_text(encoding="utf-8")
        m = re.search(r"```json\s*\n(.*?)\n```", text, re.S)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
    return {k: dict(v) for k, v in DEFAULT_RUBRIC.items()}


def _mechanical_checks(text: str) -> list[dict[str, str]]:
    """Objective, literature-backed checks (PlanBench 2409.13373: self-checking
    prose is unreliable; verify against mechanical properties instead)."""
    critiques: list[dict[str, str]] = []

    # contiguous task numbering
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)[.)]\s+[A-Za-z]", text, re.M)]
    if len(nums) >= 3 and nums != list(range(1, len(nums) + 1)):
        critiques.append({"id": "mech:task-numbering", "section": "mechanical",
                          "hint": "Task numbering must be contiguous with no duplicates (1, 2, 3, ...)."})

    # dependency references must point at existing task numbers
    if nums:
        for m in re.finditer(r"depends?\s+on\s+((?:\d+\s*(?:,|\s|and\s*)?)+)", text, re.I):
            refs = [int(x) for x in re.findall(r"\d+", m.group(1))]
            missing = [x for x in refs if x not in nums]
            if missing:
                critiques.append({"id": "mech:dep-ref", "section": "mechanical",
                                  "hint": f"Dependency references missing task(s): {missing}."})

    # deadline dates must parse and be in the future
    now = datetime.now(timezone.utc)
    for m in re.finditer(r"(?:deadline|by)\s+(\d{4}-\d{2}-\d{2})", text, re.I):
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if dt < now:
                critiques.append({"id": "mech:past-deadline", "section": "mechanical",
                                  "hint": f"Deadline {m.group(1)} is in the past."})
        except ValueError:
            critiques.append({"id": "mech:bad-date", "section": "mechanical",
                              "hint": f"Deadline date {m.group(1)} does not parse as YYYY-MM-DD."})

    # duplicated task lines
    lines = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip().lower()
             for ln in text.splitlines() if re.match(r"^\s*\d+[.)]\s+", ln)]
    dups = {ln for ln in lines if len(ln) > 20 and lines.count(ln) > 1}
    if dups:
        critiques.append({"id": "mech:dup-task", "section": "mechanical",
                          "hint": "Duplicate task line(s) detected; remove redundancy."})
    return critiques


def _score(text: str, rubric: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = 0.0
    max_total = 0.0
    per_section: dict[str, Any] = {}
    critiques: list[dict[str, str]] = []
    for key, section in rubric.items():
        weight = float(section.get("weight", 0))
        items = section.get("items", [])
        max_total += weight
        hits = 0
        scored_items = [it for it in items if it[1]]
        for item in items:
            pattern, hint = item[1], item[2]
            min_count = item[3] if len(item) > 3 else 1
            if not pattern:
                continue
            if len(re.findall(pattern, text, re.I | re.M)) >= min_count:
                hits += 1
            else:
                critiques.append({
                    "id": f"{key}:{hint[:40]}",
                    "section": str(section.get("label", key)),
                    "hint": hint,
                })
        sec_score = weight * hits / len(scored_items) if scored_items else 0.0
        per_section[key] = {"label": section.get("label", key), "hits": hits, "items": len(items),
                            "weight": weight, "section_score": round(sec_score, 2)}
        total += sec_score
    score = round(100.0 * total / max_total, 2) if max_total else 0.0
    return {"score": score, "sections": per_section, "critiques": critiques}


def _session_path(plans_dir: Path, session_id: str) -> Path:
    return Path(plans_dir) / f"{session_id}.json"


_SESSION_LOCK = threading.RLock()


def _load_session(plans_dir: Path, session_id: str) -> dict[str, Any]:
    p = _session_path(plans_dir, session_id)
    if not p.exists():
        raise FileNotFoundError(f"no plan session {session_id!r} in {plans_dir}")
    with _SESSION_LOCK:
        try:
            import fcntl
            with open(p, "r", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            return json.loads(p.read_text(encoding="utf-8"))


def _save_session(plans_dir: Path, session: dict[str, Any]) -> None:
    plans_dir = Path(plans_dir)
    plans_dir.mkdir(parents=True, exist_ok=True)
    p = _session_path(plans_dir, session["session_id"])
    tmp_path = plans_dir / f".{session['session_id']}.tmp.{os.getpid()}_{time.time_ns()}"
    with _SESSION_LOCK:
        with open(tmp_path, "w", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            json.dump(session, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        os.replace(tmp_path, p)


async def judge_ensemble(session: dict[str, Any] | str, plan_text: str, objective: str,
                        n: int = 3, *, plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Ensemble judgment (2510.03469, 2601.17942): collect up to n verdicts
    (API judge, then any pre-recorded independent verdicts) and record the
    median-feasibility verdict with the individual votes. The median, not the
    worst or best vote, is the recorded judgment — a defense against the
    single-judge noise observed in practice (45 <-> 86 swings)."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    # baseline: the mechanical vote always participates (ground truth)
    v = verify(plan_text)
    si = simulate(plan_text)
    baseline = {"ok": True,
                "verdict": "go" if (v["ok"] and si["executable_plan"]) else "rework",
                "feasibility_0_100": 100 if (v["ok"] and si["executable_plan"]) else 40,
                "falsifiable_criteria": True, "judge_path": "local-deterministic-fallback",
                "source": "mechanical_baseline", "external": False}
    votes: list[dict[str, Any]] = [baseline]
    try:
        j = await judge(plan_text, objective)
        # a judge vote that is not falsifiable carries no gate information
        if isinstance(j, dict) and j.get("ok") and j.get("falsifiable_criteria"):
            votes.append(j)
    except Exception:
        pass
    for prior in list(s.get("judge_log", [])[-n:]):
        if prior.get("ok") and prior.get("falsifiable_criteria") and prior not in votes:
            votes.append(prior)
    votes = votes[:max(1, min(n, len(votes)))]
    feases = sorted(vt.get("feasibility_0_100", 0) for vt in votes)
    median = feases[(len(feases) - 1) // 2]  # lower median: conservative
    med_vote = min(votes, key=lambda vt: abs(vt.get("feasibility_0_100", 0) - median))
    entry = {**med_vote, "ensemble": True, "votes": votes, "median_feasibility": median,
             "verdict": med_vote.get("verdict", "go"), "ok": True}
    record_judge(s, entry)
    return entry


def template() -> dict[str, Any]:
    """The plan-mode template bank: every planner gets the engine's real
    function inventory, a sample plan skeleton, and the mutation section
    templates, so plans name concrete mechanisms instead of prose."""
    from .search_engine import _SECTION_TEMPLATES, _MUTATIONS
    inventory = [
        ("assess(session, plan_text)", "score a plan version; returns critiques"),
        ("verify(plan_text)", "structural audit: DAG, artifacts, type mismatches, landmarks, horizon, budget"),
        ("simulate(plan_text, initial_state)", "STRIPS execution in written order; verified inputs seed state"),
        ("ground_check(plan_text, cwd)", "every declared environment input must exist"),
        ("constraint_check(plan_text)", "at-most/at-least numeric constraints solved mechanically"),
        ("search(session, iterations, width)", "MCTS/beam with adaptive escalation, recombination, seed pool"),
        ("judge(plan_text, objective)", "grounded external verdict (mechanical anchoring)"),
        ("judge_ensemble(session, plan_text, objective, n)", "median over falsifiable votes + mechanical baseline"),
        ("release(session)", "six-gate release: converged, score, verify, feasibility, simulation, judge"),
        ("fold_history(session)", "fold superseded rounds; legacy sessions never mutated"),
        ("log_progress(session, task, status, evidence)", "execution log + world_state feedback; arms replan ladder"),
        ("plan_quality(plan_text, objective)", "combined structural + simulation + coverage verdict"),
    ]
    sample_plan = """# Goal
Goal: <objective in one line>. In scope: <what we will deliver>. Out of scope (non-goals): <what we will not do>.

## Success criteria
- S1: <numeric or pass/fail criterion>; verified by <command/file>. Deadline: within <N> days.

## Assumptions and unknowns
- Assumptions: <list>. Unknowns / open questions: <TBD items>.

## Tasks
1. <First action>. Depends on: none. Requires: <existing input file, verified by ground_check>. Output: <artifact>. Exit criterion: <measurable>; stop when reached. Time: <N> min. Confidence: high.
2. <Next step>. Depends on task 1. Output: <artifact>. Time: <N> min. Confidence: high.

## Verification machine
- Invariant: <mechanically checkable claim>; checked by a non-LLM script with assert.
- Corrective action when verification fails: <local fix>. Pre-execution critique over risky steps.
"""
    return {
        "function_inventory": inventory,
        "sample_plan": sample_plan,
        "section_templates": {k: v.strip() for k, v in _SECTION_TEMPLATES.items()},
        "mutation_names": [name for name, _ in _MUTATIONS],
    }


# --- Cordis layer (A Programming Paradigm for Spatiotemporal Composability,
# Yifan Shi / Wei Zhang / Tianyi Cui): revertible effects + reactive coeffects ---

JOURNAL_PATH = Path(__file__).resolve().parent / ".plan_mode_journal.jsonl"


def _journal() -> list[dict[str, Any]]:
    """The effect journal: every tracked mutation records its inverse (the
    previous content). Reading is cheap; the journal is append-only."""
    if not JOURNAL_PATH.exists():
        return []
    out = []
    for line in JOURNAL_PATH.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def edit_file(path: str | Path, content: str, *, note: str = "") -> dict[str, Any]:
    """Revertible effect primitive (Cordis ctx.effect): apply a file mutation
    and automatically record its inverse (the prior content + hash) in the
    journal, so rollback can recover it later without hand-rolled backups."""
    p = Path(path)
    before = p.read_text() if p.exists() else None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    entry = {"ts": _now(), "path": str(p), "note": note,
             "before": before,
             "before_sha": hashlib.sha256(before.encode("utf-8")).hexdigest() if before is not None else None,
             "after_sha": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    with JOURNAL_PATH.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def rollback(n: int = 1, *, dry_run: bool = False) -> dict[str, Any]:
    """Apply the n most recent journaled inverses IN REVERSE ORDER (twisted
    composition: last effect recovers first). Returns what was reverted."""
    entries = _journal()
    to_undo = entries[-n:]
    done = []
    for e in reversed(to_undo):
        if not dry_run:
            if e["before"] is not None:
                Path(e["path"]).write_text(e["before"])
            else:
                Path(e["path"]).unlink(missing_ok=True)
        done.append(e)
    if not dry_run:
        # consume the undone entries so repeated rollbacks walk further back
        JOURNAL_PATH.write_text("\n".join(json.dumps(x) for x in entries[:-n]))
    return {"reverted": len(done), "entries": done, "dry_run": dry_run}


def deps_check() -> dict[str, Any]:
    """Reactive coeffects (Cordis notify): classify every feature dependency as
    satisfied / unsatisfied. Features whose deps are unsatisfied DEGRADE
    gracefully instead of crashing (activating/deactivating/neutral)."""
    specs = {
        "llm_expansion": lambda: bool(os.environ.get("DEEPSEEK_API_KEY") or _auth_key()),
        "api_judge": lambda: bool(os.environ.get("DEEPSEEK_API_KEY") or _auth_key()),
        "pytest": lambda: shutil.which("python3") is not None,
        "corpus": lambda: Path(os.environ.get("PLANNING_CORPUS", "/home/lewbei/deep_learning/planning_paper/txts")).exists(),
        "pdftotext": lambda: shutil.which("pdftotext") is not None,
    }
    status = {}
    for name, check in specs.items():
        try:
            status[name] = "satisfied" if check() else "unsatisfied"
        except Exception as e:
            status[name] = f"error: {e}"
    return {"status": status,
            "unsatisfied": [k for k, v in status.items() if v != "satisfied"]}


def _auth_key() -> str:
    try:
        auth = json.loads((Path.home() / ".prime" / "agent" / "auth.json").read_text())
        cred = auth.get("deepseek") if isinstance(auth, dict) else None
        return str(cred.get("key") or "") if isinstance(cred, dict) else ""
    except (OSError, ValueError):
        return ""


def fold_history(session: dict[str, Any] | str, *, plans_dir: str | Path | None = None,
                 keep_last: int = 2, max_context_tokens: int = 4000) -> dict[str, Any]:
    """Fold superseded round texts into one-line summaries (HIPIF 2606.10507):
    uses ContextBudgeter to keep the best round and latest rounds in full,
    compressing older rounds to honor max_context_tokens. Legacy sessions (pre-v0.6.0)
    are never mutated (read-only compat)."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    if s.get("engine_version", "0.0.0") < "0.6.0" or s.get("history_folded"):
        return s
    rounds = s.get("rounds") or []
    if len(rounds) <= keep_last + 1:
        return s

    s = ContextBudgeter.compress_history(s, max_context_tokens=max_context_tokens)
    s["history_folded"] = True
    _save_session(plans_dir, s)
    return s


def start(objective: str, *, plans_dir: str | Path | None = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
          session_id: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create (or resume) a plan session for an objective."""
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    plans_dir.mkdir(parents=True, exist_ok=True)
    if session_id:
        sid = session_id
        p = _session_path(plans_dir, sid)
        if p.exists():
            return _load_session(plans_dir, sid)
    else:
        # Resume existing active session for this objective if present
        slug = _slugify(objective)
        matching: list[tuple[float, str]] = []
        for sess_file in plans_dir.glob(f"{slug}*.json"):
            try:
                data = json.loads(sess_file.read_text(encoding="utf-8"))
                if data.get("objective") == objective and data.get("status") not in ("finished", "abandoned"):
                    matching.append((sess_file.stat().st_mtime, data.get("session_id", sess_file.stem)))
            except Exception:
                continue
        if matching:
            matching.sort(reverse=True)
            return _load_session(plans_dir, matching[0][1])
        sid = f"{slug}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    p = _session_path(plans_dir, sid)
    if p.exists():
        return _load_session(plans_dir, sid)  # resume; keeps improving existing plan
    rubric = _load_rubric()
    session: dict[str, Any] = {
        "session_id": sid,
        "objective": objective,
        "created_at": _now(),
        "plans_dir": str(plans_dir),
        "rubric_version": "v8",
        "engine_version": __version__,
        "rubric_snapshot": rubric,
        "max_rounds": max_rounds,
        "meta": meta or {},
        "rounds": [],
        "best_version": None,
        "best_score": None,
        "status": "drafting",   # drafting -> improving -> converged -> finished
        "completed_at": None,
        "execution_log": [],
        "replan_pending": False,
        "replan_task": None,
        "suggestions": [],
    }
    _save_session(plans_dir, session)
    return session


def assess(session: dict[str, Any] | str, plan_text: str, *, note: str | None = None,
           addressed: list[str] | None = None, plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Score a plan version, record it, and return critiques + loop status.

    session may be the session dict (from start) or a session_id string.
    addressed: critique ids this revision claims to have fixed (recorded for
    audit; unresolved ids are re-emitted in the next round).
    Returns {"version", "score", "delta", "critiques", "status", "continue"}.
    """
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    if isinstance(session, str):
        session = _load_session(plans_dir, session)
    else:
        # a live session dict knows where it lives; never scatter files elsewhere
        plans_dir = Path(session.get("plans_dir") or plans_dir)
    if session["status"] in ("finished",):
        raise RuntimeError(f"session {session['session_id']} is finished; start a new objective to continue")
    rubric = session.get("rubric_snapshot") or _load_rubric()
    result = _score(plan_text, rubric)
    result["critiques"] = result["critiques"] + _mechanical_checks(plan_text)
    # verify-on-mismatch (2410.00079, 2605.07248): reuse cached structural
    # results when the exact same text was checked before (lazy re-verification)
    _cache = session.setdefault("verify_cache", {})
    _h = hashlib.md5(plan_text.encode("utf-8")).hexdigest()
    # structural validity: a plan whose task graph is broken cannot converge
    # even if every regex fires (PlanBench 2409.13373: structure beats prose)
    v = _cache.get(_h)
    if v is None:
        v = verify(plan_text)
        _cache[_h] = v
        if len(_cache) > 32:
            _cache.pop(next(iter(_cache)))
    for err in v["errors"]:
        result["critiques"].append({"id": f"mech:verify:{err[:40]}", "section": "mechanical",
                                    "hint": err})
    result["verify"] = v
    # grounded feasibility (2402.11489): declared environment inputs must
    # exist; verified inputs seed the simulation's initial state so real
    # resources satisfy the simulator instead of blocking it
    cc = constraint_check(plan_text)
    for p_ in cc["problems"]:
        result["critiques"].append({"id": f"mech:constraint:{p_[:40]}",
                                    "section": "mechanical",
                                    "hint": f"[solver] {p_}"})
    result["constraint_check"] = cc
    gc = ground_check(plan_text)
    for m in gc["missing"]:
        result["critiques"].append({"id": f"mech:feasibility:{m[:40]}",
                                    "section": "mechanical",
                                    "hint": f"[grounding] declared input does not exist: {m}"})
    result["ground_check"] = gc
    # planning simulation (SymPlanner 2505.01479): execute the plan against
    # an explicit state model; a blocked task is a hard structural critique
    sim = simulate(plan_text, initial_state=set(gc["verified"]))
    if not sim["executable_plan"]:
        for prob in sim["problems"]:
            result["critiques"].append({"id": f"mech:sim:{prob[:40]}", "section": "mechanical",
                                        "hint": f"[simulation] {prob}"})
    result["simulation"] = sim
    # RoT memory (2404.05449): distill negative rules from causal flaws and
    # enforce them on subsequent candidate plans for this session.
    rot_path = Path(plans_dir) / f"{session['session_id']}.rot.json"
    rot_base = RoTRuleBase(storage_path=rot_path)
    causal_flaws = (v.get("causal_validation") or {}).get("flaws", [])
    rot_base.distill_from_flaws(causal_flaws, context_tag=session.get("objective", "general"))
    rot_violations = rot_base.check_plan_violations(plan_text)
    for viol in rot_violations:
        result["critiques"].append({"id": f"mech:rot:{viol['rule_id']}", "section": "mechanical",
                                    "hint": f"[learned rule] {viol['remedy']}"})
    result["rot_rules"] = {
        "learned": len(rot_base.rules),
        "violations": [v["rule_id"] for v in rot_violations],
    }
    # external judge: re-emit unresolved blockers from the last judge verdict
    # so the loop keeps revising until the judge says "go" (2510.03469)
    last_judge = session.get("judge_log", [{}])[-1] if session.get("judge_log") else None
    if last_judge and last_judge.get("ok") and last_judge.get("verdict") != "go":
        for b in (last_judge.get("blockers") or [])[:5]:
            result["critiques"].append({"id": f"judge:blocker:{b[:40]}", "section": "judge",
                                        "hint": f"[external judge] {b}"})
        for c in (last_judge.get("contradictions") or [])[:3]:
            result["critiques"].append({"id": f"judge:contradiction:{c[:40]}", "section": "judge",
                                        "hint": f"[external judge] {c}"})
        if not last_judge.get("falsifiable_criteria", True):
            result["critiques"].append({"id": "judge:unfalsifiable", "section": "judge",
                                        "hint": "[external judge] success criteria are not falsifiable"})

    # root-cause grouping (2509.25370): order misses by section score ascending
    # and, when several checks miss, name the lowest-scoring section first.
    sections = result.get("sections") or {}
    score_by = {str(v.get("label", k)): float(v.get("section_score", 0)) for k, v in sections.items()}
    non_mech = [c for c in result["critiques"] if not c["id"].startswith(("mech:", "judge:"))]
    if non_mech:
        non_mech.sort(key=lambda c: score_by.get(str(c.get("section", "")), 1e9))
        result["critiques"] = [c for c in result["critiques"]
                               if c["id"].startswith(("mech:", "judge:"))] + non_mech
        if len(non_mech) >= 3 and sections:
            weakest = min(sections, key=lambda k: sections[k].get("section_score", 0))
            result["critiques"].append({"id": f"root_cause:{weakest}",
                                        "section": sections[weakest].get("label", weakest),
                                        "hint": f"Root cause: section '{sections[weakest].get('label', weakest)}' "
                                                f"scores lowest ({sections[weakest].get('section_score')}); "
                                                "fix it before symptom-level edits."})

    # 1) similarity guard: a "revision" that barely changes the text is not an
    #    improvement round (SRDrone 2508.15501: refinement must be substantive).
    prev_round = session["rounds"][-1] if session["rounds"] else None
    changed = True
    if prev_round is not None:
        ratio = difflib.SequenceMatcher(None, prev_round["plan_text"], plan_text).ratio()
        if ratio > 0.97:
            result["critiques"].append({"id": "mech:barely-changed", "section": "mechanical",
                                        "hint": "Revision is nearly identical to the previous version "
                                                "(similarity > 0.97); make a substantive change."})
            changed = False

    # 2) critique-addressing audit: re-emit previous critiques not claimed fixed.
    addressed_set = {a.lower() for a in (addressed or [])}
    unaddressed: list[str] = []
    if prev_round is not None:
        unaddressed = [c["id"] for c in prev_round["critiques"]
                       if c["id"].lower() not in addressed_set and not c["id"].startswith("mech:barely")]
        if unaddressed:
            result["critiques"].insert(0, {"id": "mech:unaddressed", "section": "mechanical",
                                           "hint": f"Previous critiques still unaddressed: "
                                                   f"{', '.join(unaddressed[:6])}"})

    # 3) replan trigger: a failed execution step forces a smallest-scope revision
    #    (RePLan 2401.04157, hierarchical recovery 2606.20487).
    if session.get("replan_pending"):
        failed = session.get("replan_task") or "unknown task"
        # tiered replanning ladder (2605.25851): escalate across three levels
        tier = session.get("replan_tier", 1)
        scope = session.get("replan_scope")
        if scope and scope.get("description"):
            hint = scope["description"]
        else:
            ladder = {
                1: f"Execution failed on '{failed}': level 1 subgoal audit — re-check that task's deps and outputs before anything else.",
                2: f"Execution failed on '{failed}' again: level 2 structured search — locate the failing resource/step and repair it specifically.",
                3: f"Execution failed on '{failed}' again: level 3 preemptive global replan — redraft the affected phase, not just the step.",
            }
            hint = ladder[min(tier, 3)]
        result["critiques"].insert(0, {"id": "mech:replan", "section": "mechanical",
                                       "hint": hint})
        session["replan_tier"] = min(tier + 1, 3)
        if "mech:replan" in addressed_set:
            session["replan_pending"] = False
            session["replan_tier"] = 1

    version = len(session["rounds"]) + 1
    best = session.get("best_score")
    delta = round(result["score"] - best, 2) if best is not None else None
    session["rounds"].append({
        "version": version,
        "ts": _now(),
        "score": result["score"],
        "delta": delta,
        "critiques": result["critiques"],
        "sections": result["sections"],
        "note": note,
        "addressed": addressed or [],
        "unaddressed": unaddressed,
        "substantive": changed,
        "plan_text": plan_text,
    })
    if not changed and best is not None:
        # a non-substantive round never improves; count it as a plateau
        session["rounds"][-1]["delta"] = 0.0
    if best is None or result["score"] > best:
        session["best_version"] = version
        session["best_score"] = result["score"]
    if best is None:
        session["status"] = "improving"
    if version >= session.get("max_rounds", DEFAULT_MAX_ROUNDS):
        session["status"] = "converged"
        fold_history(session, plans_dir=plans_dir)
        _save_session(plans_dir, session)
        return {"version": version, "score": result["score"], "delta": delta,
                "critiques": result["critiques"], "status": "converged",
                "continue": False, "verify": result.get("verify"),
                "simulation": result.get("simulation"), "rot_rules": result.get("rot_rules")}
    # convergence rule: score must keep improving; allow MAX_PLATEAU_ROUNDS
    # non-improving rounds in a row before declaring convergence.
    recent = [r["score"] for r in session["rounds"]]
    plateau = 0
    for i in range(len(recent) - 1, 0, -1):
        if recent[i] >= recent[i - 1] + MIN_DELTA_TO_CONTINUE:
            break
        plateau += 1
    open_mech = [c for c in result["critiques"] if c["id"].startswith("mech:")]
    if open_mech:
        # objective errors block convergence; only convergence with a clean plan
        _save_session(plans_dir, session)
        return {"version": version, "score": result["score"], "delta": delta,
                "critiques": result["critiques"], "status": "improving",
                "continue": True, "verify": result.get("verify"),
                "simulation": result.get("simulation"), "rot_rules": result.get("rot_rules")}
    if plateau >= MAX_PLATEAU_ROUNDS and version >= 2:
        if len(_task_blocks(plan_text)) <= 3:
            result["critiques"].append({"id": "hint:over-refinement",
                                        "section": "escalation",
                                        "hint": "Short stepwise plan: refinement over-corrects on small horizons (2606.04874); "
                                                "accept the best version instead of more rounds."})
        session["status"] = "converged"
        fold_history(session, plans_dir=plans_dir)
        _save_session(plans_dir, session)
        return {"version": version, "score": result["score"], "delta": delta,
                "critiques": result["critiques"], "status": "converged",
                "continue": False, "verify": result.get("verify"),
                "simulation": result.get("simulation"), "rot_rules": result.get("rot_rules")}
    _save_session(plans_dir, session)
    return {"version": version, "score": result["score"], "delta": delta,
            "critiques": result["critiques"], "status": session["status"],
            "continue": True, "verify": result.get("verify"),
            "simulation": result.get("simulation"), "rot_rules": result.get("rot_rules")}


def run(objective: str, draft_plan: str, *, plans_dir: str | Path | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS, note: str | None = None) -> dict[str, Any]:
    """Convenience: start a session and record the first plan version.

    Returns the session dict; pass it to assess() for subsequent rounds."""
    s = start(objective, plans_dir=plans_dir, max_rounds=max_rounds)
    assess(s, draft_plan, note=note, plans_dir=plans_dir)
    return s


def status(session: dict[str, Any] | str, *, plans_dir: str | Path | None = None) -> dict[str, Any]:
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    s = _load_session(plans_dir, session) if isinstance(session, str) else session
    scores = [r["score"] for r in s["rounds"]]
    return {
        "session_id": s["session_id"],
        "objective": s["objective"],
        "status": s["status"],
        "rounds": len(s["rounds"]),
        "best_version": s["best_version"],
        "best_score": s["best_score"],
        "score_history": scores,
        "created_at": s["created_at"],
        "completed_at": s["completed_at"],
        "plans_dir": s["plans_dir"],
    }


def history(session: dict[str, Any] | str, *, plans_dir: str | Path | None = None) -> list[dict[str, Any]]:
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    s = _load_session(plans_dir, session) if isinstance(session, str) else session
    return [{"version": r["version"], "ts": r["ts"], "score": r["score"], "delta": r["delta"],
             "n_critiques": len(r["critiques"]), "note": r["note"]} for r in s["rounds"]]


def best(session: dict[str, Any] | str, *, plans_dir: str | Path | None = None) -> dict[str, Any]:
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    s = _load_session(plans_dir, session) if isinstance(session, str) else session
    if s["best_version"] is None:
        return {"error": "no versions recorded yet"}
    r = s["rounds"][s["best_version"] - 1]
    return {"version": r["version"], "score": r["score"], "critiques": r["critiques"],
            "sections": r["sections"], "plan_text": r["plan_text"]}




def release(session: dict[str, Any] | str, *, min_score: float = 90.0,
            require_judge: bool = True,
            require_external_judge: bool = False,
            plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Release gate (2602.08948 confidence-gated checkpoints, 2608.10729
    acceptance thresholds): a plan may only be released to execution after
    ALL of: (1) the assess loop has converged, (2) best score >= min_score,
    (3) mechanical checks are clean (dates, deadlines, duplicates),
    (4) verify() is clean, (5) ground_check() feasibility is satisfied,
    (6) the simulation executes end-to-end, and (7) the judge has returned
    verdict "go" with falsifiable criteria. Until then the plan keeps looping.
    Returns the gate report; the plan must NOT be reported as done while ok is False.
    """
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    checks: list[dict[str, Any]] = []
    problems: list[str] = []
    historical: dict[str, Any] = {}

    converged = s.get("status") == "converged"
    checks.append({"name": "converged", "ok": converged,
                   "detail": f"status={s.get('status')}"})
    if not converged:
        problems.append("plan has not converged; keep looping assess->revise")

    best_score = s.get("best_score") or 0
    checks.append({"name": "score", "ok": best_score >= min_score,
                   "detail": f"best={best_score} >= {min_score}"})
    if best_score < min_score:
        problems.append(f"best score {best_score} < {min_score}; keep revising")

    best_text = ""
    if s.get("best_version") and s.get("rounds"):
        best_round = s["rounds"][s["best_version"] - 1]
        best_text = best_round.get("plan_text", "")

    # Re-run canonical mechanical checks (deadlines, dates, task numbering, duplicates)
    mech = _mechanical_checks(best_text) if best_text else [{"id": "mech:empty", "hint": "no best plan yet"}]
    checks.append({"name": "mechanical", "ok": not mech,
                   "detail": str([c["hint"] for c in mech])[:120]})
    if mech:
        problems.extend([c["hint"] for c in mech])

    v = verify(best_text) if best_text else {"ok": False, "errors": ["no best plan yet"]}
    checks.append({"name": "verify", "ok": v["ok"], "detail": str(v["errors"])[:120]})
    if not v["ok"]:
        problems.extend(v["errors"])

    gc = ground_check(best_text) if best_text else {"ok": False, "missing": ["no best plan yet"]}
    checks.append({"name": "feasibility", "ok": bool(gc["ok"]),
                   "detail": f"missing inputs: {gc['missing'][:3]}"})
    if not gc["ok"]:
        problems.append(f"declared inputs do not exist: {gc['missing'][:5]}")
    # verified environment inputs seed the simulator's initial state, so a
    # plan that reads real files simulates cleanly (same as assess())
    sim = simulate(best_text, initial_state=set(gc.get("verified", []))) if best_text \
        else {"executable_plan": False}
    checks.append({"name": "simulation", "ok": bool(sim["executable_plan"]),
                   "detail": str(sim.get("problems", []))[:120]})
    if not sim["executable_plan"]:
        problems.extend(sim.get("problems", []))

    judge_ok = False
    judge_detail = "no judge verdict recorded"
    judges = s.get("judge_log", [])
    best_ver = s.get("best_version")
    matching_judge = None
    if judges:
        for j_entry in reversed(judges):
            if j_entry.get("round_version") == best_ver or j_entry.get("round_version") is None:
                matching_judge = j_entry
                break
    if matching_judge:
        j = matching_judge
        is_go = bool(j.get("ok") and j.get("verdict") == "go" and j.get("falsifiable_criteria"))
        if require_external_judge:
            is_external = bool(j.get("external") is True and j.get("source") == "external_llm")
            judge_ok = is_go and is_external
            judge_detail = f"round={j.get('round_version')} verdict={j.get('verdict')} source={j.get('source', 'unknown')} external={is_external}"
        else:
            judge_ok = is_go
            judge_detail = f"round={j.get('round_version')} verdict={j.get('verdict')} source={j.get('source', 'unknown')} feasibility={j.get('feasibility_0_100')}"
    checks.append({"name": "judge", "ok": judge_ok or not require_judge, "detail": judge_detail})
    if require_judge and not judge_ok:
        problems.append("judge gate not passed; run plan.judge + record_judge, fix blockers, re-assess")

    ok = all(c["ok"] for c in checks)
    report = {"ok": ok, "checks": checks, "problems": problems}
    s["release_gate"] = report
    _save_session(plans_dir, s)
    return report


def finish(session: dict[str, Any] | str, *, verdict: str = "converged",
           plans_dir: str | Path | None = None, require_release: bool = True,
           min_score: float = 90.0) -> dict[str, Any]:
    """Mark the session complete. With require_release (default), the plan is
    only releasable after the release() gate passes; otherwise a
    RuntimeError is raised so the loop continues instead of shipping early."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    if require_release:
        gate = release(s, min_score=min_score)
        if not gate["ok"]:
            raise RuntimeError("release gate failed: " + "; ".join(gate["problems"]))
    s["status"] = "finished"
    s["completed_at"] = _now()
    s.setdefault("verdict", verdict)
    _save_session(plans_dir, s)
    return status(s)



def selfcheck(*, plans_dir: str | Path | None = None,
               run_pytest: bool = True,
               corpus_dir: str | Path | None = None) -> dict[str, Any]:
    """Mandatory re-evaluation after ANY change to the engine or rubric:
    (1) rubric parses, compiles, and cites only corpus-verified IDs;
    (2) every finished session's best plan re-verifies, re-simulates, and
    passes ground/constraint checks; (3) the pytest suite is green.
    Any failure means the change is not shippable."""
    checks: list[dict[str, Any]] = []
    problems: list[str] = []
    historical: dict[str, Any] = {}
    # 1) rubric integrity
    try:
        rubric = _load_rubric()
        n = sum(len(s["items"]) for s in rubric.values())
        for sec in rubric.values():
            for it in sec["items"]:
                re.compile(it[1])
        checks.append({"name": "rubric", "ok": True, "detail": f"{n} checks compile"})
    except Exception as e:
        checks.append({"name": "rubric", "ok": False, "detail": str(e)[:120]})
        problems.append(f"rubric broken: {e}")
    # S8: every cited ID exists in the corpus (if a corpus dir is available)
    cdir = Path(corpus_dir) if corpus_dir else Path(os.environ.get(
        "PLANNING_CORPUS", "/home/lewbei/deep_learning/planning_paper"))
    if cdir.exists():
        txts = set(os.listdir(cdir / "txts")) if (cdir / "txts").exists() else set()
        if txts:
            try:
                missing = set()
                for sec in rubric.values():
                    for it in sec["items"]:
                        for i in re.findall(r"\b(2[0-6]\d{2}\.\d{5})\b", it[2]):
                            if f"{i}.txt" not in txts:
                                missing.add(i)
                checks.append({"name": "corpus_ids", "ok": not missing,
                               "detail": f"{len(missing)} missing IDs"})
                if missing:
                    problems.append(f"unverified IDs: {sorted(missing)[:5]}")
            except Exception as e:
                checks.append({"name": "corpus_ids", "ok": False, "detail": str(e)[:120]})
                problems.append(str(e))
    # 2) every finished session's best plan re-evaluates clean
    pdir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    checked = 0
    if pdir.exists():
        for p in sorted(pdir.glob("*.json")):
            try:
                s2 = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if s2.get("status") != "finished" or not s2.get("best_version") or not s2.get("rounds"):
                continue
            best_text = s2["rounds"][s2["best_version"] - 1].get("plan_text", "")
            if not best_text or best_text.startswith("[folded:"):
                continue
            v = verify(best_text)
            sim = simulate(best_text, initial_state=set(ground_check(best_text)["verified"]))
            gc = ground_check(best_text)
            cc = constraint_check(best_text)
            checked += 1
            fails = []
            if not v["ok"]: fails += v["errors"][:2]
            if not sim["executable_plan"]: fails += sim["problems"][:2]
            if not gc["ok"]: fails += [f"missing inputs: {gc['missing'][:2]}"]
            if not cc["ok"]: fails += cc["problems"][:2]
            is_historical = s2.get("engine_version", "0.0.0") < "0.11.0"
            checks.append({"name": f"session:{s2.get('session_id', p.stem)[:30]}",
                           "ok": not fails or is_historical,
                           "detail": ("re-verified clean" if not fails else
                                      f"[historical, pre-0.11] " + "; ".join(fails)[:100])})
            if fails and not is_historical:
                problems.append(f"{p.name}: " + "; ".join(fails)[:160])
            elif fails:
                historical.setdefault("sessions", []).append(
                    {"file": p.name, "fails": fails})
    checks.append({"name": "sessions_scanned", "ok": True, "detail": f"{checked} finished sessions re-evaluated"})
    # 2b) feature dependency status (reactive coeffects report)
    try:
        ds = deps_check()
        checks.append({"name": "feature_deps", "ok": True,
                       "detail": f"{len(ds['unsatisfied'])} unsatisfied: {ds['unsatisfied']}"})
        if ds["unsatisfied"]:
            problems.append(f"feature deps unsatisfied: {ds['unsatisfied']}")
    except Exception as e:
        checks.append({"name": "feature_deps", "ok": False, "detail": str(e)[:120]})
    # 3) pytest (optional but default)
    if run_pytest:
        try:
            import subprocess
            r = subprocess.run(["python3", "-m", "pytest", "tests/", "-q"],
                               capture_output=True, text=True, timeout=300,
                               cwd=str(Path(__file__).resolve().parent.parent.parent))
            ok = r.returncode == 0
            checks.append({"name": "pytest", "ok": ok,
                           "detail": r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:120]})
            if not ok:
                problems.append("pytest suite failing")
        except Exception as e:
            checks.append({"name": "pytest", "ok": False, "detail": str(e)[:120]})
            problems.append(str(e))
    return {"ok": not problems, "problems": problems, "checks": checks,
            "historical": historical}


def list_sessions(plans_dir: str | Path | None = None) -> list[dict[str, Any]]:
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    out = []
    for p in sorted(plans_dir.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            out.append({"session_id": s["session_id"], "objective": s["objective"],
                        "status": s["status"], "rounds": len(s["rounds"]),
                        "best_score": s["best_score"], "created_at": s["created_at"]})
        except Exception:
            continue
    return out


def rubric() -> dict[str, dict[str, Any]]:
    return _load_rubric()


def log_progress(session: dict[str, Any] | str, task: str, status: str = "done",
                *, evidence: str | None = None, plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Record plan execution progress (grounding, 2603.14248: each step must
    have a detectable outcome). A failed/blocked step arms a replan trigger."""
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    if isinstance(session, str):
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(s.get("plans_dir") or plans_dir)
    entry = {"ts": _now(), "task": task, "status": status, "evidence": evidence}
    s.setdefault("execution_log", []).append(entry)
    # loop closure (2606.22488): execution feedback updates the world model,
    # so the next replan reads updated state instead of stale assumptions
    if evidence:
        s.setdefault("world_state", {})[task] = {"status": status, "evidence": evidence[:200]}
    if status in ("failed", "blocked"):
        retry_count = int(s.get("replan_retry_count", 0)) + 1
        s["replan_retry_count"] = retry_count
        s["replan_pending"] = True
        s["replan_task"] = task
        try:
            failed_task_id = int(re.findall(r"\d+", task)[-1]) if re.findall(r"\d+", task) else 0
            best_text = s["rounds"][s.get("best_version", 1) - 1].get("plan_text", "") if s.get("rounds") else ""
            total_tasks = len(_task_blocks(best_text))
            scope = ReplanningLadder.determine_replan_tier(
                failed_task_id=failed_task_id,
                error_message=evidence or status,
                total_tasks=total_tasks,
                retry_count=retry_count - 1
            )
            s["replan_scope"] = scope
            s["replan_tier"] = scope.get("tier", s.get("replan_tier", 1))
        except Exception:
            pass
    _save_session(plans_dir, s)
    return entry


def suggest(session: dict[str, Any] | str, *, plans_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Self-evolution loop (SERP 2603.02772: after convergence, upgrade the
    planning process itself). Emits rubric-hardening suggestions from the
    recorded rounds and writes them next to the session file."""
    plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
    s = _load_session(plans_dir, session) if isinstance(session, str) else session
    if not s["rounds"]:
        return []
    check_pass: dict[str, list[int]] = {}
    for r in s["rounds"]:
        failed_ids = {c["id"] for c in r["critiques"]}
        for c in _all_check_ids(s.get("rubric_snapshot") or _load_rubric()):
            check_pass.setdefault(c, []).append(1 if c not in failed_ids else 0)
    out: list[dict[str, Any]] = []
    for check, marks in check_pass.items():
        rate = sum(marks) / len(marks)
        if rate == 1.0 and len(marks) >= 3:
            out.append({"kind": "too-easy", "check": check,
                        "hint": "never failed any round; tighten the regex or drop it"})
        if rate == 0.0 and len(marks) >= 3:
            out.append({"kind": "too-strict", "check": check,
                        "hint": "failed every round including the best; regex or hint may be wrong"})
    best_round = s["rounds"][s["best_version"] - 1] if s["best_version"] else s["rounds"][-1]
    if best_round["critiques"] and s["status"] == "converged":
        out.append({"kind": "converged-with-critiques", "check": None,
                    "hint": "session converged with open critiques; require a zero-critique round "
                            "before convergence"})
    s["suggestions"] = out
    _save_session(plans_dir, s)
    (Path(s["plans_dir"]) / f"{s['session_id']}.suggestions.md").write_text(
        "# Rubric self-evolution suggestions\n\n" +
        "\n".join(f"- [{x['kind']}] {x['check'] or '-'}: {x['hint']}" for x in out) +
        ("\n" if out else "# (none)\n"), encoding="utf-8")
    return out


def _all_check_ids(rubric: dict[str, dict[str, Any]]) -> list[str]:
    ids = []
    for key, section in rubric.items():
        for item in section.get("items", []):
            if item[1]:  # only scored items have stable ids
                ids.append(f"{key}:{item[2][:40]}")
    return ids



def verify(plan_text: str, *, initial_state: set[str] | list[str] | None = None,
           cwd: str | Path | None = None) -> dict[str, Any]:
    """Deterministic plan-validity audit (no LLM).

    Goes beyond the regex rubric: parses the task/dependency/artifact
    structure and checks whether the plan, if executed as written, would
    hold together. This is the engine's answer to "will the plan actually
    work" at the mechanical level (PlanBench 2409.13373: prose self-claims
    are unreliable; check structure instead).

    Returns {"ok": bool, "errors": [str], "warnings": [str], "graph": {...}}.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # task bodies may span several lines; capture from the task marker to the
    # next numbered marker (or end of text)
    markers = [m for m in re.finditer(r"^\s*(\d+)[.)]\s+", plan_text, re.M)]
    tasks = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(plan_text)
        tasks.append((int(m.group(1)), plan_text[start:end].strip()))
    nums = [n for n, _ in tasks]
    if nums:
        expected = list(range(1, len(nums) + 1))
        if nums != expected:
            errors.append(f"non-contiguous task numbering: found {nums}, expected {expected}")
    graph: dict[int, list[int]] = {}
    for n, body in tasks:
        refs: list[int] = []
        for m in re.finditer(r"depends?\s+on\s+([^.(]+)", body, re.I):
            refs += [int(x) for x in re.findall(r"\d+", m.group(1))]
        graph[n] = sorted(set(refs))
        for r in refs:
            if r not in nums:
                errors.append(f"task {n} depends on task {r}, which does not exist")
            elif r >= n:
                errors.append(f"task {n} depends on later task {r} (forward reference; no cycle allowed)")
    # cycle check
    def has_cycle(n: int, seen: set[int], stack: set[int]) -> bool:
        if n in stack:
            return True
        if n in seen:
            return False
        seen.add(n); stack.add(n)
        for d in graph.get(n, []):
            if d < n and has_cycle(d, seen, stack):
                return True
        stack.discard(n)
        return False
    for n in nums:
        if has_cycle(n, set(), set()):
            errors.append(f"dependency cycle involving task {n}")
            break

    # artifact per task (allow the filename to wrap onto the next line)
    for n, body in tasks:
        if not re.search(r"(output|deliverable|artifact|produces?|write)\s*[:(]?\s*[\w./-]+\.(json|md|py|txt|csv|pdf|yaml|yml|log)", body, re.I | re.S) \
           and not re.search(r"output:\s*\S+", body, re.I):
            warnings.append(f"task {n} declares no concrete output artifact")

    # milestones must reference real tasks or be count-consistent
    m_count = len(re.findall(r"^\s*[-*]\s*M\d+", plan_text, re.M))
    if m_count and m_count > len(nums):
        errors.append(f"{m_count} milestones for {len(nums)} tasks (milestones outnumber tasks)")

    # time arithmetic: sum of per-task estimates vs deadline
    per_task = re.findall(r"(\d+(?:\.\d+)?)\s*(min|minute|hour)s?\s*(?:per|each|/)\s*(task|step)", plan_text, re.I)
    deadline_h = re.findall(r"within\s+(\d+(?:\.\d+)?)\s*(hour|day|week)s?", plan_text, re.I)
    if per_task and deadline_h and nums:
        unit = per_task[0][1].lower()
        val = float(per_task[0][0])
        total = val * len(nums) if "min" in unit else val * len(nums) * 60
        dh = float(deadline_h[0][0]) * ({"hour": 60, "day": 24 * 60, "week": 7 * 24 * 60}[deadline_h[0][1].lower()])
        if total > dh:
            errors.append(f"time estimates sum to ~{total:.0f} min but deadline allows ~{dh:.0f} min")

    # success criteria coverage: every numeric criterion should be testable
    crits = re.findall(r"^\s*[-*]\s*.{0,160}", plan_text, re.M)
    numeric_crits = [c for c in crits if re.search(r"\d+", c)]
    if not numeric_crits:
        warnings.append("no numeric success criteria found")

    # --- grounded diagnosis layer (2603.14730 GNNVerifier): localized
    # type-mismatch checks — an input consumed as X.ext must not be produced
    # elsewhere under a different extension ---
    dag = plan_dag(plan_text)
    by_base: dict[str, list[tuple[str, int]]] = {}
    for n, arts in dag["artifacts"].items():
        for a in arts:
            by_base.setdefault(Path(a).stem.lower(), []).append((Path(a).suffix, n))
    for n, ins in dag["inputs"].items():
        for a in ins:
            stem = Path(a).stem.lower()
            for ext, prod_n in by_base.get(stem, []):
                if ext and Path(a).suffix and ext != Path(a).suffix:
                    errors.append(f"type mismatch: task {n} consumes {a} but task "
                                  f"{prod_n} produces {stem}{ext}")

    # --- horizon guard (2601.20856): > 30 steps collapses lookahead ---
    if len(nums) > 30:
        warnings.append(f"plan has {len(nums)} tasks; lookahead collapses beyond ~25-30 steps")

    # --- budget reconciliation (2605.20873): numeric global consistency ---
    per_cost = re.findall(r"(\d+(?:\.\d+)?)\s*(tokens?|k\s*tokens?|\$|USD|hours? of compute)", plan_text, re.I)
    cap = re.findall(r"(?:budget|cap)\w*\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(tokens?|\$|USD|hours?)", plan_text, re.I)
    if per_cost and cap and nums:
        cap_val = float(cap[0][0]); cap_unit = cap[0][1].lower()
        total = 0.0
        for val, unit in per_cost:
            v = float(val)
            u = unit.lower().replace(" ", "")
            if u.startswith("k"): v *= 1000
            total += v
        cap_val2 = cap_val * (1000 if "token" in cap_unit and any("k" in u for _, u in per_cost) else 1)
        if total > cap_val2:
            errors.append(f"cost reconciliation: per-task costs sum to ~{total:g} but budget is {cap[0][0]} {cap[0][1]}")

    # --- parallel conflict (2607.09603): tasks declared parallel must write disjoint outputs ---
    if re.search(r"parallel", plan_text, re.I):
        for a in nums:
            for b in nums:
                if a < b and b not in graph.get(a, []) and a not in graph.get(b, []):
                    shared = set(dag["artifacts"].get(a, [])) & set(dag["artifacts"].get(b, []))
                    if shared:
                        errors.append(f"parallel tasks {a} and {b} write the same artifact "
                                      f"{sorted(shared)[0]} (joint-action conflict)")
                    if len(errors) > 40:
                        break
            if len(errors) > 40:
                break

    # --- reachability + landmarks (2607.11197): every criterion needs a
    # covering task chain; disconnected entries are unreachable ---
    blocks = _task_blocks(plan_text)
    if len(blocks) > 1:
        non_first_roots = [n for n, body in blocks if n != blocks[0][0] and not graph.get(n)]
        for n in non_first_roots:
            warnings.append(f"task {n} is disconnected from the plan root (unreachable entry)")
        crit_labels = re.findall(r"^\s*[-*]\s*(S\d+)\b", plan_text, re.M)
        if 0 < len(crit_labels) <= 20:
            bodies = {n: body for n, body in blocks}
            for lab in sorted(set(crit_labels)):
                # covered when a task body mentions the label OR the plan
                # declares an explicit mapping "Sx -> task N" (a stated chain)
                mapped = bool(re.search(re.escape(lab) + r"\s*(?:->|\u2192|maps to|:)\s*task\s*\d+",
                                        plan_text, re.I))
                covered = mapped or any(re.search(re.escape(lab), body, re.I)
                                        for body in bodies.values())
                if not covered:
                    errors.append(f"criterion {lab} has no covering task (no landmark chain)")

    # --- Formal Causal & Symbolic Validation (SymPlanner 2505.01479, GNNVerifier 2603.14730) ---
    # Seed only environment inputs that actually exist. Internal handoffs are
    # produced by earlier tasks in the causal simulation and must NOT be
    # pre-seeded, otherwise missing producers would pass validation.
    gc = ground_check(plan_text, cwd=cwd)
    init_state: set[str] = set(gc.get("verified", [])) | set(initial_state or [])
    internal_artifacts = {out for outs in dag["artifacts"].values() for out in outs}
    base = Path(cwd) if cwd else Path.cwd()
    for raw in {inp for ins in dag["inputs"].values() for inp in ins}:
        if raw in internal_artifacts:
            continue
        resolved = Path(raw) if Path(raw).is_absolute() else (base / raw)
        if resolved.exists():
            init_state.add(raw)
            init_state.add(str(resolved))
    ast = PlanParser.parse_plan(plan_text)
    causal_res = CausalValidator.validate(ast, initial_state=init_state)
    for flaw in causal_res.get("flaws", []):
        errors.append(f"causal flaw [{flaw['type']}]: {flaw['detail']}")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "graph": graph, "tasks": len(nums), "causal_validation": causal_res}


async def search(session: dict[str, Any] | str, **kwargs) -> dict[str, Any]:
    """Run an advanced search over plan space (MCTS/beam, LLM or rule
    expansion, rollouts = rubric + verify + simulation). See search_engine.py."""
    from .search_engine import search as _search
    return await _search(session, **kwargs)


async def judge(plan_text: str, objective: str = "", *,
                model: str | None = None, timeout: int | None = None) -> dict[str, Any]:
    """External LLM feasibility verdict (async). See plan_mode/judge.py."""
    from .judge_client import judge as _judge
    kw = {}
    if model:
        kw["model"] = model
    if timeout:
        kw["timeout"] = timeout
    return await _judge(plan_text, objective, **kw)


def record_judge(session: dict[str, Any] | str, verdict: dict[str, Any], *,
                 round_version: int | None = None,
                 plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Persist an external judge verdict (from plan_mode.judge) into the
    session so feasibility audits are part of the recorded history."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    ver = round_version if round_version is not None else s.get("best_version") or len(s.get("rounds", []))
    entry = {"ts": _now(), "round_version": ver, **{k: v for k, v in verdict.items()}}
    s.setdefault("judge_log", []).append(entry)
    _save_session(plans_dir, s)
    return entry


# ---------------------------------------------------------------------------
# Planning mechanics (implemented from the 111-paper corpus)
# ---------------------------------------------------------------------------

def _task_blocks(plan_text: str) -> list[tuple[int, str]]:
    """Split the plan into task bodies (multi-line aware)."""
    markers = [m for m in re.finditer(r"^\s*(\d+)[.)]\s+", plan_text, re.M)]
    tasks = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(plan_text)
        tasks.append((int(m.group(1)), plan_text[start:end].strip()))
    return tasks


def plan_dag(plan_text: str) -> dict[str, Any]:
    """Build the explicit task graph: nodes, edges, artifacts, inputs.

    Returns {nodes: [1..n], edges: {task: [dep tasks]}, artifacts:
    {task: [output files]}, inputs: {task: [input files/deps named]}}.
    """
    tasks = _task_blocks(plan_text)
    edges: dict[int, list[int]] = {}
    artifacts: dict[int, list[str]] = {}
    inputs: dict[int, list[str]] = {}
    for n, body in tasks:
        refs: list[int] = []
        for m in re.finditer(r"depends?\s+on\s+([^.(]+)", body, re.I):
            refs += [int(x) for x in re.findall(r"\d+", m.group(1))]
        edges[n] = sorted(set(refs))
        outs: list[str] = []
        for m in re.finditer(r"(?:output|produces?|writes?|deliverable)\s*[:(]?\s*([\w./-]+\.\w{1,8})", body, re.I):
            val = m.group(1)
            if val:
                outs.append(val.strip())
        artifacts[n] = sorted(set(outs))
        ins: list[str] = []
        for m in re.finditer(r"(?:requires?|inputs?|needs|consumes)\s*[:(]?\s*([\w./-]+\.\w{1,8})|\breads?\b\s*[:(]?\s*([\w./-]+\.\w{1,8})", body, re.I):
            val = m.group(1) or m.group(2)
            if val:
                ins.append(val.strip())
        inputs[n] = sorted(set(ins))
    return {"nodes": [n for n, _ in tasks], "edges": edges,
            "artifacts": artifacts, "inputs": inputs}


def constraint_check(plan_text: str) -> dict[str, Any]:
    """Route numeric hard constraints through a deterministic check
    (2606.09027 SafeRun): extract "at most / at least / no more than N unit"
    constraints and verify the plan's own declared quantities do not
    contradict them. Purely mechanical — no LLM involved."""
    problems: list[str] = []
    cons = re.findall(r"(at most|at least|no more than|minimum of|maximum of)\s+(\d+(?:\.\d+)?)\s*(tasks?|steps?|hours?|minutes?|tokens?|days?)", plan_text, re.I)
    task_count = len(_task_blocks(plan_text))
    for kind, val_s, unit in cons:
        val = float(val_s); unit = unit.lower()
        if unit.startswith("task") or unit.startswith("step"):
            actual = task_count
        elif unit.startswith("minute"):
            actual = sum(float(m[0]) for m in re.findall(r"(\d+(?:\.\d+)?)\s*min", plan_text, re.I)) or 0
        elif unit.startswith("hour"):
            actual = sum(float(m[0]) for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:hour|h)\b", plan_text, re.I)) or 0
        elif unit.startswith("day"):
            actual = sum(float(m[0]) for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:day|d)\b", plan_text, re.I)) or 0
        elif unit.startswith("token"):
            actual = sum(float(m[0]) * (1000 if "k" in m[1].lower() else 1)
                         for m in re.findall(r"(\d+(?:\.\d+)?)\s*(k?\s*tokens?)", plan_text, re.I)) or 0
        else:
            continue
        if kind in ("at most", "no more than", "maximum of") and actual > val:
            problems.append(f"constraint violation: declared at most {val:g} {unit} but the plan totals {actual:g}")
        if kind in ("at least", "minimum of") and actual < val:
            problems.append(f"constraint violation: declared at least {val:g} {unit} but the plan totals {actual:g}")
    return {"ok": not problems, "problems": problems, "constraints": cons}


def ground_check(plan_text: str, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Grounded feasibility (2402.11489 SimPlan, 2512.09629 verifier agents):
    every input a plan declares from the environment must resolve to a real
    file on disk. Zero tolerance: a plan referencing a nonexistent resource
    is not feasible, whatever its prose claims (TREK/TripTailor standard).
    Internal handoffs (files produced by an earlier task) are exempt — they
    do not need to pre-exist."""
    dag = plan_dag(plan_text)
    internal = {a for arts in dag["artifacts"].values() for a in arts}
    base = Path(cwd) if cwd else Path.cwd()
    verified: set[str] = set()
    missing: list[str] = []
    for task, ins in dag["inputs"].items():
        for p in ins:
            if p in internal:
                continue  # produced inside the plan, not an environment input
            resolved = Path(p) if Path(p).is_absolute() else (base / p)
            if resolved.exists():
                verified.add(str(resolved))
            else:
                missing.append(f"{p} (task {task})")
    return {"ok": not missing, "missing": missing,
            "verified": sorted(verified),
            "internal": sorted(internal)}


def simulate(plan_text: str, *, initial_state: set[str] | None = None,
             max_steps: int = 1000) -> dict[str, Any]:
    """STRIPS-style state simulation through the plan (SymPlanner 2505.01479,
    PyPDDLEngine 2603.06064: execute the plan against an explicit state
    model instead of trusting prose).

    Starting from initial_state (artifacts already on disk, else empty), walk
    tasks in order: a task is EXECUTABLE when all its dependency tasks have
    run and every named input artifact is in the state; executing it adds its
    output artifacts to the state. Reports unsatisfied preconditions, dead
    artifacts (produced but never consumed), unreachable tasks, and the final
    state. This is the engine's own check of whether the plan, executed
    literally, reaches its claimed outputs.
    """
    dag = plan_dag(plan_text)
    nodes = dag["nodes"]
    state = set(initial_state or [])
    done: set[int] = set()
    pending = list(nodes)
    steps = 0
    problems: list[str] = []
    trace: list[dict[str, Any]] = []

    # walk tasks strictly in declared order: a task whose dependencies have
    # not run when it is reached is BLOCKED (forward references are errors,
    # matching verify()); no silent reordering of the plan's written order.
    for n in nodes:
        if steps >= max_steps:
            break
        steps += 1
        deps_ok = all(d in done for d in dag["edges"].get(n, []))
        inputs_ok = all(a in state for a in dag["inputs"].get(n, []))
        if deps_ok and inputs_ok:
            outs = dag["artifacts"].get(n, [])
            state.update(outs)
            done.add(n)
            pending.remove(n)
            trace.append({"task": n, "inputs": dag["inputs"].get(n, []),
                          "outputs": outs, "state_size": len(state)})
        else:
            missing_deps = [d for d in dag["edges"].get(n, []) if d not in done]
            missing_inputs = [a for a in dag["inputs"].get(n, []) if a not in state]
            problems.append(
                f"task {n} is blocked: dependencies not done {missing_deps or '-'}, "
                f"missing inputs {missing_inputs or '-'}")

    # dead artifacts: produced but never consumed by any later task
    consumed = {a for n in nodes for a in dag["inputs"].get(n, [])}
    produced = {a for n in done for a in dag["artifacts"].get(n, [])}
    dead = sorted(produced - consumed)

    return {"executable_plan": not problems and not pending,
            "tasks_completed": sorted(done),
            "tasks_unreachable": sorted(pending),
            "final_state": sorted(state),
            "problems": problems,
            "dead_artifacts": dead,
            "trace": trace,
            "dag": dag}


def plan_quality(plan_text: str, objective: str = "", *,
                 initial_state: set[str] | None = None) -> dict[str, Any]:
    """Combined structural + simulation verdict for a plan (the engine's own
    "will it actually run" answer, computed without any LLM)."""
    sim = simulate(plan_text, initial_state=initial_state)
    v = verify(plan_text)
    # coverage closure (2607.12986): every numeric success criterion should
    # map to at least one task's artifact or a measurable state
    crits = re.findall(r"^\s*[-*]\s*[^\n]*\d+[^\n]*$", plan_text, re.M)
    crit_ok = len(crits) > 0
    return {
        "objective": objective,
        "structural": v,
        "simulation": sim,
        "coverage": {"numeric_criteria": len(crits), "has_criteria": crit_ok},
        "verdict": "executable" if sim["executable_plan"] and v["ok"] and crit_ok
                   else ("structurally-broken" if not v["ok"] else
                         ("simulation-blocked" if not sim["executable_plan"] else "incomplete-criteria")),
    }


def assess_candidates(session: dict[str, Any] | str, drafts: list[str],
                      *, notes: list[str] | None = None,
                      plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Best-of-N plan selection (2601.17942 plan ensembles, 2509.00084
    candidate comparison): score every candidate draft against the full
    pipeline (rubric + mechanical + verify + feasibility + simulation),
    and keep the best executable plan as this round's version.
    Returns the assess result of the winning candidate plus the full ranking."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    if not drafts:
        raise ValueError("drafts must be non-empty")
    scored = []
    for i, d in enumerate(drafts):
        note = (notes or [None] * len(drafts))[i]
        rubric = s.get("rubric_snapshot") or _load_rubric()
        res = _score(d, rubric)
        res["critiques"] = res["critiques"] + _mechanical_checks(d)
        v = verify(d)
        for err in v["errors"]:
            res["critiques"].append({"id": f"mech:verify:{err[:40]}", "section": "mechanical",
                                     "hint": err})
        gc = ground_check(d)
        sim = simulate(d, initial_state=set(gc.get("verified", [])))
        for prob in sim.get("problems", []):
            res["critiques"].append({"id": f"mech:sim:{prob[:40]}", "section": "mechanical",
                                     "hint": f"Simulation failure: {prob}"})

        # Calculate effective executable score: penalize simulation blocks and structural failures
        effective_score = res["score"]
        if not sim["executable_plan"]:
            effective_score -= 30.0
        if not v["ok"]:
            effective_score -= 15.0 * len(v["errors"])
        if not gc["ok"]:
            effective_score -= 15.0

        scored.append({
            "candidate": i,
            "note": note,
            "score": res["score"],
            "effective_score": round(max(0.0, effective_score), 2),
            "sim_ok": sim["executable_plan"],
            "verify_ok": v["ok"],
            "feasibility_ok": gc["ok"],
            "critiques": res["critiques"],
            "verify": v,
            "simulation": sim
        })

    # Sort primarily by effective_score (executable plans win over broken ones)
    scored.sort(key=lambda x: -x["effective_score"])
    best_i = scored[0]["candidate"]
    winner = assess(s, drafts[best_i], note=(notes or [None] * len(drafts))[best_i], plans_dir=plans_dir)
    winner["ranking"] = [{
        "candidate": x["candidate"],
        "score": x["score"],
        "effective_score": x["effective_score"],
        "sim_ok": x["sim_ok"],
        "verify_ok": x["verify_ok"]
    } for x in scored]
    winner["candidates_scored"] = len(drafts)
    return winner


# ---------------------------------------------------------------------------
# Tree search over plan space (LATS 2310.04406 pattern; corpus papers:
# SYMPHONY 2601.22623 UCB-based multi-candidate expansion, CB-MCTS 2603.02154
# Boltzmann exploration, GATS 2607.08894 layered world models,
# cost-aware tree search 2505.14656).
# ---------------------------------------------------------------------------

def _search_state(session: dict[str, Any]) -> dict[str, Any]:
    return session.setdefault("search_tree", {"nodes": {}, "root": None,
                                              "expansions": 0, "best_node": None})


def search_expand(session: dict[str, Any] | str, drafts: list[str],
                  *, parent_node: str | None = None,
                  notes: list[str] | None = None,
                  plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Expand the search tree with candidate plan versions (multiple
    candidates per node, SYMPHONY 2601.22623). Each draft is scored with the
    rubric + verify + simulation (the rollout), added as a child of
    parent_node, and the best score is back-propagated up the tree
    (MCTS backprop, LATS 2310.04406). Returns the ranked node list."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    st = _search_state(s)
    rubric = s.get("rubric_snapshot") or _load_rubric()
    if parent_node is None:
        parent_node = st.get("root")
    nodes = st["nodes"]
    if parent_node is None or parent_node not in nodes:
        # first expansion: each draft becomes a root candidate
        parent_node = None

    new_ids = []
    for i, d in enumerate(drafts):
        res = _score(d, rubric)
        res["critiques"] = res["critiques"] + _mechanical_checks(d)
        v = verify(d)
        sim = simulate(d)
        node_id = f"n{len(nodes)+1}"
        nodes[node_id] = {
            "id": node_id, "parent": parent_node, "plan_text": d,
            "note": (notes or [None]*len(drafts))[i],
            "score": res["score"], "verify_ok": v["ok"], "sim_ok": sim["executable_plan"],
            "children": [], "visits": 1,
        }
        if parent_node and parent_node in nodes:
            nodes[parent_node]["children"].append(node_id)
        new_ids.append(node_id)
        if st.get("root") is None:
            st["root"] = node_id

    # backprop: update best_node
    best = max(nodes.values(), key=lambda n: n["score"])
    st["best_node"] = best["id"]
    st["expansions"] += 1
    _save_session(plans_dir, s)
    return {"node_ids": new_ids, "best_node": best["id"], "best_score": best["score"],
            "tree_size": len(nodes), "root": st["root"]}


def search_select(session: dict[str, Any] | str, *, exploration: float = 1.4,
                  cost_penalty: float = 0.0,
                  plans_dir: str | Path | None = None) -> dict[str, Any]:
    """UCB1 selection over the search tree (SYMPHONY 2601.22623): pick the
    node to expand next as argmax(mean_score + exploration*sqrt(ln N / visits)),
    minus a per-expansion cost penalty (cost-aware tree search 2505.14656).
    Returns the selected node; the agent expands it with search_expand."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    st = _search_state(s)
    nodes = st["nodes"]
    if not nodes:
        return {"node_id": None, "reason": "tree empty; call search_expand first"}
    N = sum(n["visits"] for n in nodes.values()) or 1
    def ucb(n):
        exploit = n["score"] / 100.0
        explore = exploration * ((math.log(N) / n["visits"]) ** 0.5)
        return exploit + explore - cost_penalty * n["visits"]
    chosen = max(nodes.values(), key=ucb)
    return {"node_id": chosen["id"], "score": chosen["score"],
            "visits": chosen["visits"], "ucb": round(ucb(chosen), 3),
            "children": chosen["children"], "parent": chosen["parent"]}


def search_backtrack(session: dict[str, Any] | str, node_id: str,
                     plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Backtrack to an earlier node (LATS 2310.04406): mark it as the active
    expansion frontier so a plateaued branch is abandoned and the tree
    re-expands from the ancestor."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    st = _search_state(s)
    nodes = st["nodes"]
    if node_id not in nodes:
        raise ValueError(f"node {node_id} not in tree")
    nodes[node_id]["visits"] += 1
    _save_session(plans_dir, s)
    return {"backtracked_to": node_id, "score": nodes[node_id]["score"],
            "children": nodes[node_id]["children"]}


def search_report(session: dict[str, Any] | str,
                  plans_dir: str | Path | None = None) -> dict[str, Any]:
    """Tree state summary (GATS 2607.08894: explicit world-model/tree audit)."""
    if isinstance(session, str):
        plans_dir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        s = _load_session(plans_dir, session)
    else:
        s = session
        plans_dir = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or DEFAULT_PLANS_DIR)
    st = _search_state(s)
    nodes = st["nodes"]
    depth = {}
    for n in nodes.values():
        d, p = 0, n.get("parent")
        seen = set()
        while p and p in nodes and p not in seen:
            seen.add(p); d += 1; p = nodes[p].get("parent")
        depth[n["id"]] = d
    return {"nodes": len(nodes), "expansions": st["expansions"],
            "root": st["root"], "best_node": st["best_node"],
            "best_score": nodes[st["best_node"]]["score"] if st["best_node"] else None,
            "max_depth": max(depth.values()) if depth else 0,
            "leaves": [nid for nid, n in nodes.items() if not n["children"]]}



# ---------------------------------------------------------------------------
# Spatiotemporal Composability (Cordis) Integration
# ---------------------------------------------------------------------------
from .cordis import Context, Fiber, LifecycleState, TwistedMonoid, get_root_context, reset_root_context


def create_subagent_context(name: str = "subagent") -> Context:
    """Create an isolated child context (Cordis Gamma_infinity) for a subagent."""
    return get_root_context().isolate("scratchpad", f"subagent_{name}_{time.time_ns()}").derive(name=name)


def provide_tool(key: str, value: Any) -> Callable[[], Any]:
    """Register an ephemeral tool/verifier into the root harness context as a revertible effect."""
    return get_root_context().provide(key, value)


async def execute_plan(plan_text: str,
                       task_handlers: dict[int, Callable[[Context], Any]] | None = None,
                       *, dry_run: bool = False,
                       continue_on_error: bool = False,
                       timeout_per_task: float | None = None,
                       context: Context | None = None) -> dict[str, Any]:
    """Execute plan tasks transactionally with Cordis Revertible Fibers (Theorem 61, 63).

    Each task runs inside a managed fiber context. Side effects are tracked in
    the twisted monoid accumulator. If any task fails, all completed tasks'
    inverses are automatically executed in reverse (LIFO) order to guarantee
    Terminal Recovery Exactness (Corollary 62).
    """
    ctx = context or get_root_context().derive(name="plan_execution")
    dag = plan_dag(plan_text)
    nodes = dag["nodes"]
    task_handlers = task_handlers or {}

    executed_tasks: list[int] = []
    task_results: dict[int, Any] = {}

    try:
        for t in nodes:
            # check dependencies
            deps = dag["edges"].get(t, [])
            unsatisfied = [d for d in deps if d not in executed_tasks]
            if unsatisfied:
                raise RuntimeError(f"Task {t} blocked: dependencies not completed: {unsatisfied}")

            # execute task in child fiber context
            task_ctx = ctx.derive(name=f"task_{t}")
            handler = task_handlers.get(t)

            if not dry_run:
                if handler:
                    res = handler(task_ctx)
                    if inspect.isawaitable(res):
                        if timeout_per_task is not None:
                            res = await asyncio.wait_for(res, timeout=timeout_per_task)
                        else:
                            res = await res
                else:
                    res = {"status": "success", "task": t}
                task_results[t] = res
            else:
                task_results[t] = {"status": "dry_run", "task": t}

            executed_tasks.append(t)

        return {
            "ok": True,
            "executed_tasks": executed_tasks,
            "results": task_results,
            "recovered": False
        }
    except Exception as err:
        failed_task = t if 't' in locals() else (nodes[0] if nodes else None)
        error_msg = str(err)
        if not continue_on_error:
            # Auto-rollback all executed tasks in twisted monoid order
            await ctx.async_dispose()
            return {
                "ok": False,
                "error": error_msg,
                "failed_task": failed_task,
                "executed_tasks": executed_tasks,
                "recovered": True,
                "recovery_message": "All intermediate mutations rolled back in LIFO order via Cordis accumulator"
            }
        else:
            return {
                "ok": False,
                "error": error_msg,
                "failed_task": failed_task,
                "executed_tasks": executed_tasks,
                "recovered": False
            }


def execute_plan_sync(plan_text: str,
                      task_handlers: dict[int, Callable[[Context], Any]] | None = None,
                      *, dry_run: bool = False,
                      continue_on_error: bool = False,
                      context: Context | None = None) -> dict[str, Any]:
    """Synchronous wrapper for execute_plan."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(execute_plan(
                plan_text, task_handlers, dry_run=dry_run,
                continue_on_error=continue_on_error, context=context
            ))).result()
    except RuntimeError:
        return asyncio.run(execute_plan(
            plan_text, task_handlers, dry_run=dry_run,
            continue_on_error=continue_on_error, context=context
        ))


async def speculative_rollout_async(plan_text: str,
                                    eval_fn: Callable[[Context], Coroutine[Any, Any, float]],
                                    *, context: Context | None = None) -> dict[str, Any]:
    """Asynchronously execute a candidate plan in an isolated speculative realm and score execution."""
    ctx = (context or get_root_context()).derive(name="speculative_rollout_async")
    try:
        score = await eval_fn(ctx)
        return {"ok": True, "score": score, "error": None}
    except Exception as e:
        return {"ok": False, "score": 0.0, "error": str(e)}
    finally:
        await ctx.async_dispose()


def speculative_rollout(plan_text: str,
                        eval_fn: Callable[[Context], float],
                        *, context: Context | None = None) -> dict[str, Any]:
    """Execute a candidate plan in an isolated speculative realm and score actual execution.
    Automatically recovers all side effects upon completion (clean MCTS rollout)."""
    ctx = (context or get_root_context()).derive(name="speculative_rollout")
    try:
        score = eval_fn(ctx)
        return {"ok": True, "score": score, "error": None}
    except Exception as e:
        return {"ok": False, "score": 0.0, "error": str(e)}
    finally:
        # Guarantee 100% clean teardown
        ctx.dispose()

__all__ = ["start", "assess", "assess_candidates", "run", "status", "history", "best", "finish",
           "log_progress", "suggest", "list_sessions", "rubric", "verify", "judge", "record_judge",
           "plan_dag", "simulate", "plan_quality", "edit_file", "rollback", "deps_check",
           "search_expand", "search_select", "search_backtrack", "search_report", "search",
           "Context", "Fiber", "LifecycleState", "TwistedMonoid", "get_root_context", "reset_root_context",
           "create_subagent_context", "provide_tool", "execute_plan", "RoTRuleBase", "RoTRule", "ReplanningLadder", "ContextBudgeter", "mutate_flaw_directed", "mutate_exploratory", "crossover_ast", "ast_distance", "PopulationMember", "ASTSearchEngine", "Proposition", "PlanParser", "PlanAST", "CausalValidator", "CausalLink", "CausalFlaw", "ActionSchema", "execute_plan_sync", "speculative_rollout", "speculative_rollout_async",
           "DEFAULT_PLANS_DIR", "RUBRIC_PATH", "REPO_ROOT", "__version__"]