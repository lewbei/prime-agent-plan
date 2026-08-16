from plan_mode.memory_distiller import ContextBudgeter
import pytest
"""Engine v0.6 tests: folding, root-cause grouping, compat load, adaptive search."""
import json
from pathlib import Path

import plan_mode
from plan_mode import fold_history, simulate, verify, _load_session, _save_session

SKILL = Path(__file__).resolve().parent.parent

GOOD_PLAN = """# Goal
Goal: ship a small patch.

## Success criteria
- S1: patch applied and tests pass. Pass/fail: exit code 0. Deadline: within 1 day.

## Assumptions and unknowns
- Assumptions: environment ready. Unknowns: none (TBD).

## Plan structure
Declared structure: sections plus numbered tasks; abstraction layers: top-level summary plus detail blocks; robust to renaming (representation invariance); detail calibrated to task complexity. Tasks compose from a small set of atomic, orthogonal primitives (audit, edit, test, release).

## Dependency graph
Directed acyclic dependency graph (DAG) with predecessor/successor edges: 1 -> 2 -> 3. Coverage closure: every success criterion maps to a task.

## Step constraints (preconditions and effects)
- Precondition per task: deps done. Expected effect: output exists. Constraints re-anchored at each step where they apply. Dual correction: logical consistency AND physical feasibility. Multi-agent coordination: planner, executor, and an independent judge; handoffs are the output files.

## Milestones and go/no-go gates
- M1 after task 2: measurable signal: file exists; decision rule: halt/revise; go/no-go on failure.
- M2 after task 3: measurable signal: exit code 0; decision rule: go on zero failures.

## Verification machine (machine-checkable)
- Invariant: JSON parses; checked by a non-LLM script with assert. Verify by tool/script; validated by python. Step-wise validation with the engine's simulator. Two axes: local step validity and global goal reachability, checked separately. Adversarial check: negative samples try to falsify. Evaluation function: rubric score 0-100 per step (scoring rule). Dense per-step feedback: re-score immediately after each task. Corrective action when verification fails: fix locally; pre-execution critique over risky steps. Revision loop iterates only on verified failures. Lookahead plus backward value propagation from the goal. Limited commitment: commit at most 3 next steps, then re-evaluate. Process-based evaluation: score the reasoning and recovery steps, not only the final outcome.

## Risks and failure modes
- Risk list: regex overlap; failure classification: local fix vs global replan; mitigation: tests; fallback path: if one step fails, try the alternative route. Rollback: revert. Distractors (plausible-but-wrong assumptions) are rejected by evidence thresholds.

## Resources and cost estimates
- Time estimates: 20 min per task; budget: <= 10k tokens; cost reconciliation: cumulative check vs cap.

## Alternatives considered
- Alternative A: rewrite; rejected because risk outweighs benefit.

## Replanning policy
- Triggers: event-driven, threshold-driven: replan when a gate fails. Replan budget: cap 3 revisions. Scoped repair: smallest local fix; reuse the valid prefix. Pre-emptive failure enumeration: enumerate likely failure modes in advance, before drafting, as negative constraints. Acceptance threshold: stop when tests pass. Fresh replan immediately before timing-sensitive steps. After repeated failure, upgrade the planner itself (SERP).

## Grounding and progress detection
- Step outcomes: detect by file existence and exit code (how we know a step succeeded). Subgoal checkpoints: M1/M2. State restatement: restate state after each step; memoize to prevent goal drift. World probe: query the state model before refinement. Silent-failure detection: empty outputs detected by assertions. Incremental presentation: inspectable, interruptible step sequence.

## History and memory
- Lessons learned: cite past failures; revision strategy: address every critique id; diverse alternatives: best-of-N via assess_candidates. Reuse validated components. Evidence-traced revisions. Fold/prune plan history; cap context budget. Distill rules from failures. Template reusable across sessions.

## Escalation policy
- Cheapest viable action first; escalate only if needed. Isolate the root cause before escalating.

## Uncertainty awareness
- Uncertainty estimate per step; conservative switch when uncertainty is high.

## Immediate executability
- First action today: write the patch. Stop when tests pass (exit criterion). Refusal policy: abort if infeasible.

## Tasks
1. Audit. Depends on: none. Output: plan_improvements/a.md. Exit criterion: file exists; stop when reached. Time: 20 min. Confidence: high.
2. Edit. Depends on task 1. Output: plan_improvements/b.md. Exit criterion: diff under 50 lines. Time: 20 min. Confidence: high.
3. Test. Depends on task 2. Output: plan_improvements/report.txt. Exit criterion: S1 passes — pytest exit code 0. Time: 20 min. Confidence: high.
"""


def _mk_session(tmp_path, engine_version=None, rounds=4, score=90.0):
    sid = "t-" + str(abs(hash(str(tmp_path))) % 10**9)
    s = {
        "session_id": sid, "objective": "test", "created_at": "2026-08-13T00:00:00+00:00",
        "plans_dir": str(tmp_path), "rubric_version": "v4", "rubric_snapshot": {},
        "max_rounds": 8, "meta": {}, "rounds": [], "best_version": 3, "best_score": score,
        "status": "converged", "completed_at": None, "execution_log": [], "replan_pending": False,
        "suggestions": [],
    }
    if engine_version:
        s["engine_version"] = engine_version
    for i in range(1, rounds + 1):
        s["rounds"].append({"version": i, "ts": "t", "score": score + i, "delta": 1.0,
                            "critiques": [{"id": f"tasks:hint{i}", "section": "Task decomposition", "hint": "h"}],
                            "plan_text": f"plan version {i} " + GOOD_PLAN[:200] * i})
    return s


def test_simulate_blocks_forward_references():
    bad = "# Goal\nGoal: x.\n\n## Tasks\n1. A. Depends on task 2. Output: a.md.\n2. B. Output: b.md.\n"
    s = simulate(bad)
    assert s["executable_plan"] is False
    assert any("task 1 is blocked" in p for p in s["problems"])
    assert 2 in s["tasks_completed"]  # task 2 still runs in declared order


def test_verify_simulate_good_plan():
    v = verify(GOOD_PLAN)
    assert v["ok"], v["errors"]
    s = simulate(GOOD_PLAN)
    assert s["executable_plan"], s["problems"]


def test_fold_history_keeps_best_and_last_two(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.6.0", rounds=6)
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    s2 = fold_history(s)
    assert s2["history_folded"] is True
    texts = [r["plan_text"] for r in s2["rounds"]]
    assert texts[2].startswith("plan version 3")      # best round kept
    assert texts[4].startswith("plan version 5")      # last two kept
    assert texts[5].startswith("plan version 6")
    assert texts[0].startswith("[folded:")
    assert texts[1].startswith("[folded:")
    assert texts[3].startswith("[folded:")


def test_fold_history_never_touches_legacy(tmp_path):
    s = _mk_session(tmp_path, rounds=5)   # no engine_version marker
    s2 = fold_history(s)
    assert not s2.get("history_folded")
    assert all(r["plan_text"].startswith("plan version") for r in s2["rounds"])


def test_fold_history_idempotent(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.6.0", rounds=6)
    s1 = fold_history(s)
    s2 = fold_history(s1)
    assert s1["rounds"] == s2["rounds"]


def test_root_cause_grouping_in_assess(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.6.0", rounds=0)
    del s["best_version"]; s["best_score"] = None; s["rounds"] = []
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    # a plan missing several weighted sections -> >= 3 misses -> root_cause id
    r = plan_mode.assess(s, "# Goal\nGoal: x.\n## Tasks\n1. Do it. Output: a.md.", plans_dir=tmp_path)
    ids = [c["id"] for c in r["critiques"]]
    assert any(i.startswith("root_cause:") for i in ids), ids[:10]
    # non-mech misses are ordered by section score ascending (weakest first)
    non_mech = [c for c in r["critiques"] if not c["id"].startswith(("mech:", "judge:", "root_cause:"))]
    scores = s["rounds"][-1]["sections"]
    # sections in critique order should be non-decreasing in score
    mapped = []
    for c in non_mech:
        for k, v in scores.items():
            if v.get("label") == c["section"]:
                mapped.append(v["section_score"])
                break
    assert mapped == sorted(mapped), (mapped, sorted(mapped))


def test_legacy_session_loads(tmp_path):
    s = _mk_session(tmp_path, rounds=3)
    p = tmp_path / f"{s['session_id']}.json"
    p.write_text(json.dumps(s))
    loaded = _load_session(tmp_path, s["session_id"])
    assert loaded["rounds"][0]["score"] == s["rounds"][0]["score"]


def test_record_judge_saves_to_session_plans_dir(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.7.0", rounds=1)
    s["plans_dir"] = str(tmp_path)
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    entry = plan_mode.record_judge(s, {"verdict": "go", "ok": True,
                                       "feasibility_0_100": 80, "blockers": [],
                                       "contradictions": [], "missing": [],
                                       "falsifiable_criteria": True})
    saved = json.loads((tmp_path / f"{s['session_id']}.json").read_text())
    assert saved["judge_log"][-1]["verdict"] == "go"
    assert entry["verdict"] == "go"


def test_log_progress_updates_world_state(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.8.0", rounds=1)
    s["plans_dir"] = str(tmp_path)
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    plan_mode.log_progress(s, "task-1", status="done", evidence="output written", plans_dir=tmp_path)
    saved = json.loads((tmp_path / f"{s['session_id']}.json").read_text())
    assert "world_state" in saved
    assert saved["world_state"]["task-1"]["evidence"] == "output written"
    # a failed step arms the replan trigger (regression)
    plan_mode.log_progress(s, "task-2", status="failed", evidence="boom", plans_dir=tmp_path)
    saved = json.loads((tmp_path / f"{s['session_id']}.json").read_text())
    assert saved["replan_pending"] is True
    assert saved["world_state"]["task-2"]["status"] == "failed"


def test_judge_ensemble_excludes_unfalsifiable(tmp_path, monkeypatch):
    import asyncio

    async def fake_judge(plan_text, objective, **kw):
        return {"ok": True, "verdict": "rework", "feasibility_0_100": 10,
                "falsifiable_criteria": False}

    monkeypatch.setattr(plan_mode, "judge", fake_judge)
    s = _mk_session(tmp_path, engine_version="0.8.0", rounds=1)
    s["plans_dir"] = str(tmp_path)
    s["judge_log"] = [
        {"ok": True, "verdict": "rework", "feasibility_0_100": 10, "falsifiable_criteria": False},
        {"ok": True, "verdict": "go", "feasibility_0_100": 90, "falsifiable_criteria": True},
    ]
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    entry = asyncio.run(plan_mode.judge_ensemble(s, GOOD_PLAN, "test objective", n=3, plans_dir=tmp_path))
    # the unfalsifiable 10-verdict is excluded; median comes from the
    # mechanical baseline (100) and the falsifiable 90 -> lower median 90
    feas = [v.get("feasibility_0_100") for v in entry["votes"]]
    assert 10 not in feas
    assert entry["median_feasibility"] == 90
    assert entry["verdict"] == "go"


def test_judge_ensemble_median(tmp_path, monkeypatch):
    import asyncio

    async def fake_judge(plan_text, objective, **kw):
        # noisy, unfalsifiable vote: must be excluded by the ensemble
        return {"ok": True, "verdict": "rework", "feasibility_0_100": 10,
                "falsifiable_criteria": False}

    monkeypatch.setattr(plan_mode, "judge", fake_judge)
    s = _mk_session(tmp_path, engine_version="0.8.0", rounds=1)
    s["plans_dir"] = str(tmp_path)
    s["judge_log"] = [
        {"ok": True, "verdict": "go", "feasibility_0_100": 40, "falsifiable_criteria": True},
        {"ok": True, "verdict": "go", "feasibility_0_100": 90, "falsifiable_criteria": True},
    ]
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    # votes = mechanical baseline (100) + the two falsifiable priors [40, 90];
    # sorted [40, 90, 100], lower median = 90, so the recorded entry carries
    # median_feasibility 90 and ensemble=True.
    entry = asyncio.run(plan_mode.judge_ensemble(s, GOOD_PLAN, "test objective", n=3, plans_dir=tmp_path))
    assert entry.get("ensemble") is True
    assert entry["median_feasibility"] == 90
    assert entry["ok"] is True
    saved = json.loads((tmp_path / f"{s['session_id']}.json").read_text())
    assert saved["judge_log"][-1]["ensemble"] is True


def test_critique_aware_mutations():
    import plan_mode.search_engine as se
    crits = [{"id": "risks:Risk list", "section": "Risks", "hint": "h"}]
    ms = se._mutations("# Goal\nGoal: x.\n", 2, crits)
    assert ms[0]["note"].startswith("target-")
    assert "## Risks" in ms[0]["text"]
    # without critiques, generic mutations are used
    ms2 = se._mutations("# Goal\nGoal: x.\n", 2)
    assert all(not m["note"].startswith("target-") for m in ms2)


def test_seed_pool_loads_finished_sessions(tmp_path):
    import asyncio
    s = _mk_session(tmp_path, engine_version="0.9.0", rounds=1)
    s["rounds"][0]["plan_text"] = GOOD_PLAN.replace("Goal: ship a small patch.",
                                                    "Goal: ship a different patch.")
    s["best_version"] = 1; s["best_score"] = 90.0
    s["status"] = "finished"; s["completed_at"] = "2026-08-13T00:00:00+00:00"
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    s2 = _mk_session(tmp_path, engine_version="0.9.0", rounds=1)
    s2["session_id"] = "seed-target"
    s2["rounds"][0]["plan_text"] = GOOD_PLAN
    s2["best_version"] = 1; s2["best_score"] = 90.0
    (tmp_path / f"{s2['session_id']}.json").write_text(json.dumps(s2))
    res = asyncio.run(plan_mode.search(s2, iterations=2, width=2, mode="mcts",
                                       expansion="rules", plans_dir=tmp_path))
    tree = s2.get("search_tree", {})
    seed_nodes = [n for n in tree.get("nodes", {}).values() if n.get("origin") == "seed" or n.get("note") == "seed"]
    assert len(seed_nodes) >= 1
    assert res["best_score"] >= 0


def test_anchor_verdict_caps_go_on_broken_plan():
    import plan_mode.judge_client as jc
    parsed = {"verdict": "go", "feasibility_0_100": 95, "blockers": []}
    out = jc._anchor_verdict(parsed, {"verify_ok": False, "sim_ok": False})
    assert out["verdict"] == "rework"
    assert out["feasibility_0_100"] <= 60
    assert any("[grounding]" in b for b in out["blockers"])
    # a healthy plan keeps the model's verdict
    out2 = jc._anchor_verdict({"verdict": "go", "feasibility_0_100": 95, "blockers": []},
                              {"verify_ok": True, "sim_ok": True})
    assert out2["verdict"] == "go"
    assert out2["feasibility_0_100"] == 95


def test_ground_check_accepts_existing_input(tmp_path):
    real = tmp_path / "real_input.md"
    real.write_text("x")
    plan = (f"# Goal\nGoal: x.\n\n## Tasks\n1. Read. Requires: {real}. Output: a.md.\n"
            f"2. B. Depends on task 1. Output: b.md.\n")
    gc = plan_mode.ground_check(plan, cwd=tmp_path)
    assert gc["ok"] is True
    assert str(real) in gc["verified"]
    sim = plan_mode.simulate(plan, initial_state=set(gc["verified"]))
    assert sim["executable_plan"] is True


def test_ground_check_rejects_missing_input(tmp_path):
    plan = ("# Goal\nGoal: x.\n\n## Tasks\n1. Read. Requires: no_such_file_zz.md. "
            "Output: a.md.\n2. B. Depends on task 1. Output: b.md.\n")
    gc = plan_mode.ground_check(plan, cwd=tmp_path)
    assert gc["ok"] is False
    assert any("no_such_file_zz.md" in m for m in gc["missing"])


def test_ground_check_exempts_internal_handoffs(tmp_path):
    plan = ("# Goal\nGoal: x.\n\n## Tasks\n1. A. Output: mid.md.\n"
            "2. B. Depends on task 1. Requires: mid.md. Output: b.md.\n")
    gc = plan_mode.ground_check(plan, cwd=tmp_path)
    assert gc["ok"] is True  # mid.md is produced inside the plan, not the environment


def test_assess_emits_feasibility_critique(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.10.0", rounds=0)
    del s["best_version"]; s["best_score"] = None; s["rounds"] = []
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    plan = ("# Goal\nGoal: x.\n\n## Tasks\n1. Read. Requires: missing_input_zz.md. "
            "Output: a.md.\n2. B. Depends on task 1. Output: b.md.\n")
    r = plan_mode.assess(s, plan, plans_dir=tmp_path)
    ids = [c["id"] for c in r["critiques"]]
    assert any(i.startswith("mech:feasibility:") for i in ids), ids[:8]


def test_constraint_check_solver():
    p1 = "# Goal\nGoal: x.\n\n## Success criteria\n- S1: at most 2 tasks. Pass/fail: count.\n\n## Tasks\n1. A. Output: a.md.\n2. B. Depends on task 1. Output: b.md.\n3. C. Depends on task 2. Output: c.md.\n"
    cc = plan_mode.constraint_check(p1)
    assert cc["ok"] is False
    assert any("at most 2 tasks" in p for p in cc["problems"])
    p2 = "# Goal\nGoal: x.\n\n## Success criteria\n- S1: at most 5 tasks. Pass/fail: count.\n\n## Tasks\n1. A. Output: a.md.\n2. B. Depends on task 1. Output: b.md.\n3. C. Depends on task 2. Output: c.md.\n"
    assert plan_mode.constraint_check(p2)["ok"] is True


def test_verify_type_mismatch_diagnosis():
    plan = ("# Goal\nGoal: x.\n\n## Tasks\n1. A. Output: out.json.\n"
            "2. B. Depends on task 1. Requires: out.md. Output: b.md.\n")
    v = plan_mode.verify(plan)
    assert any("type mismatch" in e for e in v["errors"])


def test_verify_landmark_coverage():
    plan = ("# Goal\nGoal: x.\n\n## Success criteria\n- S1: ship it. Pass/fail: yes.\n- S2: also ship. Pass/fail: yes.\n\n"
            "## Tasks\n1. A. Output: a.md.\n2. B. Depends on task 1. Output: b.md.\n")
    v = plan_mode.verify(plan)
    assert any("landmark chain" in e for e in v["errors"])


def test_replan_tier_ladder(tmp_path):
    s = _mk_session(tmp_path, engine_version="0.10.0", rounds=1)
    s["replan_pending"] = True
    s["replan_task"] = "task-2"
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    r = plan_mode.assess(s, GOOD_PLAN, plans_dir=tmp_path)
    assert s["replan_tier"] == 2
    assert any("level 1 subgoal audit" in c["hint"] for c in r["critiques"])


def test_template_bank():
    t = plan_mode.template()
    assert len(t["function_inventory"]) >= 10
    names = [n for n, _ in t["function_inventory"]]
    for required in ["assess", "verify", "simulate", "ground_check",
                     "constraint_check", "release", "judge_ensemble"]:
        assert any(n.startswith(required + "(") for n in names)
    assert "## Tasks" in t["sample_plan"]
    assert len(t["section_templates"]) >= 10


def test_selfcheck_detects_broken_session(tmp_path):
    good = _mk_session(tmp_path, engine_version="0.12.0", rounds=1)
    good["rounds"][0]["plan_text"] = GOOD_PLAN
    good["best_version"] = 1; good["best_score"] = 90.0
    good["status"] = "finished"
    (tmp_path / f"{good['session_id']}.json").write_text(json.dumps(good))
    bad = _mk_session(tmp_path, engine_version="0.12.0", rounds=1)
    bad["session_id"] = "broken-finished"
    bad["rounds"][0]["plan_text"] = "# Goal\nGoal: x.\n\n## Tasks\n1. A. Depends on task 2. Output: a.md.\n2. B.\n"
    bad["best_version"] = 1; bad["best_score"] = 50.0
    bad["status"] = "finished"
    (tmp_path / f"{bad['session_id']}.json").write_text(json.dumps(bad))
    res = plan_mode.selfcheck(plans_dir=tmp_path, run_pytest=False)
    assert res["ok"] is False
    assert any("broken-finished" in p for p in res["problems"])
    # and the good session passes
    good_res = plan_mode.selfcheck(plans_dir=tmp_path, run_pytest=False)
    assert any(c["name"].startswith("session:") and c["ok"] for c in good_res["checks"])


def test_revertible_effects_journal_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(plan_mode, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    p = tmp_path / "victim.md"
    p.write_text("v1")
    plan_mode.edit_file(p, "v2", note="test-edit-1")
    plan_mode.edit_file(p, "v3", note="test-edit-2")
    assert p.read_text() == "v3"
    r = plan_mode.rollback(1)  # last effect recovers first
    assert p.read_text() == "v2"
    assert r["reverted"] == 1
    plan_mode.rollback(1)
    assert p.read_text() == "v1"  # both journaled edits undone, back to original
    # a further rollback is a no-op (journal exhausted), never an error
    r3 = plan_mode.rollback(1)
    assert r3["reverted"] == 0
    assert p.read_text() == "v1"


def test_deps_check_classifies():
    ds = plan_mode.deps_check()
    for k in ["llm_expansion", "api_judge", "pytest", "corpus", "pdftotext"]:
        assert k in ds["status"]
    assert set(ds["unsatisfied"]) <= set(ds["status"])


def test_recombine_crossover():
    import plan_mode.search_engine as se
    a = "# Goal\nGoal: a.\n\n## Risks\n- Risk: x; mitigation: y.\n\n## Tasks\n1. A. Output: a.md.\n"
    b = "# Goal\nGoal: b.\n\n## Replan\n- Trigger: z.\n\n## Tasks\n2. B. Output: b.md.\n"
    xo = se._recombine(a, b)
    if xo is None:
        return  # graceful degradation path
    assert xo != a and xo != b
    assert len(xo) > 0
    # identical parents must not produce a crossover (same-norm guard)
    assert se._recombine(a, a) is None


def test_search_rules_offline(tmp_path):
    import asyncio
    s = _mk_session(tmp_path, engine_version="0.6.0", rounds=1)
    s["rounds"][0]["plan_text"] = GOOD_PLAN
    s["best_version"] = 1; s["best_score"] = 90.0
    (tmp_path / f"{s['session_id']}.json").write_text(json.dumps(s))
    res = asyncio.run(plan_mode.search(s, iterations=3, width=2, mode="mcts",
                                       expansion="rules", plans_dir=tmp_path))
    assert res["best_score"] >= 0
    assert res["nodes"] >= 1
    assert "escalations" in res


def test_plan_dag_reads_and_inputs_parsing_no_none():
    """Verify that plan_dag parses Reads: and Inputs: cleanly without crashing or containing None."""
    plan = """
    1. Setup
       Inputs: raw_data.csv
       Reads: config.yaml
       Output: parsed.json
    2. Train
       Depends on 1
       Inputs: parsed.json
       Output: model.bin
    """
    dag = plan_mode.plan_dag(plan)
    assert dag["nodes"] == [1, 2]
    assert dag["inputs"][1] == ["config.yaml", "raw_data.csv"]
    assert None not in dag["inputs"][1]
    assert dag["artifacts"][1] == ["parsed.json"]


def test_verify_catches_non_contiguous_tasks():
    """Verify that non-contiguous task numbers (e.g. 1, 2, 4) trigger a structural error."""
    broken_plan = """
    1. Step One
       Output: a.txt
    2. Step Two
       Depends on 1
       Output: b.txt
    4. Step Four
       Depends on 2
       Output: c.txt
    """
    v = plan_mode.verify(broken_plan)
    assert not v["ok"]
    assert any("non-contiguous" in err for err in v["errors"])


def test_start_resumes_active_session_by_objective(tmp_path):
    """Calling start(objective) without session_id resumes the active session for that objective."""
    s1 = plan_mode.start("Automate end-to-end testing pipeline", plans_dir=tmp_path)
    s1_id = s1["session_id"]

    # Assess round 1
    plan_mode.assess(s1, "1. Task 1\nOutput: out.txt", plans_dir=tmp_path)

    # Call start with same objective
    s2 = plan_mode.start("Automate end-to-end testing pipeline", plans_dir=tmp_path)
    assert s2["session_id"] == s1_id
    assert len(s2["rounds"]) == 1


def test_release_respects_custom_plans_dir_and_binds_judge_version(tmp_path):
    """Verify release() reads plans_dir from session dict and binds judge verdict by round."""
    s = plan_mode.start("Custom session for release test", plans_dir=tmp_path)

    # Assess round 1
    plan_text = "1. Setup\nOutput: a.txt\n2. Run\nDepends on 1\nOutput: b.txt"
    plan_mode.assess(s, plan_text, plans_dir=tmp_path)
    s["status"] = "converged"
    s["best_score"] = 95.0

    # Record judge verdict bound to round 1
    plan_mode.record_judge(s, {"ok": True, "verdict": "go", "falsifiable_criteria": True, "feasibility_0_100": 95},
                           round_version=1, plans_dir=tmp_path)

    # Run release on session dict without passing plans_dir explicitly
    rel = plan_mode.release(s, min_score=90.0, require_judge=True)
    # Judge check should pass because round 1 judge verdict exists
    judge_check = next(c for c in rel["checks"] if c["name"] == "judge")
    assert judge_check["ok"] is True


@pytest.mark.asyncio
async def test_search_engine_respects_custom_plans_dir(tmp_path):
    """Verify search() with a live session dict respects custom plans_dir."""
    s = plan_mode.start("Search custom dir test", plans_dir=tmp_path)
    plan_mode.assess(s, "1. Init\nOutput: a.txt", plans_dir=tmp_path)

    # Run search with session dict
    result = await plan_mode.search(s, iterations=2, width=1, mode="beam", expansion="rules", plans_dir=tmp_path)
    assert result["nodes"] >= 1
    # Check that session file was updated in tmp_path
    assert (tmp_path / f"{s['session_id']}.json").exists()


def test_assess_candidates_prioritizes_executable_over_simulation_blocked(tmp_path):
    """Verify assess_candidates ranks an executable candidate above a simulation-blocked one."""
    s = plan_mode.start("Candidate ranking test", plans_dir=tmp_path)

    # Candidate 1: valid sequence
    cand_valid = """# Objective
    Goal: Build a working data pipeline.
    1. Extract
       Output: raw.csv
    2. Transform
       Depends on 1
       Inputs: raw.csv
       Output: clean.csv
    """

    # Candidate 2: broken forward reference (fails simulation)
    cand_blocked = """# Objective
    Goal: Build a working data pipeline.
    1. Transform
       Depends on 2
       Inputs: raw.csv
       Output: clean.csv
    2. Extract
       Output: raw.csv
    """

    winner = plan_mode.assess_candidates(s, [cand_blocked, cand_valid], plans_dir=tmp_path)
    # The valid candidate (index 1) should win
    assert winner["ranking"][0]["candidate"] == 1
    assert winner["ranking"][0]["sim_ok"] is True
    assert winner["ranking"][1]["sim_ok"] is False


def test_release_gate_fails_on_past_deadline(tmp_path):
    """Verify release() gate fails if the plan contains a past deadline date."""
    s = plan_mode.start("Past deadline test", plans_dir=tmp_path)
    plan_past = """
    1. Task 1
       Output: out.txt
       Deadline: by 2020-01-01
    """
    plan_mode.assess(s, plan_past, plans_dir=tmp_path)
    s["status"] = "converged"
    s["best_score"] = 95.0

    gate = plan_mode.release(s, min_score=90.0, require_judge=False, plans_dir=tmp_path)
    assert gate["ok"] is False
    mech_check = next(c for c in gate["checks"] if c["name"] == "mechanical")
    assert mech_check["ok"] is False
    assert any("in the past" in p for p in gate["problems"])


def test_release_gate_distinguishes_external_judge(tmp_path):
    """Verify require_external_judge=True rejects local baseline heuristic verdicts."""
    s = plan_mode.start("Judge mode test", plans_dir=tmp_path)
    plan_text = "1. Setup\nOutput: a.txt"
    plan_mode.assess(s, plan_text, plans_dir=tmp_path)
    s["status"] = "converged"
    s["best_score"] = 95.0

    # Record local baseline verdict (external = False)
    plan_mode.record_judge(s, {
        "ok": True, "verdict": "go", "falsifiable_criteria": True,
        "source": "mechanical_baseline", "external": False
    }, round_version=1, plans_dir=tmp_path)

    # Default release accepts it
    gate_default = plan_mode.release(s, min_score=90.0, require_judge=True, require_external_judge=False, plans_dir=tmp_path)
    judge_check = next(c for c in gate_default["checks"] if c["name"] == "judge")
    assert judge_check["ok"] is True

    # Strict external release rejects it
    gate_strict = plan_mode.release(s, min_score=90.0, require_judge=True, require_external_judge=True, plans_dir=tmp_path)
    judge_check_strict = next(c for c in gate_strict["checks"] if c["name"] == "judge")
    assert judge_check_strict["ok"] is False


def test_search_expand_and_select_respect_custom_plans_dir(tmp_path):
    """Verify search_expand() and search_select() work with live session dict with custom plans_dir."""
    s = plan_mode.start("Search expand select test", plans_dir=tmp_path)
    plan_mode.assess(s, "1. Init\nOutput: a.txt", plans_dir=tmp_path)

    # Expand with custom plans_dir session dict
    exp = plan_mode.search_expand(s, ["1. Task 1\nOutput: out1.txt", "1. Task 2\nOutput: out2.txt"], plans_dir=tmp_path)
    assert len(exp["node_ids"]) == 2
    assert exp["best_score"] > 0

    # Select with custom plans_dir session dict
    sel = plan_mode.search_select(s, plans_dir=tmp_path)
    assert sel["node_id"] is not None
    assert (tmp_path / f"{s['session_id']}.json").exists()


def test_strict_external_judge_requires_explicit_external_llm(tmp_path):
    """Verify require_external_judge=True strictly requires source == 'external_llm' and external == True."""
    s = plan_mode.start("Strict external judge test", plans_dir=tmp_path)
    plan_mode.assess(s, "1. Setup\nOutput: a.txt", plans_dir=tmp_path)
    s["status"] = "converged"
    s["best_score"] = 95.0

    # Untagged or non-external verdict
    plan_mode.record_judge(s, {"ok": True, "verdict": "go", "falsifiable_criteria": True}, round_version=1, plans_dir=tmp_path)
    gate1 = plan_mode.release(s, min_score=90.0, require_judge=True, require_external_judge=True, plans_dir=tmp_path)
    assert gate1["ok"] is False

    # Tagged genuine external LLM verdict
    plan_mode.record_judge(s, {
        "ok": True, "verdict": "go", "falsifiable_criteria": True,
        "source": "external_llm", "external": True
    }, round_version=1, plans_dir=tmp_path)
    gate2 = plan_mode.release(s, min_score=90.0, require_judge=True, require_external_judge=True, plans_dir=tmp_path)
    assert gate2["ok"] is True


@pytest.mark.asyncio
async def test_search_ast_mode_evolution(tmp_path):
    """Verify plan.search(mode='ast') evolves candidates using ASTSearchEngine."""
    s = plan_mode.start("AST Search Mode Test", plans_dir=tmp_path)
    plan_mode.assess(s, "1. Init\nOutput: a.txt\n2. Process\nDepends on 1\nOutput: b.txt", plans_dir=tmp_path)

    res = await plan_mode.search(s, iterations=2, width=2, mode="ast", plans_dir=tmp_path)
    assert res["nodes"] >= 1
    assert res["best_score"] > 0


def test_context_budgeter_obeys_max_tokens():
    """Verify ContextBudgeter compresses history when token limit is exceeded."""
    session = {
        "best_version": 1,
        "rounds": [
            {"version": 1, "ts": "2026-08-01", "score": 90.0, "plan_text": "A" * 2000, "critiques": []},
            {"version": 2, "ts": "2026-08-02", "score": 70.0, "plan_text": "B" * 2000, "critiques": ["err"]},
            {"version": 3, "ts": "2026-08-03", "score": 75.0, "plan_text": "C" * 2000, "critiques": ["err2"]},
            {"version": 4, "ts": "2026-08-04", "score": 80.0, "plan_text": "D" * 2000, "critiques": []}
        ]
    }
    compressed = ContextBudgeter.compress_history(session, max_context_tokens=1500)
    # Rounds 2 and 3 should be compressed to meet the budget
    assert compressed["rounds"][1].get("folded") is True
    assert compressed["rounds"][2].get("folded") is True


@pytest.mark.asyncio
async def test_search_commits_winning_plan_to_session(tmp_path):
    """Verify search() automatically commits the best discovered plan to session rounds."""
    s = plan_mode.start("Search commit test", plans_dir=tmp_path)
    initial_plan = "1. Setup\nOutput: a.txt"
    plan_mode.assess(s, initial_plan, plans_dir=tmp_path)
    initial_score = s["best_score"]

    # Run search which finds higher-scoring variants
    res = await plan_mode.search(s, iterations=2, width=2, mode="beam", expansion="rules", plans_dir=tmp_path)

    # Session best score should be updated to search winner
    s_reloaded = plan_mode._load_session(tmp_path, s["session_id"])
    assert s_reloaded["best_score"] >= initial_score
    assert len(s_reloaded["rounds"]) >= 2
    assert "search:" in s_reloaded["rounds"][-1].get("note", "")


def test_atomic_session_save_under_concurrent_writes(tmp_path):
    """Verify concurrent _save_session operations do not corrupt JSON file."""
    import concurrent.futures
    s = plan_mode.start("Concurrent atomic test", plans_dir=tmp_path)

    def writer(idx):
        session_copy = plan_mode._load_session(tmp_path, s["session_id"])
        session_copy[f"key_{idx}"] = f"value_{idx}"
        plan_mode._save_session(tmp_path, session_copy)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(writer, i) for i in range(20)]
        results = [f.result() for f in futures]

    assert all(results)
    final_session = plan_mode._load_session(tmp_path, s["session_id"])
    assert final_session["session_id"] == s["session_id"]


def test_assess_structural_rot_rule_violation_with_reworded_syntax(tmp_path):
    """Verify assess() catches structural RoT violations even when task phrasing is reworded."""
    s = plan_mode.start("RoT structural test", plans_dir=tmp_path)

    # Plan 1: declares input that does not exist -> triggers flaw & distills RoT rule
    plan1 = """
    1. Extract
       Inputs: missing_zz.md
       Output: a.txt
    """
    res1 = plan_mode.assess(s, plan1, plans_dir=tmp_path)

    # Plan 2: reworded with "Needs:" instead of "Inputs:"
    plan2 = """
    1. Extract
       Needs: missing_zz.md
       Output: a.txt
    """
    res2 = plan_mode.assess(s, plan2, plans_dir=tmp_path)
    # Must catch the structural RoT violation
    assert any("rot:" in c["id"] for c in res2["critiques"])


def test_session_lock_read_modify_write_protection(tmp_path):
    """Verify session_lock protects read-modify-write cycles from lost updates."""
    from plan_mode import session_lock
    import concurrent.futures
    s = plan_mode.start("Session lock test", plans_dir=tmp_path)

    def worker(worker_id):
        with session_lock(tmp_path, s["session_id"]):
            data = plan_mode._load_session(tmp_path, s["session_id"])
            lst = data.setdefault("worker_log", [])
            lst.append(worker_id)
            plan_mode._save_session(tmp_path, data)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(16)]
        results = [f.result() for f in futures]

    assert all(results)
    final_session = plan_mode._load_session(tmp_path, s["session_id"])
    assert len(final_session["worker_log"]) == 16


def test_session_lock_timeout_raises_error(tmp_path):
    """Verify session_lock raises TimeoutError when lock is held beyond timeout."""
    from plan_mode import session_lock
    import fcntl
    lock_file = tmp_path / ".test_session.lock"

    # Hold lock
    f = open(lock_file, "a", encoding="utf-8")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    try:
        with pytest.raises(TimeoutError, match="Timed out after"):
            with session_lock(tmp_path, "test_session", timeout=0.1):
                pass
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


@pytest.mark.asyncio
async def test_search_ast_mode_with_custom_cwd(tmp_path):
    """Verify search(mode='ast', cwd=tmp_path) correctly grounds relative disk inputs."""
    data_file = tmp_path / "raw_input.csv"
    data_file.write_text("col1,col2\n1,2")

    s = plan_mode.start("AST Search CWD Test", plans_dir=tmp_path / "plans")
    plan_text = """
    1. Extract
       Inputs: raw_input.csv
       Output: extracted.json
    2. Transform
       Depends on 1
       Inputs: extracted.json
       Output: transformed.json
    """
    plan_mode.assess(s, plan_text, plans_dir=tmp_path / "plans")
    res = await plan_mode.search(s, iterations=2, width=2, mode="ast", cwd=tmp_path, plans_dir=tmp_path / "plans")
    assert res["nodes"] >= 1
    assert res["best_score"] > 0


def test_assess_convergence_reentrant_lock(tmp_path):
    """Verify assess() does not deadlock when max_rounds=1 triggers fold_history() under session_lock."""
    plans_dir = tmp_path / "plans"
    s = plan_mode.start("Reentrant Assess Test", plans_dir=plans_dir, max_rounds=1)
    plan_text = """# Plan
1. Task Alpha
   Output: alpha.json
2. Task Beta
   Depends on 1
   Inputs: alpha.json
   Output: beta.json
"""
    res = plan_mode.assess(s, plan_text, plans_dir=plans_dir)
    assert res["status"] == "converged"
    assert res["continue"] is False


def test_finish_require_release_reentrant_lock(tmp_path):
    """Verify finish() does not deadlock when require_release=True triggers release() under session_lock."""
    plans_dir = tmp_path / "plans"
    s = plan_mode.start("Reentrant Finish Test", plans_dir=plans_dir, max_rounds=1)
    plan_text = """# Plan
1. Task Alpha
   Output: alpha.json
2. Task Beta
   Depends on 1
   Inputs: alpha.json
   Output: beta.json
"""
    # assess with max_rounds=1 converges immediately
    plan_mode.assess(s, plan_text, plans_dir=plans_dir)
    # Finish with require_release=True and require_judge=False
    res = plan_mode.finish(s, plans_dir=plans_dir, require_release=True, require_judge=False, min_score=0.0)
    assert res["status"] == "finished"
    assert "release_gate" in res
