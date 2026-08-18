"""Tests for AgentRewind-style checkpoints and ACID-Agent-style commits."""
from __future__ import annotations

import asyncio

import pytest

import plan_mode
from plan_mode.memory_distiller import ReplanningLadder, RoTRuleBase


def _valid_plan(goal="x"):
    return (
        f"# Goal\nGoal: {goal}. In scope: x. Out of scope: y.\n\n"
        "## Success criteria\n- S1: 1 test passes. Pass/fail. Deadline: within 1 day.\n\n"
        "## Tasks\n1. A. Output: a.md.\n2. B. Depends on 1. Inputs: a.md. Output: b.md (verifies S1).\n"
    )


def test_commit_only_happens_on_successful_release(tmp_path):
    s = plan_mode.start("commit semantics", plans_dir=tmp_path, max_rounds=1)
    plan_mode.assess(s, _valid_plan(), plans_dir=tmp_path)
    assert s.get("committed_version") is None
    # Release with judge required should fail before a verdict exists
    gate = plan_mode.release(s, min_score=0, require_judge=True, plans_dir=tmp_path)
    assert gate["ok"] is False
    assert s.get("committed_version") is None

    plan_mode.record_judge(s, {
        "ok": True, "verdict": "go", "falsifiable_criteria": True,
        "source": "external_llm", "external": True,
    }, plans_dir=tmp_path)
    gate = plan_mode.release(s, min_score=0, require_judge=False, plans_dir=tmp_path)
    assert gate["ok"] is True
    assert s["committed_version"] == s["best_version"]
    committed = plan_mode.committed(s)
    assert committed["plan_text"] == s["rounds"][s["best_version"] - 1]["plan_text"]


def test_checkpoint_and_rewind_restore_rounds(tmp_path):
    s = plan_mode.start("checkpoint rewind", plans_dir=tmp_path)
    plan_mode.assess(s, _valid_plan("first"), plans_dir=tmp_path)
    cp = plan_mode.checkpoint(s, note="before second plan")
    plan_mode.assess(s, _valid_plan("second"), plans_dir=tmp_path)
    assert len(s["rounds"]) == 2
    restored = plan_mode.rewind(s, cp["checkpoint_id"])
    assert restored["session_id"] == s["session_id"]
    assert len(restored["rounds"]) == 1
    assert restored["rounds"][0]["plan_text"].startswith("# Goal")
    assert restored["rewind_log"][-1]["checkpoint_id"] == cp["checkpoint_id"]


@pytest.mark.asyncio
async def test_search_can_take_pre_expansion_checkpoint(tmp_path):
    s = plan_mode.start("search checkpoint", plans_dir=tmp_path)
    plan_mode.assess(s, _valid_plan(), plans_dir=tmp_path)
    before = len(s["rounds"])
    await plan_mode.search(s, iterations=1, width=1, mode="beam", expansion="rules",
                           checkpoint_before=True, plans_dir=tmp_path)
    assert len(s.get("checkpoints", [])) == 1
    assert len(s["rounds"]) >= before  # rounds preserved and search completes cleanly


def test_rot_experience_tree_perspective_and_outcomes(tmp_path):
    rb = RoTRuleBase(storage_path=tmp_path / "rot.json")
    flaws = [{
        "type": "unsatisfied_precondition",
        "detail": "Task 2 requires precondition 'exists(model.bin)' which is unsatisfied",
        "remedy": "Add a producer for model.bin",
    }]
    rb.distill_from_flaws(flaws, context_tag="training", perspective="ml-pipeline")
    rule_id = next(iter(rb.rules))
    assert rb.rules[rule_id].perspective == "ml-pipeline"
    rb.record_outcome(rule_id, success=True)
    rb.record_outcome(rule_id, success=False)
    report = rb.tree_report()
    assert report["perspectives"] == 1
    assert abs(rb.rules[rule_id].confidence - (1 / 3)) < 0.001


def test_replanning_ladder_has_drift_tier():
    tier = ReplanningLadder.determine_replan_tier(
        failed_task_id=3, error_message="silent failure: state drift detected",
        total_tasks=5, retry_count=0,
    )
    assert tier["tier"] == 4
    assert tier["scope"] == "runtime_drift_recovery"


def test_rewind_invalid_checkpoint_raises_error(tmp_path):
    """Verify rewind raises ValueError when an unknown checkpoint ID is supplied."""
    s = plan_mode.start("invalid cp", plans_dir=tmp_path)
    plan_mode.assess(s, _valid_plan("first"), plans_dir=tmp_path)
    plan_mode.checkpoint(s, note="good cp")
    with pytest.raises(ValueError, match="checkpoint 'non-existent-cp-id' not found"):
        plan_mode.rewind(s, checkpoint_id="non-existent-cp-id")


def test_committed_and_best_out_of_bounds(tmp_path):
    """Verify best() and committed() return error dictionaries on invalid versions rather than crashing."""
    s = plan_mode.start("bounds check", plans_dir=tmp_path)
    assert "error" in plan_mode.best(s)
    assert "error" in plan_mode.committed(s)

    plan_mode.assess(s, _valid_plan("first"), plans_dir=tmp_path)
    s["best_version"] = 99
    assert "out of bounds" in plan_mode.best(s).get("error", "")

    s["best_version"] = 0
    assert "out of bounds" in plan_mode.best(s).get("error", "")

    s["best_version"] = 1
    s["committed_version"] = 99
    assert "out of bounds" in plan_mode.committed(s).get("error", "")


def test_replanning_ladder_tier2_on_last_task_and_key_consistency():
    """Verify Tier 2 replanning works on final task and consistently exports task_id."""
    tier1 = ReplanningLadder.determine_replan_tier(failed_task_id=2, error_message="timeout", total_tasks=5, retry_count=0)
    assert tier1["task_id"] == 2

    tier2 = ReplanningLadder.determine_replan_tier(failed_task_id=5, error_message="compilation error", total_tasks=5, retry_count=1)
    assert tier2["tier"] == 2
    assert tier2["task_id"] == 5
    assert tier2["failed_task_id"] == 5

    tier4 = ReplanningLadder.determine_replan_tier(failed_task_id=3, error_message="schema violation divergence", total_tasks=5, retry_count=0)
    assert tier4["task_id"] == 3


def test_version_tuple_zero_padding():
    """Verify _version_tuple zero-pads two-part and single-part versions so Python tuple comparison works."""
    from plan_mode import _version_tuple
    assert _version_tuple("0.14") == (0, 14, 0)
    assert _version_tuple("0.14.0") == (0, 14, 0)
    assert _version_tuple("0.14") >= (0, 14, 0)
    assert _version_tuple("0.13.9") < (0, 14, 0)
    assert _version_tuple("1") == (1, 0, 0)
    assert _version_tuple(None) == (0, 0, 0)
