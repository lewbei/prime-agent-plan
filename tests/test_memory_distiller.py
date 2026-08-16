"""Tests for RoT Rule Distillation and Context Budgeter."""
from __future__ import annotations

import pytest
from plan_mode.memory_distiller import ContextBudgeter, ReplanningLadder, RoTRuleBase


def test_rot_rule_distillation_and_violation_check(tmp_path):
    """Verify distilling reusable rules from flaws and checking candidate violations."""
    rule_base = RoTRuleBase(storage_path=tmp_path / "rules.json")

    flaws = [
        {
            "type": "clobber_threat",
            "detail": "Task 2 deletes db_online",
            "remedy": "Reorder Task 2 after consumer"
        }
    ]

    distilled = rule_base.distill_from_flaws(flaws, context_tag="database")
    assert len(distilled) == 1
    assert "rot:clobber_threat:" in distilled[0].rule_id

    # Check violation on a bad draft
    bad_plan = "1. Setup\n2. Task 2 deletes db_online\n3. Query"
    violations = rule_base.check_plan_violations(bad_plan)
    assert len(violations) == 1
    assert "clobber_threat" in violations[0]["flaw_type"]


def test_context_budgeter_compression():
    """Verify that superseded rounds are folded while preserving best round."""
    session = {
        "best_version": 2,
        "rounds": [
            {"version": 1, "ts": "2026-08-01", "score": 60.0, "plan_text": "Round 1 text", "critiques": ["a"]},
            {"version": 2, "ts": "2026-08-02", "score": 95.0, "plan_text": "Round 2 Best Text", "critiques": []},
            {"version": 3, "ts": "2026-08-03", "score": 80.0, "plan_text": "Round 3 text", "critiques": ["b"]},
            {"version": 4, "ts": "2026-08-04", "score": 85.0, "plan_text": "Round 4 text", "critiques": ["c"]}
        ]
    }

    compressed = ContextBudgeter.compress_history(session)
    rounds = compressed["rounds"]
    # Round 1 should be folded
    assert rounds[0].get("folded") is True
    # Round 2 (best) must remain full
    assert rounds[1].get("folded") is not True
    assert rounds[1]["plan_text"] == "Round 2 Best Text"
    # Round 4 (latest) must remain full
    assert rounds[3].get("folded") is not True


def test_replanning_ladder_tiers():
    """Verify 3-tier replan escalation."""
    tier1 = ReplanningLadder.determine_replan_tier(failed_task_id=2, error_message="HTTP Timeout 504", total_tasks=5, retry_count=0)
    assert tier1["tier"] == 1
    assert tier1["scope"] == "local_task"

    tier2 = ReplanningLadder.determine_replan_tier(failed_task_id=3, error_message="Corrupted artifact", total_tasks=5, retry_count=1)
    assert tier2["tier"] == 2
    assert tier2["scope"] == "subgraph_replan"

    tier3 = ReplanningLadder.determine_replan_tier(failed_task_id=5, error_message="Fundamental architectural limitation", total_tasks=5, retry_count=3)
    assert tier3["tier"] == 3
    assert tier3["scope"] == "global_strategy_replan"
