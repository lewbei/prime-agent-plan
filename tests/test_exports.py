"""Test public interface consistency and SKILL.md documented symbol exports between plan and plan_mode."""
import inspect
import sys
from pathlib import Path
import pytest

import plan
import plan_mode


def test_plan_and_plan_mode_all_identical():
    """Verify plan.__all__ and plan_mode.__all__ contain identical symbols."""
    assert set(plan.__all__) == set(plan_mode.__all__)
    assert len(plan.__all__) == len(plan_mode.__all__)


def test_plan_and_plan_mode_attributes_match():
    """Verify every symbol in __all__ exists on both modules and resolves to identical callables/objects."""
    for sym in plan.__all__:
        assert hasattr(plan, sym), f"Symbol '{sym}' missing from plan"
        assert hasattr(plan_mode, sym), f"Symbol '{sym}' missing from plan_mode"
        obj_plan = getattr(plan, sym)
        obj_plan_mode = getattr(plan_mode, sym)
        assert obj_plan is obj_plan_mode, f"Symbol '{sym}' in plan does not match plan_mode ({obj_plan} vs {obj_plan_mode})"


def test_all_skill_documented_symbols_exported():
    """Verify all symbols referenced in SKILL.md API table exist on both plan and plan_mode."""
    skill_path = Path(__file__).resolve().parent.parent / "SKILL.md"
    assert skill_path.exists()

    expected_symbols = [
        "start", "assess", "assess_candidates", "search", "verify", "ground_check",
        "simulate", "record_judge", "committed", "checkpoint", "rewind", "release",
        "finish", "log_progress", "fold_history", "validate_execution_contract",
        "probe_contract", "symbol_audit", "execute_plan", "speculative_rollout",
        "create_subagent_context", "provide_tool", "selfcheck"
    ]

    for sym in expected_symbols:
        assert hasattr(plan, sym), f"Documented symbol '{sym}' missing from plan"
        assert hasattr(plan_mode, sym), f"Documented symbol '{sym}' missing from plan_mode"
        assert sym in plan.__all__, f"Documented symbol '{sym}' missing from plan.__all__"
        assert sym in plan_mode.__all__, f"Documented symbol '{sym}' missing from plan_mode.__all__"


def test_plan_entrypoint_calls(tmp_path):
    """Verify invoking basic entrypoints via plan namespace."""
    s = plan.start("Test objective for export verification", plans_dir=tmp_path)
    assert s["status"] == "drafting"
    assert len(s["rounds"]) == 0

    plan_text = """# Objective: Test export
We will do the testing. Out of scope: non-goals.

## Success Criteria
- S1: 100% test pass by 2026-12-31 (falsifiable via pytest)

## Assumptions
- Assumptions: pytest is installed
- Unknowns: None

## Tasks
1. Run Unit Tests (covers S1)
   Inputs: tests/test_exports.py
   Output: test_results.log
2. Review Results
   Depends on 1
   Inputs: test_results.log
   Output: review.md
3. Finalize
   Depends on 2
   Inputs: review.md
   Output: done.txt

## Milestones
- Milestone 1: Tests run (go/no-go)

## Risks
- Risk: Missing deps. Mitigation: install before running. Rollback: git reset.

## Resources
- Time: 5 minutes. Budget: 0 tokens.

## Alternatives
- Option A vs Option B: Option A chosen.

## Verification
- Test invariants verified via script.
"""
    v = plan.verify(plan_text)
    assert "ok" in v
    assert "causal_validation" in v

    # Test ground_check via plan
    gc = plan.ground_check(plan_text)
    assert isinstance(gc, dict)

    # Test assess via plan
    ass = plan.assess(s, plan_text, note="First draft")
    assert ass["version"] == 1
    assert "score" in ass

    # Test fold_history via plan
    folded = plan.fold_history(s)
    assert folded["session_id"] == s["session_id"]

    # Verify selfcheck runs from plan
    sc = plan.selfcheck(plans_dir=tmp_path, run_pytest=False)
    assert sc["ok"] is True
