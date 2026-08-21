"""PR5 second-pass RED regressions discovered by the full production audit."""
from __future__ import annotations

import asyncio
import time

import pytest

import plan_mode
from plan_mode.execution_trace import verify_execution_trace
from plan_mode.ir import ActionIR, PlanIR, Provenance, SourceType
from plan_mode.judges import OpenAIJudge
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.session import PlanningSession, StateDriftError


def _prov() -> Provenance:
    return Provenance(source_type=SourceType.USER_REQUIREMENT, source_id="pr5-second-pass")


def _validated_session_with_capability() -> tuple[PlanningSession, CapabilityRegistry]:
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="cap", description="test capability"))
    plan = PlanIR(
        plan_id="auth-second-pass",
        goal_description="validated plan must remain validated",
        actions=[ActionIR(action_id="a1", capability_name="cap", provenance=_prov())],
    )
    session = PlanningSession(session_id="auth-second-pass")
    session.submit_draft(plan)
    result = session.validate_candidate(1, registry, observed_world_state=[])
    assert result.status.value == "PASS"
    session.select_version(1)
    return session, registry


def test_authorization_rejects_plan_mutated_after_validation():
    session, registry = _validated_session_with_capability()
    session.versions[1].plan_ir.goal_description = "tampered after validation"
    with pytest.raises(StateDriftError):
        session.authorize_selected(registry, policy_hash="policy")


def test_authorization_revalidates_registry_changed_after_validation():
    session, registry = _validated_session_with_capability()
    registry.capabilities.clear()
    with pytest.raises(StateDriftError):
        session.authorize_selected(registry, policy_hash="policy")


def test_release_rejects_same_version_judge_for_mutated_plan(tmp_path, monkeypatch):
    session = plan_mode.start("same-version stale judge", plans_dir=tmp_path, max_rounds=1)
    session.update({
        "status": "converged",
        "best_version": 1,
        "best_score": 100.0,
        "rounds": [{
            "version": 1,
            "ts": "test",
            "score": 100.0,
            "delta": None,
            "critiques": [],
            "sections": {},
            "note": None,
            "plan_text": "# Goal\nGoal: original.\n## Tasks\n1. A. Output: a.txt.\n",
        }],
    })
    plan_mode._save_session(tmp_path, session)
    plan_mode.record_judge(
        session,
        {"ok": True, "verdict": "go", "falsifiable_criteria": True, "feasibility_0_100": 100},
        round_version=1,
        plans_dir=tmp_path,
    )

    # Mutate the same version in place. A round number alone can no longer bind
    # the old verdict to the new plan text.
    session["rounds"][0]["plan_text"] = "# Goal\nGoal: changed.\n## Tasks\n1. B. Output: b.txt.\n"

    monkeypatch.setattr(plan_mode, "_mechanical_checks", lambda text: [])
    monkeypatch.setattr(plan_mode, "verify", lambda *a, **k: {"ok": True, "errors": []})
    monkeypatch.setattr(plan_mode, "ground_check", lambda *a, **k: {"ok": True, "missing": [], "verified": []})
    monkeypatch.setattr(plan_mode, "simulate", lambda *a, **k: {"executable_plan": True, "problems": []})
    monkeypatch.setattr(
        plan_mode,
        "validate_execution_contract",
        lambda *a, **k: {"ok": True, "errors": [], "contract": None},
    )

    gate = plan_mode.release(
        session,
        min_score=0,
        require_judge=True,
        plans_dir=tmp_path,
    )
    judge_check = next(check for check in gate["checks"] if check["name"] == "judge")
    assert judge_check["ok"] is False
    assert gate["ok"] is False


def test_malformed_exit_criterion_task_id_returns_error_not_exception():
    plan_text = """# Goal
Goal: trace parser must fail closed.

## Execution Contract
```json
{
  "verification_commands": [["true"]],
  "exit_criteria": [{"task": "not-an-int", "command": ["true"]}]
}
```
"""
    evidence = {
        "agent_id": "executor",
        "verifier_agent_id": "verifier",
        "tasks": [],
    }
    result = verify_execution_trace(plan_text, evidence)
    assert result["ok"] is False
    assert any("task" in error.lower() and "invalid" in error.lower() for error in result["errors"])


class _SlowHTTPClient:
    async def post(self, *args, **kwargs):
        await asyncio.sleep(1.0)
        raise AssertionError("request should have been cancelled by the total timeout")


@pytest.mark.asyncio
async def test_provider_judge_enforces_total_wall_clock_timeout():
    judge = OpenAIJudge(api_key="test", http_client=_SlowHTTPClient())
    plan = PlanIR(plan_id="judge-timeout", goal_description="bounded judge")
    started = time.monotonic()
    verdict = await judge.evaluate(plan, timeout=0.02)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5
    assert verdict.verdict == "UNKNOWN"
    assert any("timeout" in blocker.lower() for blocker in verdict.blockers)


@pytest.mark.asyncio
async def test_execute_plan_does_not_claim_recovery_when_inverse_fails():
    plan_text = """# Plan
1. First
   Output: first.txt
2. Second
   Depends on 1
   Output: second.txt
"""

    async def first(ctx):
        async def bad_inverse():
            raise RuntimeError("undo failed")
        await ctx.async_effect(lambda: bad_inverse)
        return {"ok": True}

    async def second(ctx):
        raise RuntimeError("forward failed")

    result = await plan_mode.execute_plan(plan_text, task_handlers={1: first, 2: second})
    assert result["ok"] is False
    assert result["recovered"] is False
    assert "undo failed" in result.get("recovery_error", "")
