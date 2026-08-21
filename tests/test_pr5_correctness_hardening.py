"""PR #5 RED tests: adversarial correctness and truth-binding invariants.

These tests intentionally exercise invariants that are not covered by the
pre-PR5 suite.  They should fail on the pre-fix baseline and pass only after
the corresponding production path is hardened.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

import plan_mode
from plan_mode.execution_contract import validate_execution_contract
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    HardConstraint,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WorldFact,
)
from plan_mode.isolation import IsolationManager, OperationIsolation
from plan_mode.probing import DiagnosticProbe
from plan_mode.recovery import RecoveryStatus, SagaRecoveryManager
from plan_mode.recovery_graph import DriftEvidence, recovery_decision
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, CompensationAction, SchemaMismatchError
from plan_mode.runtime.executor import StepExecutionResult, WitnessStatus
from plan_mode.runtime.ledger import EvidenceLedger
from plan_mode.runtime.sandbox import ExecutionSandbox, SecurityProfile
from plan_mode.session import PlanningSession, SessionState, StateDriftError


def _prov(source: SourceType = SourceType.USER_REQUIREMENT) -> Provenance:
    return Provenance(source_type=source, source_id="pr5-test")


def _legacy_release_session(tmp_path):
    session = plan_mode.start("pr5 release atomicity", plans_dir=tmp_path, max_rounds=1)
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
            "plan_text": "# Goal\nGoal: x.\n\n## Tasks\n1. Do. Inputs: input.txt. Output: out.txt.\n",
        }],
    })
    plan_mode._save_session(tmp_path, session)
    return session


def _patch_release_non_cwd_gates(monkeypatch):
    monkeypatch.setattr(plan_mode, "_mechanical_checks", lambda text: [])
    monkeypatch.setattr(plan_mode, "verify", lambda *a, **k: {"ok": True, "errors": []})
    monkeypatch.setattr(
        plan_mode,
        "simulate",
        lambda *a, **k: {"executable_plan": True, "problems": []},
    )
    monkeypatch.setattr(
        plan_mode,
        "validate_execution_contract",
        lambda *a, **k: {"ok": True, "errors": [], "contract": None},
    )
    monkeypatch.setattr(
        plan_mode,
        "verify_execution_trace",
        lambda *a, **k: {"ok": True, "errors": [], "warnings": []},
    )


def test_failed_execution_cwd_release_is_atomic_in_memory_and_disk(tmp_path, monkeypatch):
    """A release returning False must never have committed first."""
    session = _legacy_release_session(tmp_path)
    _patch_release_non_cwd_gates(monkeypatch)

    def fake_ground(text, *, cwd=None):
        if cwd is None:
            return {"ok": True, "missing": [], "verified": []}
        return {"ok": False, "missing": ["input.txt (task 1)"], "verified": []}

    monkeypatch.setattr(plan_mode, "ground_check", fake_ground)

    gate = plan_mode.release(
        session,
        min_score=0,
        require_judge=False,
        execution_cwd=tmp_path,
        plans_dir=tmp_path,
    )
    assert gate["ok"] is False
    assert session.get("committed_version") is None
    assert session.get("committed_score") is None
    assert session.get("committed_plan_hash") in (None, "")

    persisted = plan_mode._load_session(tmp_path, session["session_id"])
    assert persisted.get("committed_version") is None
    assert persisted.get("committed_score") is None
    assert persisted.get("committed_plan_hash") in (None, "")


def test_direct_plan_mode_mutations_are_stable_without_importing_plan(tmp_path):
    """The authoritative implementation must be deterministic without import side effects."""
    code = (
        "import json, plan_mode.search_engine as se; "
        "p='# Goal\\nGoal: stable.\\n## Tasks\\n1. A. Output: a.txt.'; "
        "print(json.dumps([x['note'] for x in se._mutations(p, 4)]))"
    )
    outputs = []
    for seed in ("1", "987654"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout.strip()))
    assert outputs[0] == outputs[1]


def test_pipeline_uses_pipefail_semantics(tmp_path):
    sandbox = ExecutionSandbox(
        policy=SecurityProfile.get_profile("PERMISSIVE_DEV").model_copy(
            update={"workspace_dir": str(tmp_path)}
        )
    )
    result = sandbox.execute_argv_pipeline(
        [
            [sys.executable, "-c", "import sys; print('payload'); sys.exit(7)"],
            [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(0)"],
        ],
        cwd=str(tmp_path),
        timeout_seconds=2.0,
    )
    assert result.returncode != 0
    assert result.timeout_exceeded is False


def test_pipeline_drains_upstream_stderr_without_timeout(tmp_path):
    sandbox = ExecutionSandbox(
        policy=SecurityProfile.get_profile("PERMISSIVE_DEV").model_copy(
            update={"workspace_dir": str(tmp_path)}
        )
    )
    result = sandbox.execute_argv_pipeline(
        [
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('e'*262144); sys.stderr.flush(); sys.stdout.write('ok')",
            ],
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        ],
        cwd=str(tmp_path),
        timeout_seconds=3.0,
    )
    assert result.timeout_exceeded is False
    assert result.returncode == 0
    assert result.stdout == "ok"


def test_caller_dict_cannot_satisfy_strict_external_judge_gate(tmp_path, monkeypatch):
    """External provenance cannot be established by self-asserted booleans."""
    session = _legacy_release_session(tmp_path)
    _patch_release_non_cwd_gates(monkeypatch)
    monkeypatch.setattr(
        plan_mode,
        "ground_check",
        lambda *a, **k: {"ok": True, "missing": [], "verified": []},
    )

    plan_mode.record_judge(
        session,
        {
            "ok": True,
            "verdict": "go",
            "falsifiable_criteria": True,
            "source": "external_llm",
            "external": True,
            "provider": "made-up-provider",
            "model": "made-up-model",
        },
        round_version=1,
        plans_dir=tmp_path,
    )
    gate = plan_mode.release(
        session,
        min_score=0,
        require_judge=True,
        require_external_judge=True,
        plans_dir=tmp_path,
    )
    judge_check = next(c for c in gate["checks"] if c["name"] == "judge")
    assert judge_check["ok"] is False
    assert gate["ok"] is False


@pytest.mark.asyncio
async def test_judge_ensemble_never_reuses_prior_plan_version(monkeypatch, tmp_path):
    session = _legacy_release_session(tmp_path)
    session["rounds"].append({
        "version": 2,
        "ts": "test2",
        "score": 100.0,
        "delta": 0.0,
        "critiques": [],
        "sections": {},
        "note": None,
        "plan_text": "# Goal\nGoal: changed.\n## Tasks\n1. B. Output: b.txt.\n",
    })
    session["best_version"] = 2
    session["judge_log"] = [{
        "round_version": 1,
        "ok": True,
        "verdict": "go",
        "feasibility_0_100": 100,
        "falsifiable_criteria": True,
        "source": "external_llm",
        "external": True,
    }]

    async def no_live_vote(*args, **kwargs):
        return {"ok": False, "error": "disabled for test"}

    monkeypatch.setattr(plan_mode, "judge", no_live_vote)
    monkeypatch.setattr(plan_mode, "verify", lambda *a, **k: {"ok": True, "errors": []})
    monkeypatch.setattr(
        plan_mode,
        "simulate",
        lambda *a, **k: {"executable_plan": True, "problems": []},
    )

    entry = await plan_mode.judge_ensemble(
        session,
        session["rounds"][1]["plan_text"],
        "changed",
        n=3,
        plans_dir=tmp_path,
    )
    assert not any(v.get("round_version") == 1 for v in entry["votes"])


def _semantic_plan() -> PlanIR:
    prov = _prov()
    return PlanIR(
        plan_id="semantic-hash",
        goal_description="bind all execution semantics",
        actions=[
            ActionIR(
                action_id="a1",
                capability_name="cap",
                compensation_action_id="undo-a1",
                timeout_seconds=5.0,
                provenance=prov,
            )
        ],
        hard_constraints=[
            HardConstraint(
                constraint_id="hc1",
                description="must remain safe",
                condition=PredicateCondition(predicate="safe"),
                active_until_action_id="a1",
                provenance=prov,
            )
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="s1",
                description="done",
                condition=PredicateCondition(predicate="done"),
                is_mandatory=True,
            )
        ],
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: setattr(p.actions[0], "timeout_seconds", 99.0),
        lambda p: setattr(p.actions[0], "compensation_action_id", "different-undo"),
        lambda p: setattr(p.success_criteria[0], "is_mandatory", False),
        lambda p: setattr(p.hard_constraints[0], "active_until_action_id", None),
    ],
)
def test_plan_hash_binds_all_execution_and_commit_semantics(mutator):
    original = _semantic_plan()
    changed = original.model_copy(deep=True)
    before = original.compute_hash()
    mutator(changed)
    assert changed.compute_hash() != before


def test_expired_world_fact_cannot_start_execution_under_old_hash():
    prov = _prov(SourceType.OBSERVED_WORLD_STATE)
    ready = PredicateCondition(predicate="ready", expected_truth=FactTruth.VERIFIED_TRUE)
    fact = WorldFact(
        predicate="ready",
        truth=FactTruth.VERIFIED_TRUE,
        ttl_seconds=1.0,
        created_at=100.0,
        updated_at=100.0,
        provenance=prov,
    )
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="cap",
            description="test capability",
            preconditions=[ready],
            executor_command_template=[sys.executable, "-c", "pass"],
        )
    )
    plan = PlanIR(
        plan_id="ttl",
        goal_description="fresh state only",
        actions=[
            ActionIR(
                action_id="a1",
                capability_name="cap",
                preconditions=[ready],
                provenance=_prov(),
            )
        ],
    )
    session = PlanningSession(session_id="ttl-session")
    session.submit_draft(plan)
    result = session.validate_candidate(
        1,
        registry,
        observed_world_state=[fact],
        current_time=100.0,
    )
    assert result.status.value == "PASS"
    session.select_version(1)
    session.authorize_selected(registry, policy_hash="policy", ttl_seconds=60.0)

    with pytest.raises(StateDriftError):
        session.start_execution(
            registry,
            policy_hash="policy",
            current_world_facts=[fact],
            current_time=102.0,
        )


def test_registry_rejects_action_that_omits_capability_semantics():
    required = PredicateCondition(predicate="ready")
    effect = PredicateCondition(predicate="done")
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="cap",
            description="test capability",
            preconditions=[required],
            positive_effects=[effect],
        )
    )
    action = ActionIR(
        action_id="a1",
        capability_name="cap",
        preconditions=[],
        positive_effects=[],
        provenance=_prov(),
    )
    with pytest.raises(SchemaMismatchError):
        registry.validate_action(action)


def test_saga_recovery_cannot_report_rollback_without_running_compensation():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="do",
            description="effectful capability",
            default_compensation=CompensationAction(
                compensation_id="undo-do",
                capability_name="undo",
                parameter_mapping={},
            ),
        )
    )
    # Critically, the compensation capability has no executor contract.
    registry.register(CapabilityEntry(name="undo", description="missing executor"))
    plan = PlanIR(
        plan_id="saga",
        goal_description="must really undo",
        actions=[ActionIR(action_id="a1", capability_name="do", provenance=_prov())],
    )
    session = PlanningSession(session_id="saga-session", current_state=SessionState.EXECUTING)
    ledger = EvidenceLedger(session_id=session.session_id)
    report = SagaRecoveryManager().execute_saga_rollback(
        executed_steps=[
            StepExecutionResult(
                step_id="a1",
                capability_name="do",
                exit_code=0,
                witness_status=WitnessStatus.WITNESSED_TRUE,
                duration_ms=1.0,
            )
        ],
        plan_ir=plan,
        registry=registry,
        ledger=ledger,
        session=session,
    )
    assert report.status == RecoveryStatus.CONTAINMENT_FAILED
    assert report.compensated_steps_count == 0
    assert session.current_state in {SessionState.CONTAINMENT_FAILED, SessionState.FAILED}


def test_execution_contract_rejects_artifact_path_escape(tmp_path):
    outside = tmp_path.parent / "pr5-outside.txt"
    outside.write_text("host secret\n", encoding="utf-8")
    plan_text = f"""# Goal
Goal: test containment.

## Execution Contract
```json
{{
  "verification_commands": [["python", "-c", "print('ok')"]],
  "expected_artifacts": {{"../{outside.name}": {{"min_lines": 1}}}}
}}
```
"""
    result = validate_execution_contract(plan_text, cwd=tmp_path)
    assert result["ok"] is False
    assert any("workspace" in err.lower() or "escape" in err.lower() for err in result["errors"])


def test_diagnostic_probe_fact_key_preserves_argument_types():
    kwargs = {
        "probe_id": "p",
        "target_predicate": "item",
        "argv_pipeline": [["true"]],
    }
    as_int = DiagnosticProbe(target_args=[1], **kwargs)
    as_str = DiagnosticProbe(target_args=["1"], **kwargs)
    assert as_int.fact_key != as_str.fact_key


def test_recovery_decision_does_not_mutate_caller_evidence():
    evidence = DriftEvidence(
        task="task-3",
        step=3,
        suspected_onset=1,
        is_aligned=True,
        risk=0.1,
    )
    before = (evidence.step, evidence.suspected_onset, evidence.is_aligned)
    recovery_decision(evidence, K=2)
    assert (evidence.step, evidence.suspected_onset, evidence.is_aligned) == before


def test_release_after_artifact_release_does_not_report_historical_write_conflict():
    manager = IsolationManager()
    manager.acquire_artifact("out.txt", "agent-a", operation="write")
    manager.release_artifact("out.txt", "agent-a")
    manager.acquire_artifact("out.txt", "agent-b", operation="write")
    report = manager.detect_conflicts(isolation=OperationIsolation.SERIALIZABLE)
    assert report.ok is True


@pytest.mark.asyncio
async def test_speculative_rollout_async_has_outer_timeout():
    async def never_returns(ctx):
        await asyncio.sleep(10)
        return 1.0

    # The hardened API must bound injected callbacks.  A small timeout is
    # deliberate so CI never waits on the callback's own sleep.
    result = await plan_mode.speculative_rollout_async(
        "1. Test. Output: x.txt.",
        never_returns,
        timeout_seconds=0.02,
    )
    assert result["ok"] is False
    assert "timeout" in str(result.get("error", "")).lower()
