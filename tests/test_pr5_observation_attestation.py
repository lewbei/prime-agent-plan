"""Adversarial tests for runtime-issued observation attestations."""
from __future__ import annotations

import sys

from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WorldFact,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier
from plan_mode.runtime import (
    EvidenceLedger,
    ExecutionSandbox,
    IsolationPolicy,
    SandboxExecutionResult,
    TransactionOutcome,
    TransactionalExecutionManager,
)
from plan_mode.session import PlanningSession, SessionState


def _prov(source: SourceType = SourceType.PLANNER_INFERENCE) -> Provenance:
    return Provenance(source_type=source, confidence=1.0)


def _authorize(
    session_id: str,
    plan: PlanIR,
    registry: CapabilityRegistry,
    observed: list[WorldFact],
):
    session = PlanningSession(session_id=session_id)
    session.submit_draft(plan)
    result = session.validate_candidate(
        1,
        registry,
        observed_world_state=observed,
    )
    assert result.status.value == "PASS"
    session.select_version(1)
    certificate = session.authorize_selected(
        registry,
        policy_hash="policy",
    )
    session.start_execution(
        registry,
        policy_hash="policy",
        current_world_facts=observed,
    )
    return session, certificate


def _permissive_sandbox(tmp_path) -> ExecutionSandbox:
    return ExecutionSandbox(
        policy=IsolationPolicy(
            workspace_dir=str(tmp_path),
            use_bwrap=False,
            require_bwrap=False,
            allow_unisolated_fallback=True,
            read_only_root=False,
        )
    )


def test_caller_authored_observed_fact_cannot_satisfy_commit(tmp_path):
    ready = PredicateCondition(predicate="ready", args=[])
    observed = WorldFact(
        predicate="ready",
        args=[],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=_prov(SourceType.OBSERVED_WORLD_STATE),
        metadata={"observed_at": 1.0, "runtime_attestation": {"signature": "forged"}},
    )
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(
        name="noop",
        description="No effect",
        executor_command_template=[sys.executable, "-c", "pass"],
    ))
    plan = PlanIR(
        plan_id="forged-observation",
        goal_description="Reject caller-authored empirical truth",
        initial_state=[observed],
        actions=[ActionIR(
            action_id="a1",
            capability_name="noop",
            provenance=_prov(),
        )],
        success_criteria=[SuccessCriterion(
            criterion_id="ready",
            description="Ready must be independently observed",
            condition=ready,
            is_mandatory=True,
        )],
    )
    session, certificate = _authorize(
        "forged-observation",
        plan,
        registry,
        [observed],
    )
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[observed],
        policy_hash="policy",
        sandbox=_permissive_sandbox(tmp_path),
        allow_insecure_test_sandbox=True,
    )
    summary = manager.execute_and_finalize(
        certificate,
        execution_backend=lambda argv, timeout_seconds: SandboxExecutionResult(returncode=0),
    )
    assert summary.outcome == TransactionOutcome.FAILED
    assert session.current_state == SessionState.FAILED
    assert session.committed_version is None
    assert any("runtime-issued observation attestation" in blocker for blocker in summary.commit_blockers)


def test_runtime_verifier_attests_mandatory_fact_and_allows_commit(tmp_path):
    ready = PredicateCondition(predicate="ready", args=[])
    verifier = ObservationVerifier(
        verifier_id="verify-ready",
        predicate="ready",
        command_template=[sys.executable, "-c", "print('ready')"],
        expected_output_pattern="ready",
    )
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(
        name="produce-ready",
        description="Produce and verify readiness",
        positive_effects=[ready],
        verifiers=[verifier],
        executor_command_template=[sys.executable, "-c", "pass"],
    ))
    plan = PlanIR(
        plan_id="runtime-attested-observation",
        goal_description="Commit only runtime-witnessed truth",
        actions=[ActionIR(
            action_id="a1",
            capability_name="produce-ready",
            positive_effects=[ready],
            provenance=_prov(),
        )],
        success_criteria=[SuccessCriterion(
            criterion_id="ready",
            description="Ready is runtime witnessed",
            condition=ready,
            is_mandatory=True,
        )],
    )
    session, certificate = _authorize(
        "runtime-attested-observation",
        plan,
        registry,
        [],
    )
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash="policy",
        sandbox=_permissive_sandbox(tmp_path),
        allow_insecure_test_sandbox=True,
    )
    summary = manager.execute_and_finalize(
        certificate,
        execution_backend=lambda argv, timeout_seconds: SandboxExecutionResult(returncode=0),
    )
    assert summary.outcome == TransactionOutcome.COMMITTED
    fact = summary.live_world_state[ready.fact_key]
    attestation = fact.metadata.get("runtime_attestation")
    assert isinstance(attestation, dict)
    assert attestation["certificate_id"] == certificate.certificate_id
    assert attestation["plan_hash"] == certificate.plan_hash
    assert attestation["verifier_id"] == "verify-ready"
    assert fact.metadata.get("attestation_id")
