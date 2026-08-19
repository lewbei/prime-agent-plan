"""Adversarial fail-closed tests for Phase 3 transaction finalization."""
from __future__ import annotations

import pytest

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
from plan_mode.registry import (
    CapabilityEntry,
    CapabilityRegistry,
    CompensationAction,
    ObservationVerifier,
)
from plan_mode.runtime import (
    EvidenceLedger,
    ExecutionSandbox,
    LedgerEventType,
    TransactionOutcome,
    TransactionalExecutionManager,
)
from plan_mode.runtime.sandbox import IsolationPolicy
from plan_mode.session import CommitGateError, PlanningSession, SessionState


def _prov(source: SourceType = SourceType.PLANNER_INFERENCE) -> Provenance:
    return Provenance(source_type=source, confidence=1.0)


def _cond(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate=predicate, args=args, expected_truth=truth)


def _action(action_id: str, cap: str, params=None, effects=None) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=cap,
        parameters=params or {},
        positive_effects=effects or [],
        provenance=_prov(),
    )


def _test_sandbox() -> ExecutionSandbox:
    return ExecutionSandbox(
        IsolationPolicy(
            use_bwrap=False,
            require_bwrap=False,
            allow_unisolated_fallback=True,
            read_only_root=False,
        )
    )


def _base_registry(
    *,
    compensation_name: str = "fs.remove",
    mapping: dict[str, str] | None = None,
) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="fs.touch",
            description="Create file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="exists",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["touch", "{path}"],
            default_compensation=CompensationAction(
                compensation_id="undo",
                capability_name=compensation_name,
                parameter_mapping=mapping or {"path": "{path}"},
                timeout_seconds=10.0,
            ),
        )
    )
    registry.register(
        CapabilityEntry(
            name="fail.after",
            description="Fail after a prior side effect",
            executor_command_template=["false"],
        )
    )
    return registry


def _register_remove(
    registry: CapabilityRegistry,
    *,
    verifier_command: list[str] | None = None,
    preconditions: list[PredicateCondition] | None = None,
) -> None:
    registry.register(
        CapabilityEntry(
            name="fs.remove",
            description="Remove file",
            input_schema={"path": {"type": "str", "required": True}},
            preconditions=preconditions or [],
            negative_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="absent",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=verifier_command or ["test", "!", "-f", "{path}"],
                )
            ],
            executor_command_template=["rm", "-f", "{path}"],
        )
    )


def _plan(path: str, plan_id: str = "phase3-adv") -> PlanIR:
    return PlanIR(
        plan_id=plan_id,
        goal_description="Create a side effect then force recovery",
        actions=[
            _action("a1", "fs.touch", {"path": path}, [_cond("file_exists", [path])]),
            _action("a2", "fail.after"),
        ],
    )


def _prepare(plan: PlanIR, registry: CapabilityRegistry):
    session = PlanningSession(session_id=f"s-{plan.plan_id}")
    session.submit_draft(plan)
    result = session.validate_candidate(1, registry, observed_world_state=[])
    assert result.status.value == "PASS"
    session.select_version(1)
    policy = registry.compute_registry_hash()
    cert = session.authorize_selected(registry, policy_hash=policy)
    session.start_execution(registry, policy_hash=policy, current_world_facts=[])
    ledger = EvidenceLedger(session_id=session.session_id)
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy,
        sandbox=_test_sandbox(),
        allow_insecure_test_sandbox=True,
    )
    return session, ledger, manager, cert, policy


def test_compensation_without_observable_effects_cannot_claim_rollback(tmp_path):
    target = tmp_path / "no-observation.txt"
    registry = _base_registry(compensation_name="comp.no_effect")
    registry.register(
        CapabilityEntry(
            name="comp.no_effect",
            description="Deletes but declares no observable recovery effect",
            input_schema={"path": {"type": "str", "required": True}},
            executor_command_template=["rm", "-f", "{path}"],
        )
    )
    session, _, manager, cert, _ = _prepare(_plan(str(target), "no-observable-comp"), registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()


def test_missing_compensation_parameter_placeholder_fails_before_command(tmp_path):
    target = tmp_path / "mapping.txt"
    registry = _base_registry(mapping={"path": "{missing_path}"})
    _register_remove(registry)
    session, _, manager, cert, _ = _prepare(_plan(str(target), "bad-map"), registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()
    assert "missing original parameter" in summary.compensation_results[-1].error_message


def test_unsatisfied_compensation_precondition_fails_before_command(tmp_path):
    target = tmp_path / "precondition.txt"
    registry = _base_registry()
    _register_remove(registry, preconditions=[_cond("rollback_allowed", [])])
    session, _, manager, cert, _ = _prepare(_plan(str(target), "comp-precondition"), registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()
    assert "compensation precondition" in summary.compensation_results[-1].error_message


def test_successful_compensation_command_with_failed_verifier_is_not_rollback(tmp_path):
    target = tmp_path / "verifier-fails.txt"
    registry = _base_registry()
    _register_remove(registry, verifier_command=["false"])
    session, _, manager, cert, _ = _prepare(_plan(str(target), "comp-verifier-fail"), registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert not target.exists()
    assert summary.compensation_results[-1].executed is True
    assert summary.compensation_results[-1].verified is False


def test_registry_drift_during_execution_blocks_unbound_compensation(tmp_path):
    target = tmp_path / "registry-drift.txt"
    registry = _base_registry()
    _register_remove(registry)
    session, ledger, manager, cert, _ = _prepare(_plan(str(target), "registry-drift"), registry)
    sandbox = ExecutionSandbox()

    def backend(argv, *, timeout_seconds):
        result = sandbox.execute_argv_pipeline([argv], timeout_seconds=timeout_seconds)
        if argv == ["false"]:
            registry.register(
                CapabilityEntry(
                    name="drift.injected",
                    description="Mutates registry after authorization",
                    executor_command_template=["true"],
                )
            )
        return result

    summary = manager.execute_and_finalize(cert, execution_backend=backend)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()
    assert not any(r.event_type == LedgerEventType.COMPENSATION_EXECUTED for r in ledger.records)


def test_policy_drift_during_execution_blocks_unbound_compensation(tmp_path):
    target = tmp_path / "policy-drift.txt"
    registry = _base_registry()
    _register_remove(registry)
    session, ledger, manager, cert, _ = _prepare(_plan(str(target), "policy-drift"), registry)
    sandbox = ExecutionSandbox()

    def backend(argv, *, timeout_seconds):
        result = sandbox.execute_argv_pipeline([argv], timeout_seconds=timeout_seconds)
        if argv == ["false"]:
            manager.policy_hash = "changed-policy"
        return result

    summary = manager.execute_and_finalize(cert, execution_backend=backend)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()
    assert not any(r.event_type == LedgerEventType.COMPENSATION_EXECUTED for r in ledger.records)


def test_commit_rejects_mandatory_criterion_with_non_observed_provenance():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    criterion = _cond("approved", [])
    plan = PlanIR(
        plan_id="forged-criterion",
        goal_description="Do not accept planner-owned criterion evidence",
        actions=[_action("a1", "noop")],
        success_criteria=[
            SuccessCriterion(
                criterion_id="approved",
                description="Approval observed",
                condition=criterion,
                is_mandatory=True,
            )
        ],
    )
    session = PlanningSession(session_id="s-forged-criterion")
    session.submit_draft(plan)
    from plan_mode.epistemic_validator import PlanValidationResult, ValidationStatus

    class PassValidator:
        default_ttl_decay_to_unknown = True
        def validate_plan(self, *args, **kwargs):
            return PlanValidationResult(status=ValidationStatus.PASS)

    session.validate_candidate(1, registry, validator=PassValidator(), observed_world_state=[])
    session.select_version(1)
    policy = registry.compute_registry_hash()
    session.authorize_selected(registry, policy_hash=policy)
    session.start_execution(registry, policy_hash=policy, current_world_facts=[])
    forged = WorldFact(
        predicate="approved",
        args=[],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=_prov(SourceType.PLANNER_INFERENCE),
    )
    session.record_execution_result(True, [forged])
    with pytest.raises(CommitGateError):
        session.commit_execution(live_world_state=[forged])
    assert session.current_state == SessionState.EXECUTING


def test_commit_rejects_world_state_different_from_attested_execution():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    plan = PlanIR(plan_id="commit-world-drift", goal_description="Bind commit world", actions=[_action("a1", "noop")])
    session = PlanningSession(session_id="s-commit-world-drift")
    session.submit_draft(plan)
    session.validate_candidate(1, registry, observed_world_state=[])
    session.select_version(1)
    policy = registry.compute_registry_hash()
    session.authorize_selected(registry, policy_hash=policy)
    session.start_execution(registry, policy_hash=policy, current_world_facts=[])
    session.record_execution_result(True, [])
    extra = WorldFact(
        predicate="unexpected",
        args=[],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=_prov(SourceType.OBSERVED_WORLD_STATE),
    )
    with pytest.raises(CommitGateError):
        session.commit_execution(live_world_state=[extra])
    assert session.current_state == SessionState.EXECUTING
