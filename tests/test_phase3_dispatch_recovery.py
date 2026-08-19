"""Phase 3 dispatch-time recovery and transaction-finalization invariants."""
from __future__ import annotations

import pytest

from plan_mode.ir import ActionIR, PlanIR, PredicateCondition, Provenance, SourceType
from plan_mode.registry import (
    CapabilityEntry,
    CapabilityRegistry,
    CompensationAction,
    ObservationVerifier,
)
from plan_mode.runtime import (
    EvidenceLedger,
    ExecutionPlanManager,
    ExecutionSandbox,
    LedgerEventType,
    TransactionOutcome,
    TransactionalExecutionManager,
)
from plan_mode.session import PlanningSession, SessionState


def _prov():
    return Provenance(source_type=SourceType.PLANNER_INFERENCE, confidence=1.0)


def _action(action_id: str, cap: str, params=None, effects=None):
    return ActionIR(
        action_id=action_id,
        capability_name=cap,
        parameters=params or {},
        positive_effects=effects or [],
        provenance=_prov(),
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
    return session, policy, cert


def test_base_execution_manager_never_commits_session():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="noop",
            description="Successful no-op",
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="executor-does-not-commit",
        goal_description="Execution evidence and commit must be separate",
        actions=[_action("a1", "noop")],
    )
    session, policy, cert = _prepare(plan, registry)
    ledger = EvidenceLedger(session_id=session.session_id)
    executor = ExecutionPlanManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy,
    )
    summary = executor.execute_authorized_plan(cert)
    assert summary.success is True
    assert session.current_state == SessionState.EXECUTING
    assert session.committed_version is None
    assert not any(r.event_type == LedgerEventType.PLAN_COMMITTED for r in ledger.records)


def test_duplicate_action_ids_are_rejected_before_any_dispatch():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="noop",
            description="Successful no-op",
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="duplicate-action-ids",
        goal_description="Ambiguous saga identity must not execute",
        actions=[_action("same", "noop"), _action("same", "noop")],
    )
    session, policy, cert = _prepare(plan, registry)
    ledger = EvidenceLedger(session_id=session.session_id)
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy,
    )
    calls = []

    def backend(argv, *, timeout_seconds):
        calls.append(list(argv))
        return ExecutionSandbox().execute_argv_pipeline([argv], timeout_seconds=timeout_seconds)

    with pytest.raises(ValueError, match="unique action_id"):
        manager.execute_and_finalize(cert, execution_backend=backend)
    assert calls == []
    assert not any(r.event_type == LedgerEventType.ACTION_DISPATCHED for r in ledger.records)


def test_backend_exception_after_real_side_effect_triggers_verified_compensation(tmp_path):
    target = tmp_path / "dispatch-exception.txt"
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="fs.touch",
            description="Create file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[PredicateCondition(predicate="file_exists", args=["{path}"])],
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
                compensation_id="remove-file",
                capability_name="fs.remove",
                parameter_mapping={"path": "{path}"},
            ),
        )
    )
    registry.register(
        CapabilityEntry(
            name="fs.remove",
            description="Remove file",
            input_schema={"path": {"type": "str", "required": True}},
            negative_effects=[PredicateCondition(predicate="file_exists", args=["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="absent",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "!", "-f", "{path}"],
                )
            ],
            executor_command_template=["rm", "-f", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="backend-exception-recovery",
        goal_description="Recover even if launcher raises after side effect",
        actions=[
            _action(
                "a1",
                "fs.touch",
                {"path": str(target)},
                [PredicateCondition(predicate="file_exists", args=[str(target)])],
            )
        ],
    )
    session, policy, cert = _prepare(plan, registry)
    ledger = EvidenceLedger(session_id=session.session_id)
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy,
    )
    sandbox = ExecutionSandbox()
    touch_calls = 0

    def backend(argv, *, timeout_seconds):
        nonlocal touch_calls
        result = sandbox.execute_argv_pipeline([argv], timeout_seconds=timeout_seconds)
        if argv and argv[0] == "touch":
            touch_calls += 1
            assert target.exists()
            raise RuntimeError("launcher lost acknowledgement after process side effect")
        return result

    summary = manager.execute_and_finalize(cert, execution_backend=backend)
    assert touch_calls == 1
    assert summary.outcome == TransactionOutcome.ROLLED_BACK
    assert session.current_state == SessionState.ROLLED_BACK
    assert not target.exists()
    assert any(r.event_type == LedgerEventType.ACTION_DISPATCHED for r in ledger.records)
    assert any(r.event_type == LedgerEventType.COMPENSATION_VERIFIED for r in ledger.records)
