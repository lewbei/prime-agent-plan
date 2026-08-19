"""Phase 3: verified commit gating and real saga compensation tests."""
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
)
from plan_mode.registry import (
    CapabilityEntry,
    CapabilityRegistry,
    CompensationAction,
    ObservationVerifier,
)
from plan_mode.runtime import (
    EvidenceLedger,
    LedgerEventType,
    TransactionOutcome,
    TransactionalExecutionManager,
)
from plan_mode.runtime.sandbox import ExecutionSandbox, IsolationPolicy
from plan_mode.session import CommitGateError, PlanningSession, SessionState


def _prov() -> Provenance:
    return Provenance(source_type=SourceType.PLANNER_INFERENCE, confidence=1.0)


def _cond(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate=predicate, args=args, expected_truth=truth)


def _action(
    action_id: str,
    capability: str,
    params: dict | None = None,
    positive: list[PredicateCondition] | None = None,
) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability,
        parameters=params or {},
        positive_effects=positive or [],
        provenance=_prov(),
    )


def _test_sandbox() -> ExecutionSandbox:
    """Explicit test-only process runner for Phase 3 semantic tests."""
    return ExecutionSandbox(
        IsolationPolicy(
            use_bwrap=False,
            require_bwrap=False,
            allow_unisolated_fallback=True,
            read_only_root=False,
        )
    )


def _register_file_caps(
    registry: CapabilityRegistry,
    *,
    include_compensation: bool = True,
    compensation_executor_works: bool = True,
    compensation_verifier_works: bool = True,
) -> None:
    compensation = (
        CompensationAction(
            compensation_id="undo_touch",
            capability_name="fs.remove",
            parameter_mapping={"path": "{path}"},
            timeout_seconds=10.0,
        )
        if include_compensation
        else None
    )
    registry.register(
        CapabilityEntry(
            name="fs.touch",
            description="Create a file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="file_exists",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["touch", "{path}"],
            default_compensation=compensation,
        )
    )
    registry.register(
        CapabilityEntry(
            name="fs.remove",
            description="Remove a file",
            input_schema={"path": {"type": "str", "required": True}},
            negative_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="file_absent",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=(
                        ["test", "!", "-f", "{path}"]
                        if compensation_verifier_works
                        else ["false"]
                    ),
                )
            ],
            executor_command_template=(
                ["rm", "-f", "{path}"]
                if compensation_executor_works
                else ["false"]
            ),
        )
    )


def _register_fail_cap(registry: CapabilityRegistry) -> None:
    registry.register(
        CapabilityEntry(
            name="fail.after",
            description="Deterministic failure after earlier side effects",
            input_schema={},
            executor_command_template=["false"],
        )
    )


def _prepare(plan: PlanIR, registry: CapabilityRegistry):
    session = PlanningSession(session_id=f"s-{plan.plan_id}")
    session.submit_draft(plan)
    result = session.validate_candidate(1, registry, observed_world_state=[])
    assert result.status.value == "PASS"
    session.select_version(1)
    policy_hash = registry.compute_registry_hash()
    cert = session.authorize_selected(registry, policy_hash=policy_hash)
    session.start_execution(registry, policy_hash=policy_hash, current_world_facts=[])
    ledger = EvidenceLedger(session_id=session.session_id)
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy_hash,
        sandbox=_test_sandbox(),
        allow_insecure_test_sandbox=True,
    )
    return session, ledger, manager, cert


def test_commit_requires_runtime_attestation_even_if_state_is_executing():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="noop",
            description="No-op",
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="commit-gate",
        goal_description="Commit gate",
        actions=[_action("a1", "noop")],
    )
    session = PlanningSession(session_id="commit-gate")
    session.submit_draft(plan)
    session.validate_candidate(1, registry, observed_world_state=[])
    session.select_version(1)
    policy = registry.compute_registry_hash()
    session.authorize_selected(registry, policy_hash=policy)
    session.start_execution(registry, policy_hash=policy, current_world_facts=[])
    with pytest.raises(CommitGateError):
        session.commit_execution(live_world_state={})
    assert session.current_state == SessionState.EXECUTING


def test_verified_mandatory_criterion_is_required_before_commit(tmp_path):
    target = tmp_path / "committed.txt"
    registry = CapabilityRegistry()
    _register_file_caps(registry)
    plan = PlanIR(
        plan_id="verified-commit",
        goal_description="Create and verify file",
        actions=[
            _action(
                "a1",
                "fs.touch",
                {"path": str(target)},
                [_cond("file_exists", [str(target)])],
            )
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="file-created",
                description="Target file exists",
                condition=_cond("file_exists", [str(target)]),
                is_mandatory=True,
            )
        ],
    )
    session, ledger, manager, cert = _prepare(plan, registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.COMMITTED
    assert session.current_state == SessionState.COMMITTED
    assert target.exists()
    assert any(r.event_type == LedgerEventType.PLAN_COMMITTED for r in ledger.records)


def test_failed_witness_prevents_commit_and_rolls_back(tmp_path):
    target = tmp_path / "never-created.txt"
    registry = CapabilityRegistry()
    _register_file_caps(registry)
    cap = registry.get("fs.touch")
    registry.register(cap.model_copy(update={"executor_command_template": ["true"]}))
    plan = PlanIR(
        plan_id="witness-fail",
        goal_description="Verifier must block commit",
        actions=[
            _action(
                "a1",
                "fs.touch",
                {"path": str(target)},
                [_cond("file_exists", [str(target)])],
            )
        ],
    )
    session, _, manager, cert = _prepare(plan, registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.ROLLED_BACK
    assert session.current_state == SessionState.ROLLED_BACK
    assert session.committed_version is None


def test_compensation_command_actually_executes_and_is_verified(tmp_path):
    target = tmp_path / "side-effect.txt"
    registry = CapabilityRegistry()
    _register_file_caps(registry)
    _register_fail_cap(registry)
    plan = PlanIR(
        plan_id="real-comp",
        goal_description="Compensate after later failure",
        actions=[
            _action(
                "a1",
                "fs.touch",
                {"path": str(target)},
                [_cond("file_exists", [str(target)])],
            ),
            _action("a2", "fail.after"),
        ],
    )
    session, ledger, manager, cert = _prepare(plan, registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.ROLLED_BACK
    assert session.current_state == SessionState.ROLLED_BACK
    assert not target.exists()
    assert len(summary.compensation_results) == 1
    assert summary.compensation_results[0].executed is True
    assert summary.compensation_results[0].verified is True
    events = [r.event_type for r in ledger.records]
    assert LedgerEventType.COMPENSATION_EXECUTED in events
    assert LedgerEventType.COMPENSATION_VERIFIED in events


def test_failed_compensation_postcondition_enters_containment_failed(tmp_path):
    target = tmp_path / "uncontained.txt"
    registry = CapabilityRegistry()
    _register_file_caps(registry, compensation_executor_works=False)
    _register_fail_cap(registry)
    plan = PlanIR(
        plan_id="containment-fail",
        goal_description="Compensation failure must be explicit",
        actions=[
            _action(
                "a1",
                "fs.touch",
                {"path": str(target)},
                [_cond("file_exists", [str(target)])],
            ),
            _action("a2", "fail.after"),
        ],
    )
    session, ledger, manager, cert = _prepare(plan, registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()
    assert any(r.event_type == LedgerEventType.CONTAINMENT_FAILED for r in ledger.records)


def test_missing_compensation_for_executed_effectful_action_enters_containment_failed(tmp_path):
    target = tmp_path / "no-comp.txt"
    registry = CapabilityRegistry()
    _register_file_caps(registry, include_compensation=False)
    _register_fail_cap(registry)
    plan = PlanIR(
        plan_id="missing-comp",
        goal_description="Missing compensation must fail closed",
        actions=[
            _action(
                "a1",
                "fs.touch",
                {"path": str(target)},
                [_cond("file_exists", [str(target)])],
            ),
            _action("a2", "fail.after"),
        ],
    )
    session, _, manager, cert = _prepare(plan, registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert target.exists()


def test_compensation_runs_in_reverse_execution_order(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    registry = CapabilityRegistry()
    _register_file_caps(registry)
    _register_fail_cap(registry)
    plan = PlanIR(
        plan_id="reverse-saga",
        goal_description="Rollback in reverse order",
        actions=[
            _action("a1", "fs.touch", {"path": str(first)}, [_cond("file_exists", [str(first)])]),
            _action("a2", "fs.touch", {"path": str(second)}, [_cond("file_exists", [str(second)])]),
            _action("a3", "fail.after"),
        ],
    )
    session, ledger, manager, cert = _prepare(plan, registry)
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.ROLLED_BACK
    assert not first.exists()
    assert not second.exists()
    compensated = [
        r.payload["original_step_id"]
        for r in ledger.records
        if r.event_type == LedgerEventType.COMPENSATION_EXECUTED
    ]
    assert compensated == ["a2", "a1"]
    assert [r.original_step_id for r in summary.compensation_results] == ["a2", "a1"]
