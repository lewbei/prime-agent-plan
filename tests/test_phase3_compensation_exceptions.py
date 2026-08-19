"""Phase 3 must enter containment failure on recovery-path exceptions."""
from __future__ import annotations

from plan_mode.ir import ActionIR, PlanIR, PredicateCondition, Provenance, SourceType
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, CompensationAction, ObservationVerifier
from plan_mode.runtime import EvidenceLedger, ExecutionSandbox, TransactionOutcome, TransactionalExecutionManager
from plan_mode.runtime.sandbox import IsolationPolicy
from plan_mode.session import PlanningSession, SessionState


def _prov():
    return Provenance(source_type=SourceType.PLANNER_INFERENCE, confidence=1.0)


def _action(action_id, cap, params=None, effects=None):
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


def _setup(path: str, *, verifier_pattern=None):
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="fs.touch",
            description="Create file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[PredicateCondition(predicate="file_exists", args=["{path}"])],
            verifiers=[ObservationVerifier(
                verifier_id="exists",
                predicate="file_exists",
                target_args_mapping=["{path}"],
                command_template=["test", "-f", "{path}"],
            )],
            executor_command_template=["touch", "{path}"],
            default_compensation=CompensationAction(
                compensation_id="undo",
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
            verifiers=[ObservationVerifier(
                verifier_id="absent",
                predicate="file_exists",
                target_args_mapping=["{path}"],
                command_template=["test", "!", "-f", "{path}"],
                expected_output_pattern=verifier_pattern,
            )],
            executor_command_template=["rm", "-f", "{path}"],
        )
    )
    registry.register(CapabilityEntry(
        name="fail.after",
        description="Later failure",
        executor_command_template=["false"],
    ))
    plan = PlanIR(
        plan_id=f"phase3-exception-{verifier_pattern}",
        goal_description="Exercise recovery exception path",
        actions=[
            _action("a1", "fs.touch", {"path": path}, [PredicateCondition(predicate="file_exists", args=[path])]),
            _action("a2", "fail.after"),
        ],
    )
    session = PlanningSession(session_id=f"s-{id(plan)}")
    session.submit_draft(plan)
    assert session.validate_candidate(1, registry, observed_world_state=[]).status.value == "PASS"
    session.select_version(1)
    policy = registry.compute_registry_hash()
    cert = session.authorize_selected(registry, policy_hash=policy)
    session.start_execution(registry, policy_hash=policy, current_world_facts=[])
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy,
        sandbox=_test_sandbox(),
        allow_insecure_test_sandbox=True,
    )
    return session, manager, cert


def test_compensation_backend_exception_enters_containment_failed(tmp_path):
    target = tmp_path / "backend-exception.txt"
    session, manager, cert = _setup(str(target))
    sandbox = ExecutionSandbox()

    def backend(argv, *, timeout_seconds):
        result = sandbox.execute_argv_pipeline([argv], timeout_seconds=timeout_seconds)
        if argv and argv[0] == "rm":
            assert not target.exists()
            raise RuntimeError("lost recovery acknowledgement")
        return result

    summary = manager.execute_and_finalize(cert, execution_backend=backend)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert not target.exists()
    assert summary.compensation_results[-1].executed is True
    assert summary.compensation_results[-1].verified is False
    assert "backend raised" in summary.compensation_results[-1].error_message


def test_compensation_verifier_exception_enters_containment_failed(tmp_path):
    target = tmp_path / "verifier-exception.txt"
    session, manager, cert = _setup(str(target), verifier_pattern="[")
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.CONTAINMENT_FAILED
    assert session.current_state == SessionState.CONTAINMENT_FAILED
    assert not target.exists()
    assert summary.compensation_results[-1].executed is True
    assert summary.compensation_results[-1].verified is False
    assert "verifier raised" in summary.compensation_results[-1].error_message
