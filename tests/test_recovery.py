"""Tests for Saga-style Recovery and Compensation Execution."""

import pytest
from plan_mode.ir import (
    FactTruth,
    Provenance,
    SourceType,
    WorldFact,
    PredicateCondition,
    ActionIR,
    PlanIR,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, CompensationAction
from plan_mode.session import PlanningSession, SessionState
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.executor import StepExecutionResult, WitnessStatus
from plan_mode.recovery import SagaRecoveryManager, SagaRecoveryReport, RecoveryStatus


@pytest.fixture
def recovery_setup():
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.create_temp_dir",
            description="Create temporary directory",
            input_schema={"dir_path": {"type": "str", "required": True}},
            default_compensation=CompensationAction(
                compensation_id="fs.delete_temp_dir_comp",
                capability_name="fs.delete_temp_dir",
                parameter_mapping={"dir_path": "dir_path"},
            ),
        )
    )
    reg.register(
        CapabilityEntry(
            name="fs.delete_temp_dir",
            description="Delete temporary directory",
            input_schema={"dir_path": {"type": "str", "required": True}},
            executor_command_template=["true"],
        )
    )
    reg.register(
        CapabilityEntry(
            name="db.provision_schema",
            description="Provision database schema",
            input_schema={"schema_name": {"type": "str", "required": True}},
            # No compensation action defined
        )
    )

    prov = Provenance(source_type=SourceType.USER_REQUIREMENT)
    plan = PlanIR(
        plan_id="plan_saga_001",
        goal_description="Create dir then provision schema",
        initial_state=[
            WorldFact(predicate="disk_ready", args=[], truth=FactTruth.VERIFIED_TRUE, provenance=prov)
        ],
        actions=[
            ActionIR(
                action_id="act_01",
                capability_name="fs.create_temp_dir",
                parameters={"dir_path": "/tmp/job_123"},
                provenance=prov,
            ),
            ActionIR(
                action_id="act_02",
                capability_name="db.provision_schema",
                parameters={"schema_name": "job_schema"},
                provenance=prov,
            ),
        ],
    )

    session = PlanningSession(session_id="sess_saga_001")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    session.authorize_selected(reg, policy_hash="pol_v1")
    session.start_execution(reg, policy_hash="pol_v1")

    ledger = EvidenceLedger(session_id=session.session_id)
    return session, plan, reg, ledger


def test_saga_successful_rollback(recovery_setup):
    session, plan, reg, ledger = recovery_setup
    
    # Step 1 succeeded, Step 2 failed
    executed_steps = [
        StepExecutionResult(
            step_id="act_01",
            capability_name="fs.create_temp_dir",
            exit_code=0,
            witness_status=WitnessStatus.WITNESSED_TRUE,
            duration_ms=10.0,
        ),
        StepExecutionResult(
            step_id="act_02",
            capability_name="db.provision_schema",
            exit_code=1,
            witness_status=WitnessStatus.WITNESSED_FALSE,
            duration_ms=5.0,
            error_message="Database connection refused",
        ),
    ]

    manager = SagaRecoveryManager(sandbox=None)
    report = manager.execute_saga_rollback(
        executed_steps=executed_steps,
        plan_ir=plan,
        registry=reg,
        ledger=ledger,
        session=session,
    )

    assert report.status == RecoveryStatus.ROLLED_BACK
    assert report.compensated_steps_count == 1
    assert session.current_state == SessionState.ROLLED_BACK
    assert ledger.verify_integrity() is True


def test_saga_uncompensated_step_containment(recovery_setup):
    session, plan, reg, ledger = recovery_setup

    # If step 2 had succeeded and step 3 failed, but step 2 has no compensation:
    executed_steps = [
        StepExecutionResult(
            step_id="act_01",
            capability_name="fs.create_temp_dir",
            exit_code=0,
            witness_status=WitnessStatus.WITNESSED_TRUE,
            duration_ms=10.0,
        ),
        StepExecutionResult(
            step_id="act_02",
            capability_name="db.provision_schema",
            exit_code=0,
            witness_status=WitnessStatus.WITNESSED_TRUE,
            duration_ms=15.0,
        ),
    ]

    manager = SagaRecoveryManager(sandbox=None)
    report = manager.execute_saga_rollback(
        executed_steps=executed_steps,
        plan_ir=plan,
        registry=reg,
        ledger=ledger,
        session=session,
    )

    # Step 2 had no compensation -> containment failure flagged
    assert report.status == RecoveryStatus.CONTAINMENT_FAILED
    assert "db.provision_schema" in report.uncompensated_capabilities
    assert session.current_state == SessionState.FAILED
