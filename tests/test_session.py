"""Tests for Planning Session, State Machine, Immutable Plan Versions, and Authorization Certificates."""

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
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.runtime import EvidenceLedger, TransactionOutcome, TransactionalExecutionManager
from plan_mode.runtime.sandbox import ExecutionSandbox, IsolationPolicy
from plan_mode.session import (
    SessionState,
    PlanVersion,
    AuthorizationCertificate,
    PlanningSession,
    InvalidStateTransitionError,
    CertificateExpiredError,
    StateDriftError,
    SignatureVerificationError,
)


@pytest.fixture
def mock_registry() -> CapabilityRegistry:
    from plan_mode.registry import ObservationVerifier
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="system.test_action",
            description="Test action",
            input_schema={"msg": {"type": "str", "required": True}},
            positive_effects=[PredicateCondition(predicate="task_done", args=[])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_done",
                    predicate="task_done",
                    command_template=["true"],
                )
            ],
            executor_command_template=["true"],
        )
    )
    return reg


@pytest.fixture
def valid_plan() -> PlanIR:
    prov = Provenance(source_type=SourceType.USER_REQUIREMENT)
    return PlanIR(
        plan_id="plan_sess_001",
        goal_description="Session test goal",
        initial_state=[
            WorldFact(predicate="init_ready", args=[], truth=FactTruth.VERIFIED_TRUE, provenance=prov, metadata={"evidence_ref": "ev_init"})
        ],
        actions=[
            ActionIR(
                action_id="act_01",
                capability_name="system.test_action",
                parameters={"msg": "run"},
                preconditions=[PredicateCondition(predicate="init_ready", args=[])],
                positive_effects=[PredicateCondition(predicate="task_done", args=[])],
                provenance=prov,
            )
        ],
    )


def test_session_lifecycle_happy_path(valid_plan: PlanIR, mock_registry: CapabilityRegistry):
    session = PlanningSession(session_id="sess_001")
    assert session.current_state == SessionState.DRAFT

    pv = session.submit_draft(valid_plan)
    assert pv.version_number == 1
    assert session.current_state == SessionState.IR_VALID
    assert session.best_candidate_version == 1

    val_res = session.validate_candidate(1, mock_registry, observed_world_state=valid_plan.initial_state)
    assert session.current_state == SessionState.FEASIBILITY
    assert val_res.status.value == "PASS"
    assert session.best_verified_version == 1

    session.select_version(1)
    assert session.current_state == SessionState.SELECTED

    cert = session.authorize_selected(mock_registry, policy_hash="policy_v1", ttl_seconds=10.0)
    assert session.current_state == SessionState.AUTHORIZED
    assert cert.plan_version == 1
    assert cert.verify_signature(session.secret_key) is True

    session.start_execution(mock_registry, policy_hash="policy_v1")
    assert session.current_state == SessionState.EXECUTING

    sandbox = ExecutionSandbox(
        IsolationPolicy(
            use_bwrap=False,
            require_bwrap=False,
            allow_unisolated_fallback=True,
            read_only_root=False,
        )
    )
    manager = TransactionalExecutionManager(
        session=session,
        registry=mock_registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=valid_plan.initial_state,
        policy_hash="policy_v1",
        sandbox=sandbox,
        allow_insecure_test_sandbox=True,
    )
    summary = manager.execute_and_finalize(cert)
    assert summary.outcome == TransactionOutcome.COMMITTED
    assert session.current_state == SessionState.COMMITTED
    assert session.committed_version == 1


def test_invalid_state_transition(valid_plan: PlanIR):
    session = PlanningSession(session_id="sess_002")
    with pytest.raises(InvalidStateTransitionError):
        session.transition_to(SessionState.AUTHORIZED)


def test_certificate_expired_rejection(valid_plan: PlanIR, mock_registry: CapabilityRegistry):
    session = PlanningSession(session_id="sess_003")
    session.submit_draft(valid_plan)
    session.validate_candidate(1, mock_registry, observed_world_state=valid_plan.initial_state)
    session.select_version(1)
    session.authorize_selected(mock_registry, policy_hash="policy_v1", ttl_seconds=-1.0)

    with pytest.raises(CertificateExpiredError):
        session.start_execution(mock_registry, policy_hash="policy_v1")


def test_state_drift_rejection(valid_plan: PlanIR, mock_registry: CapabilityRegistry):
    session = PlanningSession(session_id="sess_004")
    session.submit_draft(valid_plan)
    session.validate_candidate(1, mock_registry, observed_world_state=valid_plan.initial_state)
    session.select_version(1)
    session.authorize_selected(mock_registry, policy_hash="policy_v1", ttl_seconds=60.0)

    drifted_facts = [
        WorldFact(
            predicate="corrupted_fact",
            args=["adversary"],
            truth=FactTruth.VERIFIED_TRUE,
            provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        )
    ]

    with pytest.raises(StateDriftError):
        session.start_execution(mock_registry, policy_hash="policy_v1", current_world_facts=drifted_facts)


def test_signature_tamper_detection(valid_plan: PlanIR, mock_registry: CapabilityRegistry):
    session = PlanningSession(session_id="sess_005")
    session.submit_draft(valid_plan)
    session.validate_candidate(1, mock_registry, observed_world_state=valid_plan.initial_state)
    session.select_version(1)
    cert = session.authorize_selected(mock_registry, policy_hash="policy_v1")

    tampered_cert = cert.model_copy(update={"plan_hash": "0000000000000000000000000000000000000000000000000000000000000000"})
    assert tampered_cert.verify_signature(session.secret_key) is False
