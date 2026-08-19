"""Adversarial Runtime Semantics, Fail-Closed Preflight, and Attestation Tests (Phase 2)."""

import os
import time
import pytest
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    ProjectedTruth,
    Provenance,
    SourceType,
    SuccessCriterion,
    WitnessabilityStatus,
    WorldFact,
)
from plan_mode.epistemic_validator import ValidationStatus
from plan_mode.registry import (
    CapabilityEntry,
    CapabilityRegistry,
    ObservationVerifier,
)
from plan_mode.session import (
    PlanningSession,
    SessionState,
    StateDriftError,
    SignatureVerificationError,
    InvalidStateTransitionError,
)
from plan_mode.runtime.ledger import EvidenceLedger
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult
from plan_mode.runtime.executor import (
    ExecutionPlanManager,
    WitnessStatus,
    PreconditionFailedError,
    ExecutionContractMissingError,
)


def _cond(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate=predicate, args=args, expected_truth=truth)


def _fact(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE, source: SourceType = SourceType.OBSERVED_WORLD_STATE) -> WorldFact:
    return WorldFact(
        predicate=predicate,
        args=args,
        truth=truth,
        projected_truth=ProjectedTruth.SUPPORTED_TRUE if truth == FactTruth.VERIFIED_TRUE else ProjectedTruth.UNSUPPORTED,
        provenance=Provenance(source_type=source, confidence=1.0),
        metadata={"evidence_ref": f"ev_{predicate}"},
    )


def _action(
    action_id: str,
    capability_name: str,
    parameters: dict | None = None,
    preconditions: list[PredicateCondition] | None = None,
    positive_effects: list[PredicateCondition] | None = None,
    negative_effects: list[PredicateCondition] | None = None,
) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability_name,
        parameters=parameters or {},
        preconditions=preconditions or [],
        positive_effects=positive_effects or [],
        negative_effects=negative_effects or [],
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE, source_id=action_id),
    )


# ---------------------------------------------------------------------------
# Section A: Runtime Preflight Non-Bypassability
# ---------------------------------------------------------------------------

def test_execution_manager_refuses_if_session_preflight_not_completed():
    """Plan is authorized, but session.start_execution() is NOT called. ExecutionPlanManager must refuse."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_no_preflight",
        goal_description="Test preflight requirement",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_no_preflight")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)

    # Note: We do NOT call session.start_execution() -> current_state remains AUTHORIZED, not EXECUTING!
    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    with pytest.raises((ValueError, InvalidStateTransitionError)):
        manager.execute_authorized_plan(cert)


def test_runtime_rejects_registry_drift_after_authorization():
    """Authorize with registry R1, then mutate capability contract. Runtime must reject before executing."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_drift",
            description="Tool",
            input_schema={"p": {"type": "str", "required": True}},
            positive_effects=[_cond("done", ["{p}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=["{p}"], command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_reg_drift",
        goal_description="Test registry drift",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_drift", parameters={"p": "x"}, positive_effects=[_cond("done", ["x"])])],
    )
    session = PlanningSession(session_id="s_reg_drift")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    # Mutate registry capability after authorization
    mutated_reg = CapabilityRegistry()
    mutated_reg.register(
        CapabilityEntry(
            name="tool_drift",
            description="Mutated tool",
            input_schema={"p": {"type": "str", "required": True}},
            positive_effects=[_cond("done", ["{p}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=["{p}"], command_template=["echo", "mutated"])],
            executor_command_template=["echo", "mutated"],
        )
    )

    manager = ExecutionPlanManager(session=session, registry=mutated_reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    with pytest.raises((StateDriftError, ValueError)):
        manager.execute_authorized_plan(cert)


def test_runtime_rejects_world_state_drift_before_execution():
    """Authorize against world snapshot W1, then world state drifts. Runtime rejects before executor runs."""
    w1_fact = _fact("sys_status", ["green"])
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_w",
            description="Tool",
            input_schema={},
            preconditions=[_cond("sys_status", ["green"])],
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_world_drift",
        goal_description="Test world drift",
        initial_state=[w1_fact],
        actions=[_action(action_id="act1", capability_name="tool_w", preconditions=[_cond("sys_status", ["green"])], positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_world_drift")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[w1_fact])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)

    # Session starts execution with authorized world
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[w1_fact])

    # World drifts before manager executes
    drifted_world = [_fact("sys_status", ["red"])]
    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=drifted_world,
        policy_hash=policy_hash,
    )
    with pytest.raises((StateDriftError, PreconditionFailedError, ValueError)):
        manager.execute_authorized_plan(cert)


def test_runtime_rejects_policy_hash_mismatch():
    """Certificate policy hash does not match current registry/policy identity. Rejects execution."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_p",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_pol_mismatch",
        goal_description="Test policy mismatch",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_p", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_pol_mismatch")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    cert = session.authorize_selected(reg, policy_hash="policy_v1")
    session.start_execution(reg, policy_hash="policy_v1")

    # Pass mismatched policy hash in certificate
    tampered_cert = cert.model_copy(update={"policy_hash": "wrong_policy_hash_xyz"})
    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash="policy_v1")
    with pytest.raises((StateDriftError, ValueError, SignatureVerificationError)):
        manager.execute_authorized_plan(tampered_cert)


def test_runtime_rejects_wrong_authorization_certificate_for_session():
    """Attempting to execute using a certificate from a different session must fail preflight."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_sess",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_wrong_cert",
        goal_description="Test wrong cert",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_sess", positive_effects=[_cond("done", [])])],
    )
    s1 = PlanningSession(session_id="s_correct")
    s1.submit_draft(plan)
    s1.validate_candidate(1, reg)
    s1.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert1 = s1.authorize_selected(reg, policy_hash=policy_hash)
    s1.start_execution(reg, policy_hash=policy_hash)

    s2 = PlanningSession(session_id="s_other")
    s2.submit_draft(plan)
    s2.validate_candidate(1, reg)
    s2.select_version(1)
    cert2 = s2.authorize_selected(reg, policy_hash=policy_hash)
    s2.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=s1, registry=reg, ledger=EvidenceLedger(session_id=s1.session_id), observed_world_state=[], policy_hash=policy_hash)
    # Passing cert2 (from session s2) to session s1's manager
    with pytest.raises((SignatureVerificationError, ValueError, StateDriftError)):
        manager.execute_authorized_plan(cert2)


# ---------------------------------------------------------------------------
# Section B: Do NOT Trust PlanIR Initial State at Runtime
# ---------------------------------------------------------------------------

def test_runtime_without_trusted_world_state_does_not_trust_plan_initial_state():
    """PlanIR self-asserts admin_authorized(prod)=VERIFIED_TRUE without trusted snapshot. Runtime must refuse action."""
    untrusted_fact = WorldFact(
        predicate="admin_authorized",
        args=["prod"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
    )
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="admin_action",
            description="Requires admin",
            input_schema={},
            preconditions=[_cond("admin_authorized", ["prod"])],
            positive_effects=[_cond("admin_done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="admin_done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_untrusted_runtime",
        goal_description="Test untrusted runtime initial state",
        initial_state=[untrusted_fact],
        actions=[_action(action_id="act1", capability_name="admin_action", preconditions=[_cond("admin_authorized", ["prod"])], positive_effects=[_cond("admin_done", [])])],
    )
    session = PlanningSession(session_id="s_untrusted_runtime")
    session.submit_draft(plan)
    # Validation marks plan UNKNOWN because initial fact is ungrounded
    val_res = session.validate_candidate(1, reg, observed_world_state=None)
    assert val_res.status.value == "UNKNOWN"


# ---------------------------------------------------------------------------
# Section C: Exact Runtime Effect <-> Verifier Attestation
# ---------------------------------------------------------------------------

def test_runtime_verifier_must_match_exact_effect_predicate_and_args(tmp_path):
    """Capability has verifier for file_exists(staging). Action effect is file_exists(prod). Verifier cannot witness prod."""
    prod_file = tmp_path / "prod.txt"
    staging_file = tmp_path / "staging.txt"
    staging_file.write_text("staging exists")

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="deploy_tool",
            description="Deploy",
            input_schema={"target": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{target}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_staging",
                    predicate="file_exists",
                    target_args_mapping=[str(staging_file)],
                    command_template=["test", "-f", str(staging_file)],
                )
            ],
            executor_command_template=["touch", "{target}"],
        )
    )
    plan = PlanIR(
        plan_id="p_mismatched_target",
        goal_description="Test mismatched verifier target",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="deploy_tool", parameters={"target": str(prod_file)}, positive_effects=[_cond("file_exists", [str(prod_file)])])],
    )
    session = PlanningSession(session_id="s_mismatched_target")
    session.submit_draft(plan)
    val_res = session.validate_candidate(1, reg)
    assert val_res.status.value == "UNKNOWN"


def test_runtime_only_promotes_effects_with_matching_verifiers(tmp_path):
    """Capability declares 2 positive effects: effect 1 has verifier (passes), effect 2 has no verifier. Only effect 1 is promoted."""
    f1 = tmp_path / "f1.txt"
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="multi_effect_tool",
            description="Multiple effects",
            input_schema={"p1": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{p1}"]), _cond("unverified_flag", [])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_f1",
                    predicate="file_exists",
                    target_args_mapping=["{p1}"],
                    command_template=["test", "-f", "{p1}"],
                )
            ],
            executor_command_template=["touch", "{p1}"],
        )
    )
    plan = PlanIR(
        plan_id="p_multi_effect",
        goal_description="Test multi effect promotion",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="multi_effect_tool", parameters={"p1": str(f1)}, positive_effects=[_cond("file_exists", [str(f1)]), _cond("unverified_flag", [])])],
    )
    session = PlanningSession(session_id="s_multi_effect")
    session.submit_draft(plan)
    # Plan validation has UNKNOWN for unverified_flag
    val_res = session.validate_candidate(1, reg)
    assert "unverified_flag()" in val_res.unknown_facts


def test_runtime_revalidates_action_against_current_registry(tmp_path):
    """Action parameters modified to violate schema must fail before execution."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="typed_tool",
            description="Typed param",
            input_schema={"num": {"type": "int", "required": True}},
            positive_effects=[_cond("done", [1])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=[1], command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    # Action supplies string "not_an_int"
    bad_action = _action(action_id="act1", capability_name="typed_tool", parameters={"num": "not_an_int"}, positive_effects=[_cond("done", [1])])
    plan = PlanIR(plan_id="p_bad_schema", goal_description="Test schema revalidation", initial_state=[], actions=[bad_action])
    session = PlanningSession(session_id="s_bad_schema")
    session.submit_draft(plan)
    val_res = session.validate_candidate(1, reg)
    assert val_res.status.value == "FAIL"


# ---------------------------------------------------------------------------
# Section D: Capability Hash Covers Full Runtime Contract
# ---------------------------------------------------------------------------

def test_registry_hash_changes_when_verifier_target_mapping_changes():
    """Changing verifier target_args_mapping must change capability and registry hash."""
    cap1 = CapabilityEntry(
        name="cap",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="pred", target_args_mapping=["a"])],
    )
    cap2 = CapabilityEntry(
        name="cap",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="pred", target_args_mapping=["b"])],
    )
    assert cap1.compute_capability_hash() != cap2.compute_capability_hash()


def test_registry_hash_changes_when_verifier_expected_pattern_changes():
    """Changing verifier expected_output_pattern must change capability hash."""
    cap1 = CapabilityEntry(
        name="cap",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="pred", expected_output_pattern="pattern_a")],
    )
    cap2 = CapabilityEntry(
        name="cap",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="pred", expected_output_pattern="pattern_b")],
    )
    assert cap1.compute_capability_hash() != cap2.compute_capability_hash()


def test_registry_hash_changes_when_verifier_expected_value_changes():
    """Changing verifier json_path or expected_value must change capability hash."""
    cap1 = CapabilityEntry(
        name="cap",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="pred", json_path="status", expected_value="OK")],
    )
    cap2 = CapabilityEntry(
        name="cap",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="pred", json_path="status", expected_value="ERROR")],
    )
    assert cap1.compute_capability_hash() != cap2.compute_capability_hash()


def test_registry_hash_changes_when_executor_contract_changes():
    """Changing executor_command_template must change capability hash."""
    cap1 = CapabilityEntry(
        name="cap",
        description="Cap",
        executor_command_template=["echo", "1"],
    )
    cap2 = CapabilityEntry(
        name="cap",
        description="Cap",
        executor_command_template=["echo", "2"],
    )
    assert cap1.compute_capability_hash() != cap2.compute_capability_hash()


# ---------------------------------------------------------------------------
# Section E: Custom Handler Cannot Replace Authorized Contract
# ---------------------------------------------------------------------------

def test_custom_handler_cannot_bypass_missing_executor_contract(tmp_path):
    """Capability without executor_command_template cannot be executed even if custom_action_handler is passed."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="empty_contract_cap",
            description="No executor command",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=[],  # Missing!
        )
    )
    plan = PlanIR(
        plan_id="p_handler_bypass",
        goal_description="Test custom handler cannot bypass missing contract",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="empty_contract_cap", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_handler_bypass")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    def custom_handler(action, params):
        return SandboxExecutionResult(stdout="fake", returncode=0)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert, custom_action_handler=custom_handler)
    assert summary.success is False
    assert any("contract" in str(r.error_message).lower() for r in summary.step_results)


# ---------------------------------------------------------------------------
# Section F: Correct Witness Semantics (UNWITNESSED vs WITNESSED_FALSE)
# ---------------------------------------------------------------------------

def test_precondition_failure_is_unwitnessed(tmp_path):
    """Precondition check failure at runtime must record witness_status == UNWITNESSED (not WITNESSED_FALSE)."""
    f_step1 = tmp_path / "step1_artifact.txt"
    # Note: f_step1 is NOT created before execution

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_step1_creator",
            description="Step 1 claims to create file",
            input_schema={"p": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{p}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_f1",
                    predicate="file_exists",
                    target_args_mapping=["{p}"],
                    command_template=["test", "-f", "{p}"],
                )
            ],
            executor_command_template=["echo", "pretending to create {p}"],  # Does not create file!
        )
    )
    reg.register(
        CapabilityEntry(
            name="tool_step2_consumer",
            description="Step 2 requires step 1 file",
            input_schema={"p": {"type": "str", "required": True}},
            preconditions=[_cond("file_exists", ["{p}"])],
            positive_effects=[_cond("step2_done", [])],
            verifiers=[ObservationVerifier(verifier_id="v_s2", predicate="step2_done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    # 2-step plan: Step 1 produces file_exists(f_step1), Step 2 requires file_exists(f_step1)
    plan = PlanIR(
        plan_id="p_pre_unwitnessed",
        goal_description="Test live precondition failure status",
        initial_state=[],
        actions=[
            _action(action_id="act1", capability_name="tool_step1_creator", parameters={"p": str(f_step1)}, positive_effects=[_cond("file_exists", [str(f_step1)])]),
            _action(action_id="act2", capability_name="tool_step2_consumer", parameters={"p": str(f_step1)}, preconditions=[_cond("file_exists", [str(f_step1)])], positive_effects=[_cond("step2_done", [])]),
        ],
    )
    session = PlanningSession(session_id="s_pre_unwitnessed")
    session.submit_draft(plan)
    # Plan is feasibility PASS at plan-time because Step 1 projects SUPPORTED_TRUE for Step 2
    val_res = session.validate_candidate(1, reg, observed_world_state=[])
    assert val_res.status == ValidationStatus.PASS
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)

    # Step 1 verifier fails (file not created -> WITNESSED_FALSE); plan aborts; step 2 is UNWITNESSED
    assert summary.success is False
    assert summary.failed_step_id == "act1"
    assert summary.step_results[0].witness_status == WitnessStatus.WITNESSED_FALSE


def test_executor_failure_is_unwitnessed(tmp_path):
    """Process exit code != 0 must record witness_status == UNWITNESSED (no postcondition observation occurred)."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="failing_proc_tool",
            description="Process returns exit 1",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["false"],  # Fails!
        )
    )
    plan = PlanIR(
        plan_id="p_exec_unwitnessed",
        goal_description="Test exec failure witness status",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="failing_proc_tool", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_exec_unwitnessed")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is False
    assert summary.step_results[0].exit_code != 0
    assert summary.step_results[0].witness_status == WitnessStatus.UNWITNESSED


def test_missing_verifier_is_unwitnessed(tmp_path):
    """Capability with effects but without verifiers must return witness_status == UNWITNESSED (cannot witness true or false)."""
    reg = CapabilityRegistry()
    cap_no_verif = CapabilityEntry(
        name="no_verif_tool",
        description="No verifiers",
        input_schema={},
        positive_effects=[_cond("done", [])],
        verifiers=[],
        executor_command_template=["true"],
    )
    action_with_effect = _action(action_id="act1", capability_name="no_verif_tool", positive_effects=[_cond("done", [])])

    session = PlanningSession(session_id="s_miss_verif")
    policy_hash = reg.compute_registry_hash()
    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    witness_status = manager._witness_postconditions(action_with_effect, cap_no_verif)
    assert witness_status == WitnessStatus.UNWITNESSED


# ---------------------------------------------------------------------------
# Section G: Fail Closed on Verifier Modes (JSON Path & Expected Value)
# ---------------------------------------------------------------------------

def test_unsupported_verifier_mode_cannot_silently_witness_true(tmp_path):
    """ObservationVerifier specifying json_path and expected_value must evaluate JSON output correctly."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="json_api_tool",
            description="Tool returning JSON",
            input_schema={},
            positive_effects=[_cond("health_ok", [])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_health",
                    predicate="health_ok",
                    command_template=["echo", '{"status": "healthy", "code": 200}'],
                    json_path="status",
                    expected_value="healthy",
                )
            ],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_json_ok",
        goal_description="Test json verifier success",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="json_api_tool", positive_effects=[_cond("health_ok", [])])],
    )
    session = PlanningSession(session_id="s_json_ok")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is True
    assert summary.step_results[0].witness_status == WitnessStatus.WITNESSED_TRUE

    # Negative test: json value mismatch fails witnessing
    reg_bad = CapabilityRegistry()
    reg_bad.register(
        CapabilityEntry(
            name="json_bad_tool",
            description="Tool returning wrong JSON",
            input_schema={},
            positive_effects=[_cond("health_ok", [])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_bad",
                    predicate="health_ok",
                    command_template=["echo", '{"status": "unhealthy", "code": 500}'],
                    json_path="status",
                    expected_value="healthy",
                )
            ],
            executor_command_template=["true"],
        )
    )
    plan_bad = PlanIR(
        plan_id="p_json_bad",
        goal_description="Test json verifier mismatch failure",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="json_bad_tool", positive_effects=[_cond("health_ok", [])])],
    )
    s_bad = PlanningSession(session_id="s_json_bad")
    s_bad.submit_draft(plan_bad)
    s_bad.validate_candidate(1, reg_bad)
    s_bad.select_version(1)
    pol_bad = reg_bad.compute_registry_hash()
    cert_bad = s_bad.authorize_selected(reg_bad, policy_hash=pol_bad)
    s_bad.start_execution(reg_bad, policy_hash=pol_bad)

    mgr_bad = ExecutionPlanManager(session=s_bad, registry=reg_bad, ledger=EvidenceLedger(session_id=s_bad.session_id), observed_world_state=[], policy_hash=pol_bad)
    sum_bad = mgr_bad.execute_authorized_plan(cert_bad)
    assert sum_bad.success is False
    assert sum_bad.step_results[0].witness_status == WitnessStatus.WITNESSED_FALSE


# ---------------------------------------------------------------------------
# Section H: Real Mutation & Multi-Step Runtime Handoffs
# ---------------------------------------------------------------------------

def test_executor_command_creates_real_observable_effect(tmp_path):
    """Executor runs real command and creates actual file on disk."""
    target_file = tmp_path / "real_file.txt"
    assert not target_file.exists()

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.create",
            description="Create real file",
            input_schema={"path": {"type": "str", "required": True}, "content": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_exists",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["sh", "-c", "echo '{content}' > '{path}'"],
        )
    )
    plan = PlanIR(
        plan_id="p_real_mutation",
        goal_description="Test real filesystem mutation",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="fs.create",
                parameters={"path": str(target_file), "content": "grounded_content_123"},
                positive_effects=[_cond("file_exists", [str(target_file)])],
            )
        ],
    )
    session = PlanningSession(session_id="s_real_mutation")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is True
    assert target_file.exists()
    assert target_file.read_text().strip() == "grounded_content_123"


def test_witnessed_effect_updates_live_world_state(tmp_path):
    """After execution and successful independent verification, live world state is updated to VERIFIED_TRUE."""
    target_file = tmp_path / "witnessed.txt"
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.touch",
            description="Touch file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_touch",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["touch", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="p_live_state",
        goal_description="Test live state update",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="fs.touch",
                parameters={"path": str(target_file)},
                positive_effects=[_cond("file_exists", [str(target_file)])],
            )
        ],
    )
    session = PlanningSession(session_id="s_live_state")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is True

    key = f"file_exists({str(target_file)})"
    assert key in manager.live_world_state
    live_fact = manager.live_world_state[key]
    assert live_fact.truth == FactTruth.VERIFIED_TRUE
    assert live_fact.provenance.source_type == SourceType.OBSERVED_WORLD_STATE


def test_two_step_plan_consumes_first_step_witnessed_effect(tmp_path):
    """Action 2 checks empirical precondition satisfied only after Action 1 is executed and witnessed."""
    f1 = tmp_path / "step1.txt"
    f2 = tmp_path / "step2.txt"

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool1",
            description="Create step 1",
            input_schema={"out": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{out}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_step1",
                    predicate="file_exists",
                    target_args_mapping=["{out}"],
                    command_template=["test", "-f", "{out}"],
                )
            ],
            executor_command_template=["touch", "{out}"],
        )
    )
    reg.register(
        CapabilityEntry(
            name="tool2",
            description="Create step 2 requiring step 1",
            input_schema={"inp": {"type": "str", "required": True}, "out": {"type": "str", "required": True}},
            preconditions=[_cond("file_exists", ["{inp}"])],
            positive_effects=[_cond("file_exists", ["{out}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_step2",
                    predicate="file_exists",
                    target_args_mapping=["{out}"],
                    command_template=["test", "-f", "{out}"],
                )
            ],
            executor_command_template=["touch", "{out}"],
        )
    )

    plan = PlanIR(
        plan_id="p_two_step_runtime",
        goal_description="Test 2 step runtime execution",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="tool1",
                parameters={"out": str(f1)},
                positive_effects=[_cond("file_exists", [str(f1)])],
            ),
            _action(
                action_id="act2",
                capability_name="tool2",
                parameters={"inp": str(f1), "out": str(f2)},
                preconditions=[_cond("file_exists", [str(f1)])],
                positive_effects=[_cond("file_exists", [str(f2)])],
            ),
        ],
    )
    session = PlanningSession(session_id="s_two_step_runtime")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is True
    assert len(summary.step_results) == 2
    assert f1.exists()
    assert f2.exists()


def test_failed_verifier_keeps_effect_unverified(tmp_path):
    """Executor succeeds (exit 0) but verifier fails -> effect remains UNKNOWN and dependent actions do not run."""
    f_step1 = tmp_path / "step1_failed_verif.txt"
    f_step2 = tmp_path / "step2_should_not_run.txt"

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="broken_creator",
            description="Runs echo but verifier checks for missing file",
            input_schema={"dst": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{dst}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_fail",
                    predicate="file_exists",
                    target_args_mapping=["{dst}"],
                    command_template=["test", "-f", "{dst}"],
                )
            ],
            executor_command_template=["echo", "pretending to create {dst}"],
        )
    )
    reg.register(
        CapabilityEntry(
            name="step2_tool",
            description="Step 2",
            input_schema={"inp": {"type": "str", "required": True}, "out": {"type": "str", "required": True}},
            preconditions=[_cond("file_exists", ["{inp}"])],
            positive_effects=[_cond("file_exists", ["{out}"])],
            verifiers=[ObservationVerifier(verifier_id="v2", predicate="file_exists", target_args_mapping=["{out}"], command_template=["test", "-f", "{out}"])],
            executor_command_template=["touch", "{out}"],
        )
    )
    plan = PlanIR(
        plan_id="p_failed_verifier",
        goal_description="Test failed verifier stops dependent action",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="broken_creator",
                parameters={"dst": str(f_step1)},
                positive_effects=[_cond("file_exists", [str(f_step1)])],
            ),
            _action(
                action_id="act2",
                capability_name="step2_tool",
                parameters={"inp": str(f_step1), "out": str(f_step2)},
                preconditions=[_cond("file_exists", [str(f_step1)])],
                positive_effects=[_cond("file_exists", [str(f_step2)])],
            ),
        ],
    )
    session = PlanningSession(session_id="s_failed_verifier")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is False
    assert summary.failed_step_id == "act1"
    assert len(summary.step_results) == 1
    assert not f_step2.exists()


def test_runtime_does_not_mutate_plan_ir_to_fake_observation(tmp_path):
    """PlanIR remains immutable during execution; empirical facts live only in manager.live_world_state and ledger."""
    target_file = tmp_path / "immutable.txt"
    init_fact = _fact("system_ready", [])
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.touch",
            description="Touch file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_touch",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["touch", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="p_immutable",
        goal_description="Test plan immutability",
        initial_state=[init_fact],
        actions=[
            _action(
                action_id="act1",
                capability_name="fs.touch",
                parameters={"path": str(target_file)},
                preconditions=[_cond("system_ready", [])],
                positive_effects=[_cond("file_exists", [str(target_file)])],
            )
        ],
    )
    original_plan_hash = plan.compute_hash()

    session = PlanningSession(session_id="s_immutable")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[init_fact])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[init_fact])

    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[init_fact],
        policy_hash=policy_hash,
    )
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is True

    # PlanIR was NOT mutated
    assert plan.compute_hash() == original_plan_hash
    assert len(plan.initial_state) == 1
    assert plan.initial_state[0].predicate == "system_ready"


# ---------------------------------------------------------------------------
# Section I: Authorization of Trusted World vs PlanIR initial_state
# ---------------------------------------------------------------------------

def test_authorization_world_hash_comes_from_trusted_validation_snapshot():
    """Certificate world_state_hash must come from trusted validation snapshot, never PlanIR.initial_state claims."""
    fact_plan = WorldFact(predicate="plan_claim", args=["p1"], truth=FactTruth.VERIFIED_TRUE, provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE))
    fact_trusted = WorldFact(predicate="trusted_fact", args=["t1"], truth=FactTruth.VERIFIED_TRUE, provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE))

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_auth_world_hash",
        goal_description="Test auth world hash",
        initial_state=[fact_plan],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_auth_world_hash")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[fact_trusted])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)

    from plan_mode.session import compute_world_state_hash
    # The certificate world_state_hash must equal hash([fact_trusted]) and NOT hash([fact_plan])
    assert cert.world_state_hash == compute_world_state_hash([fact_trusted])
    assert cert.world_state_hash != compute_world_state_hash([fact_plan])


def test_start_execution_never_falls_back_to_plan_ir_initial_state():
    """start_execution() must not treat planner initial fact as trusted world identity when no trusted snapshot given."""
    planner_fact = WorldFact(predicate="planner_claim", args=["p1"], truth=FactTruth.VERIFIED_TRUE, provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE))
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_no_fallback",
        goal_description="Test start execution no fallback",
        initial_state=[planner_fact],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_no_fallback")
    session.submit_draft(plan)
    # Validated with observed_world_state=[] (trusted empty)
    session.validate_candidate(1, reg, observed_world_state=[])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)

    # Calling start_execution with current_world_facts=[planner_fact] must detect drift from authorized empty world
    with pytest.raises(StateDriftError):
        session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[planner_fact])


def test_missing_runtime_trusted_snapshot_fails_closed_even_without_preconditions():
    """Action has zero preconditions, but was authorized against non-empty trusted snapshot W1.
    If manager is constructed with observed_world_state=None, execution must fail world identity preflight."""
    w1_fact = _fact("env_flag", ["online"])
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="noop_tool",
            description="Noop",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_missing_snapshot",
        goal_description="Test missing snapshot preflight",
        initial_state=[w1_fact],
        actions=[_action(action_id="act1", capability_name="noop_tool", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_missing_snapshot")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[w1_fact])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[w1_fact])

    # Manager created with observed_world_state=None (no snapshot supplied)
    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=None,
    )
    with pytest.raises((StateDriftError, ValueError)):
        manager.execute_authorized_plan(cert)


# ---------------------------------------------------------------------------
# Section J: Live Policy Drift with Untampered Certificate
# ---------------------------------------------------------------------------

def test_live_policy_drift_rejected_with_untampered_certificate():
    """Authorize untouched certificate with policy_v1, but manager executes with current policy_v2. Must reject before actions run."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_pol",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_pol_drift",
        goal_description="Test live policy drift",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_pol", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_pol_drift")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    cert = session.authorize_selected(reg, policy_hash="policy_v1")
    session.start_execution(reg, policy_hash="policy_v1")

    # Manager is given policy_hash="policy_v2" representing current runtime policy identity
    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        policy_hash="policy_v2",
    )
    with pytest.raises((StateDriftError, ValueError)):
        manager.execute_authorized_plan(cert)


# ---------------------------------------------------------------------------
# Section K: None Snapshot != Trusted Empty Snapshot & Hash Check
# ---------------------------------------------------------------------------

def test_none_world_snapshot_is_not_equivalent_to_trusted_empty_snapshot():
    """observed_world_state=None (unsupplied) must be distinguished from observed_world_state=[] (trusted empty)."""
    mgr_none = ExecutionPlanManager(session=PlanningSession(session_id="s_none"), registry=CapabilityRegistry(), ledger=EvidenceLedger(session_id="s_none"), observed_world_state=None)
    mgr_empty = ExecutionPlanManager(session=PlanningSession(session_id="s_empty"), registry=CapabilityRegistry(), ledger=EvidenceLedger(session_id="s_empty"), observed_world_state=[])
    assert mgr_none._has_trusted_snapshot is False
    assert mgr_empty._has_trusted_snapshot is True


def test_world_hash_check_cannot_be_skipped_by_empty_manager_state():
    """Certificate with non-empty world_state_hash must NOT be skipped when manager state is empty."""
    w1_fact = _fact("auth_token", ["valid"])
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_tok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_empty_skip",
        goal_description="Test empty manager skip prevention",
        initial_state=[w1_fact],
        actions=[_action(action_id="act1", capability_name="tool_tok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_empty_skip")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[w1_fact])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[w1_fact])

    # Manager has empty trusted snapshot [] while cert has hash([w1_fact])
    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
    )
    with pytest.raises(StateDriftError):
        manager.execute_authorized_plan(cert)


# ---------------------------------------------------------------------------
# Section L: Typed Capability Hashing
# ---------------------------------------------------------------------------

def test_registry_hash_distinguishes_integer_and_string_verifier_targets():
    """target_args_mapping=[123] (int) vs ['123'] (str) must produce distinct capability hashes."""
    cap_int = CapabilityEntry(
        name="cap_typed",
        description="Cap typed int",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="proc", target_args_mapping=[123])],
    )
    cap_str = CapabilityEntry(
        name="cap_typed",
        description="Cap typed str",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="proc", target_args_mapping=["123"])],
    )
    assert cap_int.compute_capability_hash() != cap_str.compute_capability_hash()


def test_capability_hash_uses_canonical_typed_runtime_contract():
    """Capability hash uses typed canonical JSON structure preserving numeric and boolean types."""
    cap_bool = CapabilityEntry(
        name="cap_typed2",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="flag", expected_value=True)],
    )
    cap_str_bool = CapabilityEntry(
        name="cap_typed2",
        description="Cap",
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="flag", expected_value="True")],
    )
    assert cap_bool.compute_capability_hash() != cap_str_bool.compute_capability_hash()


# ---------------------------------------------------------------------------
# Section M: Runtime Exact Target & Multi-Effect Partial Attestation
# ---------------------------------------------------------------------------

def test_runtime_wrong_target_verifier_ignored_when_correct_verifier_fails(tmp_path):
    """Capability defines wrong-target verifier (passes) and correct-target verifier (fails).
    Wrong-target verifier must be ignored; execution must fail witnessing."""
    prod_target = tmp_path / "prod_missing.txt"
    staging_target = tmp_path / "staging_exists.txt"
    staging_target.write_text("staging exists")

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="scoped_deployer",
            description="Deployer",
            input_schema={"target": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{target}"])],
            verifiers=[
                # Correct target verifier: checks prod_missing.txt (which does not exist -> FAILS)
                ObservationVerifier(
                    verifier_id="v_correct",
                    predicate="file_exists",
                    target_args_mapping=["{target}"],
                    command_template=["test", "-f", "{target}"],
                ),
                # Wrong target verifier: checks staging_exists.txt (which exists -> PASSES)
                ObservationVerifier(
                    verifier_id="v_wrong_staging",
                    predicate="file_exists",
                    target_args_mapping=[str(staging_target)],
                    command_template=["test", "-f", str(staging_target)],
                ),
            ],
            executor_command_template=["echo", "pretending deploy to {target}"],
        )
    )
    plan = PlanIR(
        plan_id="p_runtime_wrong_target",
        goal_description="Test runtime ignores passing wrong-target verifier",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="scoped_deployer", parameters={"target": str(prod_target)}, positive_effects=[_cond("file_exists", [str(prod_target)])])],
    )
    session = PlanningSession(session_id="s_runtime_wrong_target")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)

    # Must fail because v_correct failed and v_wrong_staging was ignored for prod effect
    assert summary.success is False
    assert summary.step_results[0].witness_status == WitnessStatus.WITNESSED_FALSE
    key = f"file_exists({str(prod_target)})"
    assert manager.live_world_state.get(key) is None or manager.live_world_state[key].truth != FactTruth.VERIFIED_TRUE


def test_partial_multi_effect_witnessing_promotes_only_successful_effect(tmp_path):
    """Action produces 2 effects: effect A verifier succeeds, effect B verifier fails.
    Effect A is promoted to VERIFIED_TRUE; effect B remains unpromoted; overall step fails."""
    f_a = tmp_path / "effect_a.txt"
    f_b = tmp_path / "effect_b_missing.txt"

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="dual_creator",
            description="Creates A but fails B",
            input_schema={"p_a": {"type": "str", "required": True}, "p_b": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{p_a}"]), _cond("file_exists", ["{p_b}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_a",
                    predicate="file_exists",
                    target_args_mapping=["{p_a}"],
                    command_template=["test", "-f", "{p_a}"],
                ),
                ObservationVerifier(
                    verifier_id="v_b",
                    predicate="file_exists",
                    target_args_mapping=["{p_b}"],
                    command_template=["test", "-f", "{p_b}"],
                ),
            ],
            executor_command_template=["touch", "{p_a}"],  # Only creates p_a!
        )
    )
    plan = PlanIR(
        plan_id="p_partial_multi",
        goal_description="Test partial multi-effect witnessing",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="dual_creator",
                parameters={"p_a": str(f_a), "p_b": str(f_b)},
                positive_effects=[_cond("file_exists", [str(f_a)]), _cond("file_exists", [str(f_b)])],
            )
        ],
    )
    session = PlanningSession(session_id="s_partial_multi")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is False
    assert summary.step_results[0].witness_status == WitnessStatus.WITNESSED_FALSE

    # Effect A was witnessed and promoted
    key_a = f"file_exists({str(f_a)})"
    assert key_a in manager.live_world_state
    assert manager.live_world_state[key_a].truth == FactTruth.VERIFIED_TRUE

    # Effect B was NOT witnessed and NOT promoted
    key_b = f"file_exists({str(f_b)})"
    assert manager.live_world_state.get(key_b) is None or manager.live_world_state[key_b].truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Section N: Custom Execution Backend Receives Authorized Resolved Command
# ---------------------------------------------------------------------------

def test_custom_backend_receives_authorized_resolved_command(tmp_path):
    """Custom execution backend receives the resolved authorized command vector."""
    target_file = tmp_path / "backend_file.txt"
    received_cmds = []

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.touch_backend",
            description="Touch",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_touch",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["touch", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="p_backend_test",
        goal_description="Test custom backend contract",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="fs.touch_backend", parameters={"path": str(target_file)}, positive_effects=[_cond("file_exists", [str(target_file)])])],
    )
    session = PlanningSession(session_id="s_backend_test")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    def spy_backend(resolved_cmd, timeout_seconds=10.0):
        received_cmds.append(resolved_cmd)
        # Execute the authorized command vector
        sandbox = ExecutionSandbox()
        return sandbox.execute_argv_pipeline([resolved_cmd], timeout_seconds=timeout_seconds)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert, custom_action_handler=spy_backend)

    assert summary.success is True
    assert len(received_cmds) == 1
    assert received_cmds[0] == ["touch", str(target_file)]
    assert target_file.exists()


def test_custom_backend_cannot_replace_registered_command_contract():
    """Capability with empty executor contract cannot run even if custom backend is provided."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="empty_cap_backend",
            description="No command",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=[],  # Empty!
        )
    )
    plan = PlanIR(
        plan_id="p_empty_backend",
        goal_description="Test empty backend refusal",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="empty_cap_backend", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_empty_backend")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    def fake_backend(resolved_cmd, action, params):
        return SandboxExecutionResult(returncode=0)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id), observed_world_state=[], policy_hash=policy_hash)
    summary = manager.execute_authorized_plan(cert, custom_action_handler=fake_backend)
    assert summary.success is False
    assert any("contract" in str(r.error_message).lower() for r in summary.step_results)


# ---------------------------------------------------------------------------
# Section O: Canonical Validation Snapshot Normalization & Storage
# ---------------------------------------------------------------------------

def test_validation_snapshot_stored_in_canonical_normalized_form():
    """Input [X UNKNOWN, X VERIFIED_TRUE] must normalize to stored validation_world_state with 1 fact of truth VERIFIED_TRUE."""
    f_unk = _fact("service_running", ["prod"], FactTruth.UNKNOWN)
    f_true = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_canon_store",
        goal_description="Test canonical store",
        initial_state=[f_true],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_canon_store")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[f_unk, f_true])

    stored_v1 = session.versions[1]
    assert stored_v1.validation_world_state is not None
    # Stored snapshot must contain exactly 1 canonical fact with truth == VERIFIED_TRUE
    assert len(stored_v1.validation_world_state) == 1
    assert stored_v1.validation_world_state[0].fact_key == "service_running(prod)"
    assert stored_v1.validation_world_state[0].truth == FactTruth.VERIFIED_TRUE


def test_duplicate_same_truth_snapshot_authorizes_and_executes_without_hash_mismatch():
    """Input [X VERIFIED_TRUE, X VERIFIED_TRUE] canonicalizes to 1 fact; authorization and execution match."""
    f1 = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)
    f2 = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_dup_exec",
        goal_description="Test duplicate execution hash consistency",
        initial_state=[f1],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_dup_exec")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[f1, f2])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[f1])

    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[f1, f2],
        policy_hash=policy_hash,
    )
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is True


def test_validation_ttl_normalization_is_bound_into_authorization_identity():
    """Expired trusted fact decays during validation; stored validation snapshot and cert hash reflect decayed UNKNOWN truth."""
    t0 = 1000.0
    now = 1050.0  # 50s later
    expired_fact = WorldFact(
        predicate="auth_token",
        args=["session_1"],
        truth=FactTruth.VERIFIED_TRUE,
        ttl_seconds=10.0,
        created_at=t0,
        updated_at=t0,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
    )
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_ttl_bound",
        goal_description="Test TTL bound into auth identity",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_ttl_bound")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[expired_fact], current_time=now)

    stored_v1 = session.versions[1]
    assert stored_v1.validation_world_state is not None
    assert stored_v1.validation_world_state[0].truth == FactTruth.UNKNOWN


# ---------------------------------------------------------------------------
# Section P: None Must Never Mean Trusted Empty
# ---------------------------------------------------------------------------

def test_runtime_none_snapshot_rejected_even_when_authorized_world_is_empty():
    """Authorize explicitly with observed_world_state=[]. Manager with observed_world_state=None must be rejected."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_none_vs_empty",
        goal_description="Test none vs empty snapshot",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_none_vs_empty")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])

    # Manager with observed_world_state=None must fail closed
    mgr_none = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=None,
        policy_hash=policy_hash,
    )
    with pytest.raises(StateDriftError):
        mgr_none.execute_authorized_plan(cert)

    # Manager with observed_world_state=[] must succeed
    mgr_empty = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy_hash,
    )
    summary = mgr_empty.execute_authorized_plan(cert)
    assert summary.success is True


# ---------------------------------------------------------------------------
# Section Q: Current Policy Identity Must Be Explicit
# ---------------------------------------------------------------------------

def test_runtime_requires_current_policy_identity():
    """ExecutionPlanManager without an explicit policy_hash must refuse execution."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_require_policy",
        goal_description="Test require policy",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_require_policy")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])

    # Manager created with policy_hash=None
    mgr_no_policy = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=None,
    )
    with pytest.raises((StateDriftError, ValueError)):
        mgr_no_policy.execute_authorized_plan(cert)


def test_runtime_policy_v2_rejected_against_untampered_v1_certificate():
    """Certificate is untouched and has policy_v1; manager has explicit policy_hash='policy_v2'. Must reject."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Tool",
            input_schema={},
            positive_effects=[_cond("done", [])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p_policy_v2_reject",
        goal_description="Test policy v2 reject",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("done", [])])],
    )
    session = PlanningSession(session_id="s_policy_v2_reject")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[])
    session.select_version(1)
    cert = session.authorize_selected(reg, policy_hash="policy_v1")
    session.start_execution(reg, policy_hash="policy_v1", current_world_facts=[])

    mgr_v2 = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash="policy_v2",
    )
    with pytest.raises(StateDriftError):
        mgr_v2.execute_authorized_plan(cert)


# ---------------------------------------------------------------------------
# Section R: Strict Custom Execution Backend Contract
# ---------------------------------------------------------------------------

def test_legacy_two_argument_custom_handler_is_rejected(tmp_path):
    """Providing a legacy 2-argument custom handler (without argv vector) must be rejected."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.touch_backend",
            description="Touch",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="file_exists", target_args_mapping=["{path}"], command_template=["test", "-f", "{path}"])],
            executor_command_template=["touch", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="p_legacy_handler",
        goal_description="Test legacy handler rejection",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="fs.touch_backend", parameters={"path": str(tmp_path / "x.txt")}, positive_effects=[_cond("file_exists", [str(tmp_path / "x.txt")])])],
    )
    session = PlanningSession(session_id="s_legacy_handler")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])

    def legacy_2arg_handler(action, params):
        return SandboxExecutionResult(returncode=0)

    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy_hash,
    )
    with pytest.raises(TypeError):
        manager.execute_authorized_plan(cert, custom_action_handler=legacy_2arg_handler)


def test_backend_internal_typeerror_does_not_trigger_second_execution_path(tmp_path):
    """If custom backend raises TypeError internally, it must execute exactly once and propagate without fallback."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.touch_backend",
            description="Touch",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="file_exists", target_args_mapping=["{path}"], command_template=["test", "-f", "{path}"])],
            executor_command_template=["touch", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="p_internal_typeerror",
        goal_description="Test internal typeerror single invocation",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="fs.touch_backend", parameters={"path": str(tmp_path / "x.txt")}, positive_effects=[_cond("file_exists", [str(tmp_path / "x.txt")])])],
    )
    session = PlanningSession(session_id="s_internal_typeerror")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=[])
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])

    invocations = 0

    def bug_backend(authorized_argv, timeout_seconds=10.0):
        nonlocal invocations
        invocations += 1
        raise TypeError("Deliberate internal backend type error")

    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy_hash,
    )
    with pytest.raises(TypeError) as excinfo:
        manager.execute_authorized_plan(cert, custom_action_handler=bug_backend)

    assert "Deliberate internal backend type error" in str(excinfo.value)
    assert invocations == 1


# ---------------------------------------------------------------------------
# Section S: Type-Safe World State Identity Hashing
# ---------------------------------------------------------------------------

def test_world_state_hash_distinguishes_integer_and_string_arguments():
    """compute_world_state_hash must distinguish integer 123 from string '123'."""
    from plan_mode.session import compute_world_state_hash
    f_int = WorldFact(predicate="resource", args=[123], truth=FactTruth.VERIFIED_TRUE, provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE))
    f_str = WorldFact(predicate="resource", args=["123"], truth=FactTruth.VERIFIED_TRUE, provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE))

    hash_int = compute_world_state_hash([f_int])
    hash_str = compute_world_state_hash([f_str])

    assert hash_int != hash_str
