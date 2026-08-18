"""Adversarial Runtime Semantics and Empirical Witnessing Tests (Phase 2)."""

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
# Test 1: Default executor cannot be a silent 'true' no-op
# ---------------------------------------------------------------------------
def test_default_executor_cannot_be_true_noop(tmp_path):
    """A capability without a concrete executor contract must not silently execute 'true' and fabricate completion."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="empty_executor_cap",
            description="Capability with no executor command template",
            input_schema={"val": {"type": "str", "required": True}},
            positive_effects=[_cond("done", ["{val}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=["{val}"])],
            executor_command_template=[],  # No concrete command!
        )
    )
    plan = PlanIR(
        plan_id="p1_no_noop",
        goal_description="Test no true noop",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="empty_executor_cap",
                parameters={"val": "a"},
                positive_effects=[_cond("done", ["a"])],
            )
        ],
    )
    session = PlanningSession(session_id="s1_no_noop")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    ledger = EvidenceLedger(session_id=session.session_id)
    manager = ExecutionPlanManager(session=session, registry=reg, ledger=ledger)

    # Must refuse execution or fail, NOT execute 'true' and return success
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is False
    assert any("contract" in str(r.error_message).lower() or "executor" in str(r.error_message).lower() for r in summary.step_results)


# ---------------------------------------------------------------------------
# Test 2: Capability without executor contract cannot execute
# ---------------------------------------------------------------------------
def test_capability_without_executor_contract_cannot_execute(tmp_path):
    """Planning semantics alone are insufficient for runtime execution; missing contract causes explicit refusal."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="abstract_cap",
            description="Abstract planning capability without execution binding",
            input_schema={},
            positive_effects=[_cond("state_ready", [])],
            verifiers=[ObservationVerifier(verifier_id="v_ready", predicate="state_ready")],
            executor_command_template=[],  # Empty
        )
    )
    plan = PlanIR(
        plan_id="p2_abstract",
        goal_description="Test abstract capability refusal",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="abstract_cap", positive_effects=[_cond("state_ready", [])])],
    )
    session = PlanningSession(session_id="s2_abstract")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is False
    assert summary.step_results[0].exit_code != 0


# ---------------------------------------------------------------------------
# Test 3: Capability without verifier cannot witness VERIFIED_TRUE
# ---------------------------------------------------------------------------
def test_capability_without_verifier_cannot_witness_true(tmp_path):
    """Successful process exit code 0 alone must never create VERIFIED_TRUE if capability lacks verifier."""
    out_file = tmp_path / "output.txt"
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="unverified_creator",
            description="Creates file but has zero verifiers",
            input_schema={"dst": {"type": "str", "required": True}},
            positive_effects=[_cond("file_created", ["{dst}"])],
            verifiers=[],  # Zero verifiers!
            executor_command_template=["touch", "{dst}"],
        )
    )
    plan = PlanIR(
        plan_id="p3_unverified",
        goal_description="Test unverified capability cannot witness true",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="unverified_creator",
                parameters={"dst": str(out_file)},
                positive_effects=[_cond("file_created", [str(out_file)])],
            )
        ],
    )
    session = PlanningSession(session_id="s3_unverified")
    session.submit_draft(plan)
    # Plan validation is UNKNOWN because missing verifier
    val_res = session.validate_candidate(1, reg)
    assert val_res.status.value == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 4: Executor command creates real observable filesystem effect
# ---------------------------------------------------------------------------
def test_executor_command_creates_real_observable_effect(tmp_path):
    """Executor runs the real command and creates actual file artifact on disk."""
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
        plan_id="p4_real_mutation",
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
    session = PlanningSession(session_id="s4_real_mutation")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is True
    # Real file was created on disk!
    assert target_file.exists()
    assert target_file.read_text().strip() == "grounded_content_123"


# ---------------------------------------------------------------------------
# Test 5: Witnessed effect updates live world state to VERIFIED_TRUE
# ---------------------------------------------------------------------------
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
        plan_id="p5_live_state",
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
    session = PlanningSession(session_id="s5_live_state")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is True

    # Check live world state in manager
    key = f"file_exists({str(target_file)})"
    assert key in manager.live_world_state
    live_fact = manager.live_world_state[key]
    assert live_fact.truth == FactTruth.VERIFIED_TRUE
    assert live_fact.provenance.source_type == SourceType.OBSERVED_WORLD_STATE


# ---------------------------------------------------------------------------
# Test 6: Two-step plan consumes first-step witnessed effect at runtime
# ---------------------------------------------------------------------------
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
        plan_id="p6_two_step_runtime",
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
    session = PlanningSession(session_id="s6_two_step_runtime")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is True
    assert len(summary.step_results) == 2
    assert f1.exists()
    assert f2.exists()


# ---------------------------------------------------------------------------
# Test 7: Projected truth alone cannot satisfy runtime precondition
# ---------------------------------------------------------------------------
def test_projected_truth_alone_cannot_satisfy_runtime_precondition(tmp_path):
    """At runtime, if an action precondition has projected_truth=SUPPORTED_TRUE but truth=UNKNOWN, execution aborts."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="dependent_tool",
            description="Tool requiring unobserved fact",
            input_schema={},
            preconditions=[_cond("unobserved_fact", ["x"])],
            positive_effects=[_cond("done", ["x"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=["x"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p7_projected_alone",
        goal_description="Test projected alone fails runtime precheck",
        initial_state=[],  # unobserved_fact is not in trusted initial state
        actions=[
            _action(
                action_id="act1",
                capability_name="dependent_tool",
                preconditions=[_cond("unobserved_fact", ["x"])],
                positive_effects=[_cond("done", ["x"])],
            )
        ],
    )
    session = PlanningSession(session_id="s7_projected_alone")
    session.submit_draft(plan)
    val_res = session.validate_candidate(1, reg)
    assert val_res.status.value == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 8: Custom handler cannot bypass observation attestation
# ---------------------------------------------------------------------------
def test_custom_handler_cannot_bypass_attestation(tmp_path):
    """Custom handler returns exit code 0, but if independent verifier fails, effect is NOT marked VERIFIED_TRUE."""
    missing_file = tmp_path / "never_created.txt"

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="lying_custom_tool",
            description="Claims to create file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_verify_file",
                    predicate="file_exists",
                    target_args_mapping=["{path}"],
                    command_template=["test", "-f", "{path}"],
                )
            ],
            executor_command_template=["sh", "-c", "echo 'no-op'"],
        )
    )
    plan = PlanIR(
        plan_id="p8_custom_bypass",
        goal_description="Test custom handler cannot fake verifier success",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="lying_custom_tool",
                parameters={"path": str(missing_file)},
                positive_effects=[_cond("file_exists", [str(missing_file)])],
            )
        ],
    )
    session = PlanningSession(session_id="s8_custom_bypass")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    # Custom handler returns exit_code=0 without creating file
    def lying_handler(action, params):
        return SandboxExecutionResult(stdout="fake success", returncode=0)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))
    summary = manager.execute_authorized_plan(cert, custom_action_handler=lying_handler)

    assert summary.success is False
    assert summary.step_results[0].witness_status == WitnessStatus.WITNESSED_FALSE
    # Live state must NOT have VERIFIED_TRUE
    key = f"file_exists({str(missing_file)})"
    assert manager.live_world_state.get(key) is None or manager.live_world_state[key].truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 9: Failed verifier keeps effect unverified and aborts plan
# ---------------------------------------------------------------------------
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
        plan_id="p9_failed_verifier",
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
    session = PlanningSession(session_id="s9_failed_verifier")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash)

    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))
    summary = manager.execute_authorized_plan(cert)

    assert summary.success is False
    assert summary.failed_step_id == "act1"
    # Action 2 must never have executed
    assert len(summary.step_results) == 1
    assert not f_step2.exists()


# ---------------------------------------------------------------------------
# Test 10: Verifier observes exact bound target
# ---------------------------------------------------------------------------
def test_verifier_observes_exact_bound_target(tmp_path):
    """Verifier with wrong target argument binding fails to witness effect."""
    target_a = tmp_path / "file_a.txt"
    target_b = tmp_path / "file_b.txt"
    target_b.write_text("exists")

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="mismatched_verifier_tool",
            description="Creates target A but verifier checks target B",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[
                # Verifier is bound to target_b, not target_a
                ObservationVerifier(
                    verifier_id="v_mismatch",
                    predicate="file_exists",
                    target_args_mapping=[str(target_b)],
                    command_template=["test", "-f", str(target_b)],
                )
            ],
            executor_command_template=["touch", "{path}"],
        )
    )
    plan = PlanIR(
        plan_id="p10_target_match",
        goal_description="Test verifier bound target matching",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="mismatched_verifier_tool",
                parameters={"path": str(target_a)},
                positive_effects=[_cond("file_exists", [str(target_a)])],
            )
        ],
    )
    session = PlanningSession(session_id="s10_target_match")
    session.submit_draft(plan)
    # Plan validation should mark effect as UNWITNESSABLE because target_args_mapping does not match
    val_res = session.validate_candidate(1, reg)
    assert val_res.status.value == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 11: Execution re-checks plan, registry, and world identity before execution
# ---------------------------------------------------------------------------
def test_execution_rechecks_plan_registry_and_world_identity(tmp_path):
    """Mutating plan or registry after authorization causes execution to be rejected due to hash mismatch."""
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="tool_ok",
            description="Ok tool",
            input_schema={},
            positive_effects=[_cond("ok", [])],
            verifiers=[ObservationVerifier(verifier_id="v_ok", predicate="ok", command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="p11_drift",
        goal_description="Test drift rejection",
        initial_state=[],
        actions=[_action(action_id="act1", capability_name="tool_ok", positive_effects=[_cond("ok", [])])],
    )
    session = PlanningSession(session_id="s11_drift")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    cert = session.authorize_selected(reg, policy_hash="policy_initial")

    # Tamper with certificate plan hash
    tampered_cert = cert.model_copy(update={"plan_hash": "bad_hash_0000000000000000000000000000000000000000000000000000000000000000"})
    session.start_execution(reg, policy_hash="policy_initial")
    manager = ExecutionPlanManager(session=session, registry=reg, ledger=EvidenceLedger(session_id=session.session_id))

    with pytest.raises((ValueError, SignatureVerificationError, StateDriftError)):
        manager.execute_authorized_plan(tampered_cert)


# ---------------------------------------------------------------------------
# Test 12: Runtime does not mutate PlanIR to fake observations
# ---------------------------------------------------------------------------
def test_runtime_does_not_mutate_plan_ir_to_fake_observation(tmp_path):
    """PlanIR remains immutable during execution; empirical facts live only in manager.live_world_state and ledger."""
    target_file = tmp_path / "immutable.txt"
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
        plan_id="p12_immutable",
        goal_description="Test plan immutability",
        initial_state=[_fact("system_ready", [])],
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

    session = PlanningSession(session_id="s12_immutable")
    session.submit_draft(plan)
    session.validate_candidate(1, reg, observed_world_state=plan.initial_state)
    session.select_version(1)
    policy_hash = reg.compute_registry_hash()
    cert = session.authorize_selected(reg, policy_hash=policy_hash)
    session.start_execution(reg, policy_hash=policy_hash, current_world_facts=plan.initial_state)

    manager = ExecutionPlanManager(
        session=session,
        registry=reg,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=plan.initial_state,
    )
    summary = manager.execute_authorized_plan(cert)
    assert summary.success is True

    # PlanIR was NOT mutated
    assert plan.compute_hash() == original_plan_hash
    assert len(plan.initial_state) == 1
    assert plan.initial_state[0].predicate == "system_ready"
