"""Adversarial Soundness and Invariant Tests for Epistemic Causal Validator (Phase 1)."""

import time
import pytest
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    HardConstraint,
    PlanIR,
    PredicateCondition,
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
    SchemaMismatchError,
    CapabilityNotFoundError,
)
from plan_mode.epistemic_validator import (
    EpistemicCausalValidator,
    ValidationStatus,
    merge_fact_truth,
)


def _fact(predicate: str, args: list[str], truth: FactTruth, ttl: float | None = None, created_at: float | None = None, source: SourceType = SourceType.OBSERVED_WORLD_STATE) -> WorldFact:
    now = created_at if created_at is not None else time.time()
    return WorldFact(
        predicate=predicate,
        args=args,
        truth=truth,
        ttl_seconds=ttl,
        created_at=now,
        updated_at=now,
        provenance=Provenance(source_type=source, confidence=1.0),
    )


def _cond(predicate: str, args: list[str], truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate=predicate, args=args, expected_truth=truth)


def _action(
    action_id: str,
    capability_name: str,
    preconditions: list[PredicateCondition] | None = None,
    positive_effects: list[PredicateCondition] | None = None,
    negative_effects: list[PredicateCondition] | None = None,
    parameters: dict | None = None,
    timeout_seconds: float = 10.0,
) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability_name,
        parameters=parameters or {},
        preconditions=preconditions or [],
        positive_effects=positive_effects or [],
        negative_effects=negative_effects or [],
        timeout_seconds=timeout_seconds,
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE, source_id=action_id),
    )


def _criterion(criterion_id: str, condition: PredicateCondition, is_mandatory: bool = True) -> SuccessCriterion:
    return SuccessCriterion(
        criterion_id=criterion_id,
        description=f"Criterion {criterion_id}",
        condition=condition,
        is_mandatory=is_mandatory,
    )


# ---------------------------------------------------------------------------
# Test 1: Effectful action without registry MUST yield UNKNOWN (not PASS)
# ---------------------------------------------------------------------------
def test_effectful_action_without_registry_yields_unknown():
    """An effectful action validated with registry=None must yield UNKNOWN due to missing grounding."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p1",
        goal_description="Test effectful without registry",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="magic_tool",
                positive_effects=[_cond("data_written", ["file.txt"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("data_written", ["file.txt"]))],
    )
    result = validator.validate_plan(plan, registry=None)
    assert result.status == ValidationStatus.UNKNOWN
    assert any("registry" in r.lower() or "grounding" in r.lower() or "unregistered" in r.lower() or "verifier" in r.lower() for r in (result.blocker_reasons + result.unknown_facts))


# ---------------------------------------------------------------------------
# Test 2: Unregistered capability yields UNKNOWN (lack of knowledge, not FAIL)
# ---------------------------------------------------------------------------
def test_unregistered_capability_yields_unknown_not_fail():
    """Empty registry + unknown capability must return UNKNOWN, not FAIL."""
    validator = EpistemicCausalValidator()
    registry = CapabilityRegistry()  # Empty registry
    plan = PlanIR(
        plan_id="p2",
        goal_description="Test unknown capability",
        initial_state=[_fact("input_ready", ["file.txt"], FactTruth.VERIFIED_TRUE)],
        actions=[
            _action(
                action_id="act1",
                capability_name="novel_probe_tool",
                preconditions=[_cond("input_ready", ["file.txt"])],
                positive_effects=[_cond("probed", ["file.txt"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("probed", ["file.txt"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    # Lack of capability registration is epistemic uncertainty (UNKNOWN), not a proven contradiction (FAIL)
    assert result.status == ValidationStatus.UNKNOWN
    assert result.status != ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# Test 3: Empty registered positive effect set strictly rejects claimed effect
# ---------------------------------------------------------------------------
def test_empty_registered_effect_set_rejects_claimed_effect():
    """Capability with positive_effects=[] strictly forbids any positive effects."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="read_only_tool",
            description="Pure read-only operation with zero side effects",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[],  # ZERO positive effects allowed
            negative_effects=[],
            verifiers=[],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p3",
        goal_description="Test empty positive effect set",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="read_only_tool",
                parameters={"path": "/tmp/a"},
                positive_effects=[_cond("file_modified", ["/tmp/a"])],  # Forbidden!
            )
        ],
        success_criteria=[_criterion("c1", _cond("file_modified", ["/tmp/a"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL
    assert any("effect" in b.lower() or "undeclared" in b.lower() for b in result.blocker_reasons)


# ---------------------------------------------------------------------------
# Test 4: Empty registered negative effect set strictly rejects claimed negative effect
# ---------------------------------------------------------------------------
def test_empty_registered_negative_effect_set_rejects_claimed_negative_effect():
    """Capability with negative_effects=[] strictly forbids any negative effects."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="creator_tool",
            description="Creation tool with no deletions",
            input_schema={"name": {"type": "str", "required": True}},
            positive_effects=[_cond("created", ["{name}"])],
            negative_effects=[],  # ZERO negative effects allowed
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="created")],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p4",
        goal_description="Test empty negative effect set",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="creator_tool",
                parameters={"name": "box"},
                positive_effects=[_cond("created", ["box"])],
                negative_effects=[_cond("deleted", ["old_box"])],  # Forbidden!
            )
        ],
        success_criteria=[_criterion("c1", _cond("created", ["box"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL
    assert any("negative" in b.lower() or "undeclared" in b.lower() for b in result.blocker_reasons)


# ---------------------------------------------------------------------------
# Test 5: Effect binding requires exact instantiated arguments (template unification)
# ---------------------------------------------------------------------------
def test_effect_binding_requires_exact_arguments():
    """Action cannot claim an effect on /etc/passwd when parameter is path=/tmp/a."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="file_writer",
            description="Writes to target path",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_written", ["{path}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="file_written")],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p5",
        goal_description="Test argument binding mismatch",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="file_writer",
                parameters={"path": "/tmp/a"},
                positive_effects=[_cond("file_written", ["/etc/passwd"])],  # Mismatch with path=/tmp/a!
            )
        ],
        success_criteria=[_criterion("c1", _cond("file_written", ["/etc/passwd"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL
    assert any("mismatch" in b.lower() or "argument" in b.lower() or "effect" in b.lower() for b in result.blocker_reasons)


# ---------------------------------------------------------------------------
# Test 6: Effect binding requires matching truth and sign
# ---------------------------------------------------------------------------
def test_effect_binding_requires_matching_truth_and_sign():
    """Action claiming VERIFIED_FALSE for an effect registered as VERIFIED_TRUE must fail contract."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="starter",
            description="Starts a service",
            input_schema={"svc": {"type": "str", "required": True}},
            positive_effects=[_cond("service_running", ["{svc}"], FactTruth.VERIFIED_TRUE)],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="service_running")],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p6",
        goal_description="Test truth sign mismatch",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="starter",
                parameters={"svc": "db"},
                positive_effects=[_cond("service_running", ["db"], FactTruth.VERIFIED_FALSE)],  # Sign mismatch!
            )
        ],
        success_criteria=[_criterion("c1", _cond("service_running", ["db"], FactTruth.VERIFIED_FALSE))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# Test 7: Verifier presence means WITNESSABLE / PLANNER_INFERENCE, NOT empirical OBSERVED_WORLD_STATE
# ---------------------------------------------------------------------------
def test_verifier_predicate_match_does_not_mean_effect_witnessed():
    """Plan simulation must produce PLANNER_INFERENCE provenance, not fabricate OBSERVED_WORLD_STATE."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="service_manager",
            description="Manage service",
            input_schema={"svc": {"type": "str", "required": True}},
            positive_effects=[_cond("service_active", ["{svc}"])],
            verifiers=[ObservationVerifier(verifier_id="v_svc", predicate="service_active")],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p7",
        goal_description="Test simulation provenance",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="service_manager",
                parameters={"svc": "web"},
                positive_effects=[_cond("service_active", ["web"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("service_active", ["web"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("service_active(web)")
    assert fact is not None
    # Simulation creates WITNESSABLE facts from planner inference, NOT empirical observed world facts
    assert fact.witnessability == WitnessabilityStatus.WITNESSABLE
    assert fact.provenance.source_type == SourceType.PLANNER_INFERENCE
    assert fact.provenance.source_type != SourceType.OBSERVED_WORLD_STATE
    assert fact.truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 8: Verifier binding requires matching effect arguments
# ---------------------------------------------------------------------------
def test_verifier_binding_requires_exact_effect_arguments():
    """Verifier must match the instantiated effect arguments, not just predicate name."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="scoped_reader",
            description="Reads scoped file",
            input_schema={"scope": {"type": "str", "required": True}},
            positive_effects=[_cond("scoped_read", ["{scope}"])],
            verifiers=[ObservationVerifier(verifier_id="v_scoped", predicate="other_predicate")],  # Predicate mismatch
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p8",
        goal_description="Test verifier arg match",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="scoped_reader",
                parameters={"scope": "user"},
                positive_effects=[_cond("scoped_read", ["user"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("scoped_read", ["user"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    # Effect is not witnessable because verifier predicate does not match
    assert result.status == ValidationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Test 9: Negative effect without verifier is not witnessable / verified
# ---------------------------------------------------------------------------
def test_negative_effect_without_verifier_is_not_verified():
    """Negative effects without observation verifiers cannot be witnessed and must yield UNKNOWN."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="killer",
            description="Kills process",
            input_schema={"pid": {"type": "int", "required": True}},
            negative_effects=[_cond("proc_running", ["{pid}"])],
            verifiers=[],  # No verifier for proc_running removal!
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p9",
        goal_description="Test negative effect verifier gating",
        initial_state=[_fact("proc_running", [123], FactTruth.VERIFIED_TRUE)],
        actions=[
            _action(
                action_id="act1",
                capability_name="killer",
                parameters={"pid": 123},
                negative_effects=[_cond("proc_running", [123])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("proc_running", [123], FactTruth.VERIFIED_FALSE))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Test 10: UNKNOWN hard constraint yields UNKNOWN (not FAIL)
# ---------------------------------------------------------------------------
def test_unknown_hard_constraint_yields_unknown_not_fail():
    """An invariant with UNKNOWN truth must yield UNKNOWN, not FAIL."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p10",
        goal_description="Test unknown invariant",
        initial_state=[],  # safety_flag is UNKNOWN
        actions=[],
        hard_constraints=[
            HardConstraint(
                constraint_id="inv1",
                description="Safety flag must be true",
                condition=_cond("safety_flag", ["active"], FactTruth.VERIFIED_TRUE),
                provenance=Provenance(source_type=SourceType.DOMAIN_POLICY),
            )
        ],
        success_criteria=[],
    )
    result = validator.validate_plan(plan)
    assert result.status == ValidationStatus.UNKNOWN
    assert result.status != ValidationStatus.FAIL
    assert "safety_flag(active)" in result.unknown_facts


# ---------------------------------------------------------------------------
# Test 11: VERIFIED_FALSE hard constraint yields FAIL
# ---------------------------------------------------------------------------
def test_verified_false_hard_constraint_yields_fail():
    """An invariant with VERIFIED_FALSE truth is a proven contradiction and must yield FAIL."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p11",
        goal_description="Test false invariant",
        initial_state=[_fact("safety_flag", ["active"], FactTruth.VERIFIED_FALSE)],
        actions=[],
        hard_constraints=[
            HardConstraint(
                constraint_id="inv1",
                description="Safety flag must be true",
                condition=_cond("safety_flag", ["active"], FactTruth.VERIFIED_TRUE),
                provenance=Provenance(source_type=SourceType.DOMAIN_POLICY),
            )
        ],
        success_criteria=[],
    )
    result = validator.validate_plan(plan)
    assert result.status == ValidationStatus.FAIL
    assert any("invariant" in b.lower() or "constraint" in b.lower() for b in result.blocker_reasons)


# ---------------------------------------------------------------------------
# Test 12: Point-of-use fact expiration during step-by-step execution
# ---------------------------------------------------------------------------
def test_point_of_use_fact_expiration_decays_at_action_step():
    """Fact is fresh at t0 but expires during step 1 duration (timeout), becoming UNKNOWN at step 2."""
    t0 = 1000.0
    validator = EpistemicCausalValidator()
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="long_task",
            description="Runs for 15s",
            input_schema={},
            positive_effects=[_cond("task1_done", ["true"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="task1_done")],
        )
    )
    registry.register(
        CapabilityEntry(
            name="dependent_task",
            description="Requires fresh auth",
            input_schema={},
            positive_effects=[_cond("task2_done", ["true"])],
            verifiers=[ObservationVerifier(verifier_id="v2", predicate="task2_done")],
        )
    )

    plan = PlanIR(
        plan_id="p12",
        goal_description="Test point-of-use TTL decay",
        initial_state=[
            # TTL is 10s
            _fact("auth_token", ["valid"], FactTruth.VERIFIED_TRUE, ttl=10.0, created_at=t0),
        ],
        actions=[
            # Act 1 takes 15s (advancing simulated time from t0 to t0+15)
            _action(
                action_id="act1",
                capability_name="long_task",
                positive_effects=[_cond("task1_done", ["true"])],
                timeout_seconds=15.0,
            ),
            # Act 2 runs at t0+15 and requires auth_token, which expired at t0+10
            _action(
                action_id="act2",
                capability_name="dependent_task",
                preconditions=[_cond("auth_token", ["valid"])],
                positive_effects=[_cond("task2_done", ["true"])],
            ),
        ],
        success_criteria=[_criterion("c1", _cond("task2_done", ["true"]))],
    )

    result = validator.validate_plan(plan, registry=registry, current_time=t0)
    assert result.status == ValidationStatus.UNKNOWN
    assert "auth_token(valid)" in result.unknown_facts


# ---------------------------------------------------------------------------
# Test 13: Witnessable planner effect is NOT FactTruth.VERIFIED_TRUE
# ---------------------------------------------------------------------------
def test_witnessable_planner_effect_is_not_verified_true():
    """A planner-predicted positive effect with a registered ObservationVerifier must NOT become FactTruth.VERIFIED_TRUE merely because the verifier exists."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="service_manager",
            description="Manage service",
            input_schema={"svc": {"type": "str", "required": True}},
            positive_effects=[_cond("service_running", ["{svc}"])],
            verifiers=[ObservationVerifier(verifier_id="v_svc", predicate="service_running", target_args_mapping=["{svc}"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p_witnessable_not_verified",
        goal_description="Test witnessable effect is not verified true",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="service_manager",
                parameters={"svc": "web"},
                positive_effects=[_cond("service_running", ["web"])],
            )
        ],
        success_criteria=[],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("service_running(web)")
    assert fact is not None
    assert fact.provenance.source_type == SourceType.PLANNER_INFERENCE
    assert fact.witnessability == WitnessabilityStatus.WITNESSABLE
    # Empirical truth MUST remain UNKNOWN until Phase 2 runtime executes the verifier
    assert fact.truth == FactTruth.UNKNOWN
    assert fact.truth != FactTruth.VERIFIED_TRUE
    assert fact.metadata.get("predicted_truth") == FactTruth.VERIFIED_TRUE.value


# ---------------------------------------------------------------------------
# Test 14: Witnessable predicted effect cannot satisfy verified precondition
# ---------------------------------------------------------------------------
def test_witnessable_predicted_effect_cannot_satisfy_verified_precondition():
    """Action A predicts X and X is witnessable but not yet empirically witnessed. Action B requires X == VERIFIED_TRUE.
    At planning-only validation, Action B must NOT treat X as empirically verified. The plan must remain UNKNOWN / require runtime witnessing."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="step_a_tool",
            description="Tool A",
            input_schema={},
            positive_effects=[_cond("state_x", ["ready"])],
            verifiers=[ObservationVerifier(verifier_id="v_x", predicate="state_x", target_args_mapping=["ready"])],
        )
    )
    registry.register(
        CapabilityEntry(
            name="step_b_tool",
            description="Tool B",
            input_schema={},
            positive_effects=[_cond("state_y", ["done"])],
            verifiers=[ObservationVerifier(verifier_id="v_y", predicate="state_y", target_args_mapping=["done"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p_predicted_precondition",
        goal_description="Test predicted effect cannot satisfy verified precondition",
        initial_state=[],
        actions=[
            _action(
                action_id="actA",
                capability_name="step_a_tool",
                positive_effects=[_cond("state_x", ["ready"])],
            ),
            _action(
                action_id="actB",
                capability_name="step_b_tool",
                preconditions=[_cond("state_x", ["ready"], FactTruth.VERIFIED_TRUE)],
                positive_effects=[_cond("state_y", ["done"])],
            ),
        ],
    )
    result = validator.validate_plan(plan, registry=registry)
    # Plan must remain UNKNOWN because actB requires empirical VERIFIED_TRUE which has only been predicted
    assert result.status == ValidationStatus.UNKNOWN
    assert "state_x(ready)" in result.unknown_facts


# ---------------------------------------------------------------------------
# Test 15: Verifier binding rejects same predicate with wrong target arguments
# ---------------------------------------------------------------------------
def test_verifier_binding_rejects_same_predicate_wrong_arguments():
    """Verifier has same predicate but bound to different target arguments. It must not witness the effect."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="db_deployer",
            description="Deploy database",
            input_schema={"env": {"type": "str", "required": True}},
            positive_effects=[_cond("service_running", ["{env}"])],
            verifiers=[ObservationVerifier(verifier_id="v_staging", predicate="service_running", target_args_mapping=["staging"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p_wrong_verifier_args",
        goal_description="Test verifier wrong arguments",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="db_deployer",
                parameters={"env": "prod"},
                positive_effects=[_cond("service_running", ["prod"])],
            )
        ],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("service_running(prod)")
    assert fact is not None
    # Because verifier targets 'staging', effect on 'prod' is unwitnessable
    assert fact.witnessability == WitnessabilityStatus.UNWITNESSABLE
    assert "service_running(prod)" in result.unknown_facts


# ---------------------------------------------------------------------------
# Test 16: Verifier binding accepts same predicate with exact bound arguments
# ---------------------------------------------------------------------------
def test_verifier_binding_accepts_same_predicate_exact_bound_arguments():
    """Positive control: verifier predicate and bound arguments match exact effect arguments."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="db_deployer",
            description="Deploy database",
            input_schema={"env": {"type": "str", "required": True}},
            positive_effects=[_cond("service_running", ["{env}"])],
            verifiers=[ObservationVerifier(verifier_id="v_env", predicate="service_running", target_args_mapping=["{env}"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p_exact_verifier_args",
        goal_description="Test verifier exact arguments match",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="db_deployer",
                parameters={"env": "prod"},
                positive_effects=[_cond("service_running", ["prod"])],
            )
        ],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("service_running(prod)")
    assert fact is not None
    assert fact.witnessability == WitnessabilityStatus.WITNESSABLE


# ---------------------------------------------------------------------------
# Test 17: Negative effect verifier binding rejects wrong target arguments
# ---------------------------------------------------------------------------
def test_negative_verifier_binding_rejects_wrong_arguments():
    """Negative effect verifier with wrong target arguments must reject witnessing."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="db_stopper",
            description="Stop database",
            input_schema={"env": {"type": "str", "required": True}},
            negative_effects=[_cond("service_running", ["{env}"], FactTruth.VERIFIED_FALSE)],
            verifiers=[ObservationVerifier(verifier_id="v_staging_stop", predicate="service_running", target_args_mapping=["staging"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p_negative_wrong_verifier_args",
        goal_description="Test negative effect verifier wrong arguments",
        initial_state=[_fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)],
        actions=[
            _action(
                action_id="act1",
                capability_name="db_stopper",
                parameters={"env": "prod"},
                negative_effects=[_cond("service_running", ["prod"], FactTruth.VERIFIED_FALSE)],
            )
        ],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("service_running(prod)")
    assert fact is not None
    assert fact.witnessability == WitnessabilityStatus.UNWITNESSABLE
    assert "service_running(prod)" in result.unknown_facts


# ---------------------------------------------------------------------------
# Test 18: Effect argument types are not collapsed to strings
# ---------------------------------------------------------------------------
def test_effect_argument_types_are_not_collapsed_to_strings():
    """Registered effect arg = integer 123 vs action effect arg = string '123' must not match."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="proc_killer",
            description="Kills process",
            input_schema={"pid": {"type": "int", "required": True}},
            negative_effects=[_cond("proc_running", [123], FactTruth.VERIFIED_FALSE)],  # integer 123
            verifiers=[ObservationVerifier(verifier_id="v_proc", predicate="proc_running", target_args_mapping=[123])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p_type_collapse",
        goal_description="Test type collapse rejection",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="proc_killer",
                parameters={"pid": 123},
                negative_effects=[_cond("proc_running", ["123"], FactTruth.VERIFIED_FALSE)],  # string '123'!
            )
        ],
    )
    result = validator.validate_plan(plan, registry=registry)
    # Schema validation must fail because type int != type str
    assert result.status == ValidationStatus.FAIL
    assert any("mismatch" in b.lower() or "undeclared" in b.lower() for b in result.blocker_reasons)
