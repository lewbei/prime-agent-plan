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


def _fact(predicate: str, args: list[str], truth: FactTruth, ttl: float | None = None, created_at: float | None = None, source: SourceType = SourceType.OBSERVED_WORLD_STATE, evidence_ref: str | None = None) -> WorldFact:
    now = created_at if created_at is not None else time.time()
    meta = {}
    if evidence_ref:
        meta["evidence_ref"] = evidence_ref
    elif source == SourceType.OBSERVED_WORLD_STATE:
        meta["evidence_ref"] = f"ev_{predicate}"
    return WorldFact(
        predicate=predicate,
        args=args,
        truth=truth,
        ttl_seconds=ttl,
        created_at=now,
        updated_at=now,
        provenance=Provenance(source_type=source, confidence=1.0),
        metadata=meta,
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
            verifiers=[ObservationVerifier(verifier_id="v_svc", predicate="service_active", target_args_mapping=["{svc}"])],
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
    trusted_flag = _fact("safety_flag", ["active"], FactTruth.VERIFIED_FALSE)
    plan = PlanIR(
        plan_id="p11",
        goal_description="Test false invariant",
        initial_state=[trusted_flag],
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
    result = validator.validate_plan(plan, observed_world_state=[trusted_flag])
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


# ---------------------------------------------------------------------------
# Test 19: Argument-bearing effect verifier without target binding is UNWITNESSABLE
# ---------------------------------------------------------------------------
def test_argful_effect_verifier_without_target_binding_is_unwitnessable():
    """For an argument-bearing predicate, absence of explicit verifier target binding means UNWITNESSABLE."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="service_manager",
            description="Manage service",
            input_schema={"svc": {"type": "str", "required": True}},
            positive_effects=[_cond("service_running", ["{svc}"])],
            verifiers=[ObservationVerifier(verifier_id="v_svc", predicate="service_running", target_args_mapping=[])],  # Empty mapping!
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p19_unbound_argful_verifier",
        goal_description="Test argful effect verifier without target binding",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="service_manager",
                parameters={"svc": "prod"},
                positive_effects=[_cond("service_running", ["prod"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("service_running", ["prod"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("service_running(prod)")
    assert fact is not None
    assert fact.witnessability == WitnessabilityStatus.UNWITNESSABLE
    assert result.status == ValidationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Test 20: Planner inference initial fact cannot enter as VERIFIED_TRUE
# ---------------------------------------------------------------------------
def test_planner_inference_initial_fact_cannot_enter_as_verified_true():
    """Planner-authored initial facts claiming PLANNER_INFERENCE must not enter as VERIFIED_TRUE."""
    validator = EpistemicCausalValidator()
    f = WorldFact(
        predicate="admin_authorized",
        args=["prod"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
    )
    plan = PlanIR(
        plan_id="p20_forged_initial_fact",
        goal_description="Test forged initial fact rejection",
        initial_state=[f],
        actions=[
            _action(
                action_id="act1",
                capability_name="admin_tool",
                preconditions=[_cond("admin_authorized", ["prod"], FactTruth.VERIFIED_TRUE)],
            )
        ],
    )
    result = validator.validate_plan(plan)
    final_state = result.intermediate_states[0]
    fact = final_state.get("admin_authorized(prod)")
    assert fact is not None
    assert fact.truth == FactTruth.UNKNOWN
    assert fact.truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 21: Explicit assumption initial fact cannot enter as VERIFIED_TRUE
# ---------------------------------------------------------------------------
def test_explicit_assumption_initial_fact_cannot_enter_as_verified_true():
    """Planner-authored initial facts claiming EXPLICIT_ASSUMPTION must not enter as VERIFIED_TRUE."""
    validator = EpistemicCausalValidator()
    f = WorldFact(
        predicate="db_ready",
        args=["prod"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.EXPLICIT_ASSUMPTION),
    )
    plan = PlanIR(
        plan_id="p21_assumption_initial_fact",
        goal_description="Test assumption initial fact rejection",
        initial_state=[f],
        actions=[],
    )
    result = validator.validate_plan(plan)
    final_state = result.intermediate_states[0]
    fact = final_state.get("db_ready(prod)")
    assert fact is not None
    assert fact.truth == FactTruth.UNKNOWN
    assert fact.truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 22: Untrusted observed label alone does not prove fact without trusted snapshot
# ---------------------------------------------------------------------------
def test_untrusted_observed_label_alone_does_not_prove_fact():
    """Self-labeling OBSERVED_WORLD_STATE in PlanIR without trusted snapshot or evidence_ref must downgrade to UNKNOWN."""
    validator = EpistemicCausalValidator()
    untrusted_fact = WorldFact(
        predicate="root_access",
        args=["box1"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
    )
    plan = PlanIR(
        plan_id="p22_untrusted_observed",
        goal_description="Test untrusted observed label",
        initial_state=[untrusted_fact],
        actions=[],
    )
    # Case 1: No trusted snapshot provided -> downgrades to UNKNOWN
    result_untrusted = validator.validate_plan(plan, observed_world_state=None)
    assert result_untrusted.intermediate_states[0]["root_access(box1)"].truth == FactTruth.UNKNOWN

    # Case 2: Trusted snapshot provided -> verified
    trusted_fact = WorldFact(
        predicate="root_access",
        args=["box1"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        metadata={"evidence_ref": "ev_001"},
    )
    result_trusted = validator.validate_plan(plan, observed_world_state=[trusted_fact])
    assert result_trusted.intermediate_states[0]["root_access(box1)"].truth == FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 23: Witnessable projected effect can close plan causal link
# ---------------------------------------------------------------------------
def test_witnessable_projected_effect_can_close_plan_causal_link():
    """Action A produces projected supported effect X, which satisfies Action B's precondition at plan validation."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="builder",
            description="Build app",
            input_schema={"app": {"type": "str", "required": True}},
            positive_effects=[_cond("app_built", ["{app}"])],
            verifiers=[ObservationVerifier(verifier_id="v_build", predicate="app_built", target_args_mapping=["{app}"])],
        )
    )
    registry.register(
        CapabilityEntry(
            name="deployer",
            description="Deploy app",
            input_schema={"app": {"type": "str", "required": True}},
            preconditions=[_cond("app_built", ["{app}"])],
            positive_effects=[_cond("app_deployed", ["{app}"])],
            verifiers=[ObservationVerifier(verifier_id="v_deploy", predicate="app_deployed", target_args_mapping=["{app}"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p23_causal_link",
        goal_description="Test causal link closure",
        initial_state=[],
        actions=[
            _action(
                action_id="step1",
                capability_name="builder",
                parameters={"app": "my_app"},
                positive_effects=[_cond("app_built", ["my_app"])],
            ),
            _action(
                action_id="step2",
                capability_name="deployer",
                parameters={"app": "my_app"},
                preconditions=[_cond("app_built", ["my_app"])],
                positive_effects=[_cond("app_deployed", ["my_app"])],
            ),
        ],
        success_criteria=[_criterion("c1", _cond("app_deployed", ["my_app"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.PASS
    assert len(result.criteria_satisfied) == 1


# ---------------------------------------------------------------------------
# Test 24: Projected effect does not become empirically verified
# ---------------------------------------------------------------------------
def test_projected_effect_does_not_become_empirically_verified():
    """Projected causal effect has projected_truth == SUPPORTED_TRUE but empirical truth == UNKNOWN."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="builder",
            description="Build app",
            input_schema={"app": {"type": "str", "required": True}},
            positive_effects=[_cond("app_built", ["{app}"])],
            verifiers=[ObservationVerifier(verifier_id="v_build", predicate="app_built", target_args_mapping=["{app}"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p24_projected_vs_empirical",
        goal_description="Test projected vs empirical state",
        initial_state=[],
        actions=[
            _action(
                action_id="step1",
                capability_name="builder",
                parameters={"app": "my_app"},
                positive_effects=[_cond("app_built", ["my_app"])],
            )
        ],
    )
    result = validator.validate_plan(plan, registry=registry)
    final_state = result.intermediate_states[-1]
    fact = final_state.get("app_built(my_app)")
    assert fact is not None
    assert fact.truth == FactTruth.UNKNOWN
    assert fact.truth != FactTruth.VERIFIED_TRUE
    assert getattr(fact, "projected_truth", None) is not None
    from plan_mode.ir import ProjectedTruth
    assert fact.projected_truth == ProjectedTruth.SUPPORTED_TRUE


# ---------------------------------------------------------------------------
# Test 25: Runtime precondition still requires empirical witness
# ---------------------------------------------------------------------------
def test_runtime_precondition_still_requires_empirical_witness():
    """At runtime check, an action requiring empirical VERIFIED_TRUE fails if truth is still UNKNOWN."""
    fact_projected_only = WorldFact(
        predicate="app_built",
        args=["my_app"],
        truth=FactTruth.UNKNOWN,
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
    )
    from plan_mode.ir import ProjectedTruth
    fact_projected_only.projected_truth = ProjectedTruth.SUPPORTED_TRUE

    # Empirical check requires fact.truth == VERIFIED_TRUE
    assert fact_projected_only.truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 26: Two-step registered plan can be plan feasibility PASS
# ---------------------------------------------------------------------------
def test_two_step_registered_plan_can_be_plan_feasibility_pass():
    """A 2-step plan with trusted initial state and registered verifiers achieves plan-time feasibility PASS."""
    trusted_init = WorldFact(
        predicate="src_ready",
        args=["repo1"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        metadata={"evidence_ref": "ev_init"},
    )
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="compile",
            description="Compile code",
            input_schema={"repo": {"type": "str", "required": True}},
            preconditions=[_cond("src_ready", ["{repo}"])],
            positive_effects=[_cond("binary_ready", ["{repo}"])],
            verifiers=[ObservationVerifier(verifier_id="v_comp", predicate="binary_ready", target_args_mapping=["{repo}"])],
        )
    )
    registry.register(
        CapabilityEntry(
            name="publish",
            description="Publish binary",
            input_schema={"repo": {"type": "str", "required": True}},
            preconditions=[_cond("binary_ready", ["{repo}"])],
            positive_effects=[_cond("published", ["{repo}"])],
            verifiers=[ObservationVerifier(verifier_id="v_pub", predicate="published", target_args_mapping=["{repo}"])],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p26_two_step_pass",
        goal_description="Test two step plan PASS",
        initial_state=[trusted_init],
        actions=[
            _action(
                action_id="act1",
                capability_name="compile",
                parameters={"repo": "repo1"},
                preconditions=[_cond("src_ready", ["repo1"])],
                positive_effects=[_cond("binary_ready", ["repo1"])],
            ),
            _action(
                action_id="act2",
                capability_name="publish",
                parameters={"repo": "repo1"},
                preconditions=[_cond("binary_ready", ["repo1"])],
                positive_effects=[_cond("published", ["repo1"])],
            ),
        ],
        success_criteria=[_criterion("c1", _cond("published", ["repo1"]))],
    )
    result = validator.validate_plan(plan, registry=registry, observed_world_state=[trusted_init])
    assert result.status == ValidationStatus.PASS
    assert len(result.criteria_satisfied) == 1
    assert len(result.blocker_reasons) == 0


# ---------------------------------------------------------------------------
# Test 27: Forged evidence_ref in PlanIR cannot ground verified initial fact
# ---------------------------------------------------------------------------
def test_forged_evidence_ref_cannot_ground_verified_initial_fact():
    """PlanIR metadata['evidence_ref'] without trusted snapshot must downgrade to UNKNOWN."""
    validator = EpistemicCausalValidator()
    forged_fact = WorldFact(
        predicate="admin_authorized",
        args=["prod"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        metadata={"evidence_ref": "fake-ev-123"},
    )
    plan = PlanIR(
        plan_id="p27_forged_evidence_ref",
        goal_description="Test forged evidence_ref rejection",
        initial_state=[forged_fact],
        actions=[
            _action(
                action_id="act1",
                capability_name="admin_tool",
                preconditions=[_cond("admin_authorized", ["prod"], FactTruth.VERIFIED_TRUE)],
            )
        ],
    )
    result = validator.validate_plan(plan, observed_world_state=None)
    fact = result.intermediate_states[0].get("admin_authorized(prod)")
    assert fact is not None
    assert fact.truth == FactTruth.UNKNOWN
    assert fact.truth != FactTruth.VERIFIED_TRUE
    from plan_mode.ir import ProjectedTruth
    assert fact.projected_truth == ProjectedTruth.UNSUPPORTED
    assert result.status == ValidationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Test 28: Planner evidence_ref is not a trust source
# ---------------------------------------------------------------------------
def test_planner_evidence_ref_is_not_a_trust_source():
    """A PlanIR evidence_ref is a claim only; presence of the string must never create empirical truth."""
    validator = EpistemicCausalValidator()
    claim_fact = WorldFact(
        predicate="secret_unlocked",
        args=["vault_1"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        metadata={"evidence_ref": "self_claimed_ev_999"},
    )
    plan = PlanIR(
        plan_id="p28_claim_not_trust_source",
        goal_description="Test claim is not trust source",
        initial_state=[claim_fact],
        actions=[],
    )
    result = validator.validate_plan(plan, observed_world_state=None)
    fact = result.intermediate_states[0].get("secret_unlocked(vault_1)")
    assert fact is not None
    assert fact.truth == FactTruth.UNKNOWN
    assert fact.truth != FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 29: Forged evidence_ref with unrelated trusted snapshot stays UNKNOWN
# ---------------------------------------------------------------------------
def test_forged_evidence_ref_with_unrelated_trusted_snapshot_stays_unknown():
    """Fact with forged evidence_ref omitted from trusted snapshot must remain UNKNOWN."""
    validator = EpistemicCausalValidator()
    trusted_other = WorldFact(
        predicate="other_system_online",
        args=["node1"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
    )
    forged_target = WorldFact(
        predicate="target_system_online",
        args=["node2"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        metadata={"evidence_ref": "ev_target_node2"},
    )
    plan = PlanIR(
        plan_id="p29_unrelated_snapshot",
        goal_description="Test unrelated snapshot leaves ungrounded fact UNKNOWN",
        initial_state=[forged_target],
        actions=[],
    )
    result = validator.validate_plan(plan, observed_world_state=[trusted_other])
    fact = result.intermediate_states[0].get("target_system_online(node2)")
    assert fact is not None
    assert fact.truth == FactTruth.UNKNOWN
    from plan_mode.ir import ProjectedTruth
    assert fact.projected_truth == ProjectedTruth.UNSUPPORTED


# ---------------------------------------------------------------------------
# Test 30: Trusted observed_world_state can ground initial fact
# ---------------------------------------------------------------------------
def test_trusted_observed_world_state_can_ground_initial_fact():
    """Positive control: matching fact in observed_world_state grounds initial state as VERIFIED_TRUE."""
    validator = EpistemicCausalValidator()
    trusted_db = WorldFact(
        predicate="db_online",
        args=["prod"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
    )
    plan_fact = WorldFact(
        predicate="db_online",
        args=["prod"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
    )
    plan = PlanIR(
        plan_id="p30_trusted_grounding",
        goal_description="Test trusted snapshot grounding",
        initial_state=[plan_fact],
        actions=[],
    )
    result = validator.validate_plan(plan, observed_world_state=[trusted_db])
    fact = result.intermediate_states[0].get("db_online(prod)")
    assert fact is not None
    assert fact.truth == FactTruth.VERIFIED_TRUE
    from plan_mode.ir import ProjectedTruth
    assert fact.projected_truth == ProjectedTruth.SUPPORTED_TRUE


# ---------------------------------------------------------------------------
# Test 31: Untrusted evidence_ref does not change validation semantics
# ---------------------------------------------------------------------------
def test_untrusted_evidence_ref_does_not_change_validation_semantics():
    """Adding or removing an unresolved evidence_ref inside PlanIR must not promote UNKNOWN to VERIFIED_TRUE."""
    validator = EpistemicCausalValidator()
    f_without = WorldFact(
        predicate="cache_warmed",
        args=["redis"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
    )
    f_with = WorldFact(
        predicate="cache_warmed",
        args=["redis"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
        metadata={"evidence_ref": "arbitrary_ref_abc"},
    )
    p_without = PlanIR(plan_id="p31_without", goal_description="Without ref", initial_state=[f_without], actions=[])
    p_with = PlanIR(plan_id="p31_with", goal_description="With ref", initial_state=[f_with], actions=[])

    r_without = validator.validate_plan(p_without, observed_world_state=None)
    r_with = validator.validate_plan(p_with, observed_world_state=None)

    assert r_without.intermediate_states[0]["cache_warmed(redis)"].truth == FactTruth.UNKNOWN
    assert r_with.intermediate_states[0]["cache_warmed(redis)"].truth == FactTruth.UNKNOWN
    assert r_without.status == r_with.status == ValidationStatus.PASS  # (PASS because no actions/criteria, but fact is UNKNOWN)


# ---------------------------------------------------------------------------
# Test 32: Conflicting duplicate trusted observations yield CONFLICT (FAIL)
# ---------------------------------------------------------------------------
def test_conflicting_duplicate_trusted_observations_yield_conflict():
    """Conflicting duplicate facts in observed_world_state (TRUE + FALSE) must merge to CONFLICT and FAIL validation."""
    validator = EpistemicCausalValidator()
    f_true = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)
    f_false = _fact("service_running", ["prod"], FactTruth.VERIFIED_FALSE)
    plan = PlanIR(
        plan_id="p32_trusted_conflict",
        goal_description="Test trusted conflict",
        initial_state=[f_true],
        actions=[],
    )
    result = validator.validate_plan(plan, observed_world_state=[f_true, f_false])
    assert result.status == ValidationStatus.FAIL
    fact = result.intermediate_states[0].get("service_running(prod)")
    assert fact is not None
    assert fact.truth == FactTruth.CONFLICT
    from plan_mode.ir import ProjectedTruth
    assert fact.projected_truth == ProjectedTruth.CONFLICT
    assert any("conflict" in b.lower() or "contradict" in b.lower() for b in result.blocker_reasons)


# ---------------------------------------------------------------------------
# Test 33: Duplicate trusted observations with same truth preserve truth
# ---------------------------------------------------------------------------
def test_duplicate_trusted_observations_same_truth_preserve_truth():
    """Duplicate observations with same truth (TRUE + TRUE) must preserve VERIFIED_TRUE."""
    validator = EpistemicCausalValidator()
    f1 = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)
    f2 = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)
    plan = PlanIR(
        plan_id="p33_trusted_same_truth",
        goal_description="Test trusted same truth duplicate",
        initial_state=[f1],
        actions=[],
    )
    result = validator.validate_plan(plan, observed_world_state=[f1, f2])
    fact = result.intermediate_states[0].get("service_running(prod)")
    assert fact is not None
    assert fact.truth == FactTruth.VERIFIED_TRUE
    from plan_mode.ir import ProjectedTruth
    assert fact.projected_truth == ProjectedTruth.SUPPORTED_TRUE


# ---------------------------------------------------------------------------
# Test 34: UNKNOWN and VERIFIED trusted observations merge by lattice
# ---------------------------------------------------------------------------
def test_unknown_and_verified_trusted_observations_merge_by_lattice():
    """UNKNOWN + VERIFIED_TRUE -> VERIFIED_TRUE and VERIFIED_TRUE + UNKNOWN -> VERIFIED_TRUE."""
    validator = EpistemicCausalValidator()
    f_unk = _fact("service_running", ["prod"], FactTruth.UNKNOWN)
    f_true = _fact("service_running", ["prod"], FactTruth.VERIFIED_TRUE)
    plan = PlanIR(
        plan_id="p34_lattice_merge",
        goal_description="Test lattice upgrade",
        initial_state=[f_true],
        actions=[],
    )
    # UNKNOWN then TRUE
    r1 = validator.validate_plan(plan, observed_world_state=[f_unk, f_true])
    assert r1.intermediate_states[0]["service_running(prod)"].truth == FactTruth.VERIFIED_TRUE

    # TRUE then UNKNOWN
    r2 = validator.validate_plan(plan, observed_world_state=[f_true, f_unk])
    assert r2.intermediate_states[0]["service_running(prod)"].truth == FactTruth.VERIFIED_TRUE


# ---------------------------------------------------------------------------
# Test 35: Trusted observation merge is order independent
# ---------------------------------------------------------------------------
def test_trusted_observation_merge_is_order_independent():
    """For conflicting observations, reversing the list produces the exact same merged CONFLICT result."""
    validator = EpistemicCausalValidator()
    f_true = _fact("flag", ["x"], FactTruth.VERIFIED_TRUE)
    f_false = _fact("flag", ["x"], FactTruth.VERIFIED_FALSE)
    plan = PlanIR(plan_id="p35_order_indep", goal_description="Order indep", initial_state=[f_true], actions=[])

    r_forward = validator.validate_plan(plan, observed_world_state=[f_true, f_false])
    r_reverse = validator.validate_plan(plan, observed_world_state=[f_false, f_true])

    assert r_forward.status == r_reverse.status == ValidationStatus.FAIL
    assert r_forward.intermediate_states[0]["flag(x)"].truth == r_reverse.intermediate_states[0]["flag(x)"].truth == FactTruth.CONFLICT


# ---------------------------------------------------------------------------
# Test 36: Trusted snapshot dictionary key mismatch is canonicalized
# ---------------------------------------------------------------------------
def test_trusted_snapshot_dict_key_mismatch_is_rejected_or_canonicalized():
    """Dictionary key mismatch against WorldFact.fact_key must normalize by WorldFact.fact_key without aliasing."""
    validator = EpistemicCausalValidator()
    f = _fact("system_ready", ["prod"], FactTruth.VERIFIED_TRUE)
    dict_input = {"wrong_alias_key": f}
    plan = PlanIR(
        plan_id="p36_dict_key_mismatch",
        goal_description="Test dict key normalization",
        initial_state=[f],
        actions=[],
    )
    result = validator.validate_plan(plan, observed_world_state=dict_input)
    assert "wrong_alias_key" not in result.intermediate_states[0]
    assert "system_ready(prod)" in result.intermediate_states[0]
    assert result.intermediate_states[0]["system_ready(prod)"].truth == FactTruth.VERIFIED_TRUE
