"""Adversarial Soundness and Invariant Tests for Epistemic Causal Validator."""

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


def _fact(predicate: str, args: list[str], truth: FactTruth, ttl: float | None = None, created_at: float | None = None) -> WorldFact:
    now = created_at if created_at is not None else time.time()
    return WorldFact(
        predicate=predicate,
        args=args,
        truth=truth,
        ttl_seconds=ttl,
        created_at=now,
        updated_at=now,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE, confidence=1.0),
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


def _criterion(criterion_id: str, condition: PredicateCondition, is_mandatory: bool = True) -> SuccessCriterion:
    return SuccessCriterion(
        criterion_id=criterion_id,
        description=f"Criterion {criterion_id}",
        condition=condition,
        is_mandatory=is_mandatory,
    )


def test_unknown_precondition_does_not_apply_effects():
    """Invariant 1: If precondition is UNKNOWN, action effects MUST NOT be applied to state."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p1",
        goal_description="Test unknown gating",
        initial_state=[],  # is_authenticated is UNKNOWN
        actions=[
            _action(
                action_id="act1",
                capability_name="delete_database",
                preconditions=[_cond("is_authenticated", ["admin"])],
                positive_effects=[_cond("database_deleted", ["prod"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("database_deleted", ["prod"]))],
    )
    result = validator.validate_plan(plan)
    # The action could not execute with certainty; database_deleted must NOT be verified in intermediate/final state
    assert result.status == ValidationStatus.UNKNOWN
    assert "is_authenticated(admin)" in result.unknown_facts
    # Final state must NOT contain verified database_deleted
    final_state = result.intermediate_states[-1]
    db_fact = final_state.get("database_deleted(prod)")
    assert db_fact is None or db_fact.truth != FactTruth.VERIFIED_TRUE


def test_unknown_action_cannot_satisfy_downstream_precondition():
    """Invariant 2: Downstream action depending on an un-executed / unknown action must remain UNKNOWN."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p2",
        goal_description="Test downstream propagation",
        initial_state=[],  # has_code is UNKNOWN
        actions=[
            _action(
                action_id="act1",
                capability_name="build_app",
                preconditions=[_cond("has_code", ["main"])],
                positive_effects=[_cond("app_built", ["v1"])],
            ),
            _action(
                action_id="act2",
                capability_name="deploy_app",
                preconditions=[_cond("app_built", ["v1"])],
                positive_effects=[_cond("app_deployed", ["prod"])],
            ),
        ],
        success_criteria=[_criterion("c1", _cond("app_deployed", ["prod"]))],
    )
    result = validator.validate_plan(plan)
    assert result.status == ValidationStatus.UNKNOWN
    final_state = result.intermediate_states[-1]
    deployed = final_state.get("app_deployed(prod)")
    assert deployed is None or deployed.truth != FactTruth.VERIFIED_TRUE


def test_conflicting_duplicate_initial_facts_yield_conflict():
    """Invariant 3: Duplicate contradictory facts in initial_state must merge to CONFLICT."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p3",
        goal_description="Test duplicate conflict",
        initial_state=[
            _fact("service_running", ["api"], FactTruth.VERIFIED_TRUE),
            _fact("service_running", ["api"], FactTruth.VERIFIED_FALSE),
        ],
        actions=[
            _action(
                action_id="act1",
                capability_name="check_service",
                preconditions=[_cond("service_running", ["api"], FactTruth.VERIFIED_TRUE)],
                positive_effects=[_cond("checked", ["api"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("checked", ["api"]))],
    )
    result = validator.validate_plan(plan)
    assert result.status == ValidationStatus.FAIL
    assert any("conflict" in b.lower() for b in result.blocker_reasons)


def test_registered_and_observed_fact_conflict_is_preserved():
    """Invariant 4: Merging conflicting truth states preserves CONFLICT in 4-state lattice."""
    assert merge_fact_truth(FactTruth.VERIFIED_TRUE, FactTruth.VERIFIED_FALSE) == FactTruth.CONFLICT
    assert merge_fact_truth(FactTruth.CONFLICT, FactTruth.VERIFIED_TRUE) == FactTruth.CONFLICT
    assert merge_fact_truth(FactTruth.CONFLICT, FactTruth.UNKNOWN) == FactTruth.CONFLICT


def test_unregistered_effectful_capability_cannot_pass():
    """Invariant 5: Validating an action whose capability is not in the registry must fail validation."""
    validator = EpistemicCausalValidator()
    registry = CapabilityRegistry()  # Empty registry
    plan = PlanIR(
        plan_id="p5",
        goal_description="Test unregistered capability",
        initial_state=[_fact("flag", ["on"], FactTruth.VERIFIED_TRUE)],
        actions=[
            _action(
                action_id="act1",
                capability_name="unregistered_tool",
                preconditions=[_cond("flag", ["on"])],
                positive_effects=[_cond("done", ["yes"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("done", ["yes"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL
    assert any("not found" in b.lower() or "unregistered" in b.lower() or "missing" in b.lower() for b in result.blocker_reasons)


def test_action_effect_must_be_declared_by_capability():
    """Invariant 6: Action effects must be a subset of the registered capability declared effects."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="safe_reader",
            description="Read a file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_read", ["{path}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="file_read")],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p6",
        goal_description="Test undeclared effect",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="safe_reader",
                parameters={"path": "/tmp/a"},
                preconditions=[],
                positive_effects=[
                    _cond("file_read", ["/tmp/a"]),
                    _cond("system_reformatted", ["root"]),  # Undeclared!
                ],
            )
        ],
        success_criteria=[_criterion("c1", _cond("system_reformatted", ["root"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL
    assert any("undeclared" in b.lower() or "effect" in b.lower() for b in result.blocker_reasons)


def test_undeclared_extra_effect_cannot_become_verified():
    """Invariant 7: Planner cannot hallucinate effects not supported by registry capability."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="mkdir",
            description="Make directory",
            input_schema={"dir": "str"},
            positive_effects=[_cond("dir_exists", ["{dir}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="dir_exists")],
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p7",
        goal_description="Test hallucinated effect",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="mkdir",
                parameters={"dir": "/data"},
                positive_effects=[_cond("database_seeded", ["all"])],  # Not in capability
            )
        ],
        success_criteria=[_criterion("c1", _cond("database_seeded", ["all"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status == ValidationStatus.FAIL


def test_missing_effect_verifier_prevents_pass():
    """Invariant 8: An effect without an observation verifier in the registry cannot yield PASS."""
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="unverified_action",
            description="Action without verifiers",
            input_schema={},
            positive_effects=[_cond("unverifiable_fact", ["val"])],
            verifiers=[],  # Missing verifier!
        )
    )
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p8",
        goal_description="Test missing verifier",
        initial_state=[],
        actions=[
            _action(
                action_id="act1",
                capability_name="unverified_action",
                positive_effects=[_cond("unverifiable_fact", ["val"])],
            )
        ],
        success_criteria=[_criterion("c1", _cond("unverifiable_fact", ["val"]))],
    )
    result = validator.validate_plan(plan, registry=registry)
    assert result.status in (ValidationStatus.UNKNOWN, ValidationStatus.FAIL)
    assert result.status != ValidationStatus.PASS


def test_active_until_action_constraint_is_enforced():
    """Invariant 9: Constraint with active_until_action_id must be checked until that step."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p9",
        goal_description="Test active_until constraint",
        initial_state=[_fact("lock_held", ["mutex"], FactTruth.VERIFIED_TRUE)],
        actions=[
            _action(
                action_id="step1",
                capability_name="work",
                preconditions=[_cond("lock_held", ["mutex"])],
                positive_effects=[_cond("step1_done", ["true"])],
            ),
            _action(
                action_id="step2",
                capability_name="release_lock",
                preconditions=[],
                negative_effects=[_cond("lock_held", ["mutex"])],
                positive_effects=[_cond("lock_released", ["true"])],
            ),
            _action(
                action_id="step3",
                capability_name="after_release",
                preconditions=[],
                positive_effects=[_cond("step3_done", ["true"])],
            ),
        ],
        hard_constraints=[
            HardConstraint(
                constraint_id="c_lock",
                description="Lock must be held during initial work",
                condition=_cond("lock_held", ["mutex"], FactTruth.VERIFIED_TRUE),
                active_until_action_id="step2",  # Only required until step2 executes
                provenance=Provenance(source_type=SourceType.DOMAIN_POLICY),
            )
        ],
        success_criteria=[_criterion("sc", _cond("step3_done", ["true"]))],
    )
    result = validator.validate_plan(plan)
    # Releasing the lock at step2 must NOT violate constraint c_lock because c_lock was only active until step2
    assert result.status == ValidationStatus.PASS


def test_expired_fact_is_unknown_at_point_of_use():
    """Invariant 10: Fact with TTL expired at point of use decays to UNKNOWN."""
    validator = EpistemicCausalValidator()
    t0 = 1000.0
    plan = PlanIR(
        plan_id="p10",
        goal_description="Test TTL point-of-use",
        initial_state=[
            _fact("auth_token_valid", ["session"], FactTruth.VERIFIED_TRUE, ttl=10.0, created_at=t0)
        ],
        actions=[
            _action(
                action_id="act1",
                capability_name="api_call",
                preconditions=[_cond("auth_token_valid", ["session"])],
                positive_effects=[_cond("api_success", ["true"])],
            )
        ],
        success_criteria=[_criterion("sc", _cond("api_success", ["true"]))],
    )
    # Validated at t0 + 20 (expired)
    result = validator.validate_plan(plan, current_time=t0 + 20.0)
    assert result.status == ValidationStatus.UNKNOWN
    assert "auth_token_valid(session)" in result.unknown_facts


def test_mandatory_unknown_success_criterion_yields_unknown():
    """Invariant 11: Mandatory criterion with UNKNOWN truth must yield UNKNOWN plan status."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p11",
        goal_description="Test mandatory unknown criterion",
        initial_state=[],
        actions=[],
        success_criteria=[
            _criterion("sc1", _cond("remote_server_healthy", ["node1"]), is_mandatory=True)
        ],
    )
    result = validator.validate_plan(plan)
    assert result.status == ValidationStatus.UNKNOWN
    assert "remote_server_healthy(node1)" in result.unknown_facts


def test_verified_false_mandatory_success_criterion_yields_fail():
    """Invariant 12: Mandatory criterion with VERIFIED_FALSE truth must yield FAIL plan status."""
    validator = EpistemicCausalValidator()
    plan = PlanIR(
        plan_id="p12",
        goal_description="Test mandatory false criterion",
        initial_state=[_fact("remote_server_healthy", ["node1"], FactTruth.VERIFIED_FALSE)],
        actions=[],
        success_criteria=[
            _criterion("sc1", _cond("remote_server_healthy", ["node1"], FactTruth.VERIFIED_TRUE), is_mandatory=True)
        ],
    )
    result = validator.validate_plan(plan)
    assert result.status == ValidationStatus.FAIL
    assert any("unmet" in b.lower() or "expected" in b.lower() for b in result.blocker_reasons)
