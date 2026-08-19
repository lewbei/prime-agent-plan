"""Adversarial typed fact identity tests for planner/runtime state keys."""

from plan_mode.epistemic_validator import (
    EpistemicCausalValidator,
    PlanValidationResult,
    ValidationStatus,
    normalize_trusted_snapshot,
)
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    WorldFact,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier
from plan_mode.runtime.executor import ExecutionPlanManager
from plan_mode.runtime.ledger import EvidenceLedger
from plan_mode.runtime.sandbox import SandboxExecutionResult
from plan_mode.session import PlanningSession


def _prov(source: SourceType = SourceType.OBSERVED_WORLD_STATE) -> Provenance:
    return Provenance(source_type=source, confidence=1.0)


def _fact(arg, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> WorldFact:
    return WorldFact(
        predicate="resource",
        args=[arg],
        truth=truth,
        provenance=_prov(),
    )


def _cond(arg, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate="resource", args=[arg], expected_truth=truth)


def _action(pre_arg=None, effect_arg=None) -> ActionIR:
    return ActionIR(
        action_id="a1",
        capability_name="typed_tool",
        parameters={},
        preconditions=[] if pre_arg is None else [_cond(pre_arg)],
        positive_effects=[] if effect_arg is None else [_cond(effect_arg)],
        provenance=_prov(SourceType.PLANNER_INFERENCE),
    )


class _PassValidator:
    """Test seam used to attack runtime precondition lookup independently."""

    default_ttl_decay_to_unknown = True

    def validate_plan(self, *args, **kwargs):
        return PlanValidationResult(status=ValidationStatus.PASS)


def _runtime_registry(pre_arg) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="typed_tool",
            description="Typed runtime lookup test capability",
            input_schema={},
            preconditions=[_cond(pre_arg)],
            positive_effects=[],
            executor_command_template=["true"],
        )
    )
    return registry


def test_fact_identity_distinguishes_integer_and_string_arguments():
    assert _fact(123).fact_key != _fact("123").fact_key
    assert _cond(123).fact_key != _cond("123").fact_key


def test_fact_identity_distinguishes_bool_and_integer_arguments():
    assert _fact(True).fact_key != _fact(1).fact_key
    assert _cond(True).fact_key != _cond(1).fact_key


def test_fact_identity_preserves_existing_readable_simple_string_keys():
    assert WorldFact(
        predicate="service_running",
        args=["prod"],
        provenance=_prov(),
    ).fact_key == "service_running(prod)"


def test_fact_identity_avoids_string_delimiter_collisions():
    one_arg = WorldFact(predicate="p", args=["a,b"], provenance=_prov())
    two_args = WorldFact(predicate="p", args=["a", "b"], provenance=_prov())
    assert one_arg.fact_key != two_args.fact_key


def test_trusted_snapshot_does_not_merge_int_and_string_facts():
    trusted = normalize_trusted_snapshot(
        [_fact(123, FactTruth.VERIFIED_TRUE), _fact("123", FactTruth.VERIFIED_FALSE)]
    )
    assert len(trusted) == 2
    assert trusted[_fact(123).fact_key].truth == FactTruth.VERIFIED_TRUE
    assert trusted[_fact("123").fact_key].truth == FactTruth.VERIFIED_FALSE


def test_genuinely_identical_typed_facts_still_merge_by_lattice():
    trusted = normalize_trusted_snapshot(
        [_fact(123, FactTruth.VERIFIED_TRUE), _fact(123, FactTruth.VERIFIED_FALSE)]
    )
    assert len(trusted) == 1
    assert trusted[_fact(123).fact_key].truth == FactTruth.CONFLICT


def test_validator_string_fact_cannot_satisfy_integer_precondition():
    plan = PlanIR(
        plan_id="typed-validator",
        goal_description="Typed validator lookup",
        actions=[_action(pre_arg=123)],
    )
    result = EpistemicCausalValidator().validate_plan(
        plan,
        observed_world_state=[_fact("123")],
    )
    assert result.status == ValidationStatus.UNKNOWN


def test_runtime_string_fact_cannot_satisfy_integer_precondition():
    registry = _runtime_registry(123)
    plan = PlanIR(
        plan_id="typed-runtime-int",
        goal_description="String fact cannot satisfy integer precondition",
        actions=[_action(pre_arg=123)],
    )
    session = PlanningSession(session_id="typed-runtime-int")
    session.submit_draft(plan)
    session.validate_candidate(
        1,
        registry,
        validator=_PassValidator(),
        observed_world_state=[_fact("123")],
    )
    session.select_version(1)
    policy_hash = registry.compute_registry_hash()
    cert = session.authorize_selected(registry, policy_hash=policy_hash)
    session.start_execution(
        registry,
        policy_hash=policy_hash,
        current_world_facts=[_fact("123")],
    )

    calls = []

    def backend(argv, *, timeout_seconds):
        calls.append(argv)
        return SandboxExecutionResult(stdout="", stderr="", returncode=0, duration_ms=0.0)

    manager = ExecutionPlanManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[_fact("123")],
        policy_hash=policy_hash,
    )
    summary = manager.execute_authorized_plan(cert, execution_backend=backend)
    assert summary.success is False
    assert calls == []


def test_runtime_integer_fact_cannot_satisfy_string_precondition():
    registry = _runtime_registry("123")
    plan = PlanIR(
        plan_id="typed-runtime-str",
        goal_description="Integer fact cannot satisfy string precondition",
        actions=[_action(pre_arg="123")],
    )
    session = PlanningSession(session_id="typed-runtime-str")
    session.submit_draft(plan)
    session.validate_candidate(
        1,
        registry,
        validator=_PassValidator(),
        observed_world_state=[_fact(123)],
    )
    session.select_version(1)
    policy_hash = registry.compute_registry_hash()
    cert = session.authorize_selected(registry, policy_hash=policy_hash)
    session.start_execution(
        registry,
        policy_hash=policy_hash,
        current_world_facts=[_fact(123)],
    )

    calls = []

    def backend(argv, *, timeout_seconds):
        calls.append(argv)
        return SandboxExecutionResult(stdout="", stderr="", returncode=0, duration_ms=0.0)

    manager = ExecutionPlanManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[_fact(123)],
        policy_hash=policy_hash,
    )
    summary = manager.execute_authorized_plan(cert, execution_backend=backend)
    assert summary.success is False
    assert calls == []


def test_plan_hash_distinguishes_typed_initial_fact_arguments():
    p_int = PlanIR(
        plan_id="same",
        goal_description="same",
        initial_state=[_fact(123)],
    )
    p_str = PlanIR(
        plan_id="same",
        goal_description="same",
        initial_state=[_fact("123")],
    )
    assert p_int.compute_hash() != p_str.compute_hash()


def test_plan_hash_distinguishes_typed_precondition_arguments():
    p_int = PlanIR(plan_id="same", goal_description="same", actions=[_action(pre_arg=123)])
    p_str = PlanIR(plan_id="same", goal_description="same", actions=[_action(pre_arg="123")])
    assert p_int.compute_hash() != p_str.compute_hash()


def test_plan_hash_distinguishes_typed_effect_arguments():
    p_int = PlanIR(plan_id="same", goal_description="same", actions=[_action(effect_arg=123)])
    p_str = PlanIR(plan_id="same", goal_description="same", actions=[_action(effect_arg="123")])
    assert p_int.compute_hash() != p_str.compute_hash()


def test_capability_hash_distinguishes_typed_condition_arguments():
    cap_int = CapabilityEntry(
        name="same",
        description="same",
        positive_effects=[_cond(123)],
    )
    cap_str = CapabilityEntry(
        name="same",
        description="same",
        positive_effects=[_cond("123")],
    )
    assert cap_int.compute_capability_hash() != cap_str.compute_capability_hash()


def test_nested_dict_identity_is_order_independent_and_type_sensitive():
    a = WorldFact(
        predicate="nested",
        args=[{"x": 1, "y": [True, "1"]}],
        provenance=_prov(),
    )
    b = WorldFact(
        predicate="nested",
        args=[{"y": [True, "1"], "x": 1}],
        provenance=_prov(),
    )
    c = WorldFact(
        predicate="nested",
        args=[{"x": "1", "y": [True, "1"]}],
        provenance=_prov(),
    )
    assert a.fact_key == b.fact_key
    assert a.fact_key != c.fact_key
