"""Tests for probabilistic Best-of-N self-verification with a hard deterministic gate."""
from __future__ import annotations

from types import SimpleNamespace

from plan_mode.epistemic_validator import PlanValidationResult, ValidationStatus
from plan_mode.ir import ActionIR, PlanIR, Provenance, SourceType
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.self_verification import PlanSelfVerifier, ProbabilisticSelfVerifier


def _action(action_id: str, capability: str) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability,
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
    )


def _registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(CapabilityEntry(name="noop.a", description="A", executor_command_template=["true"]))
    reg.register(CapabilityEntry(name="noop.b", description="B", executor_command_template=["true"]))
    return reg


def test_same_model_best_of_n_selection_is_advisory_and_certified_only_after_validator_pass():
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(index=1, ranking=[1, 0], scores=[0.25, 0.91])

    verifier = ProbabilisticSelfVerifier(select_fn=fake_select)
    selector = PlanSelfVerifier(registry=_registry(), verifier=verifier)
    candidates = [
        PlanIR(plan_id="p0", goal_description="choose", actions=[_action("a0", "noop.a")]),
        PlanIR(plan_id="p1", goal_description="choose", actions=[_action("a1", "noop.b")]),
    ]

    result = selector.select(
        candidates,
        generator_model="deepseek-v4-flash",
        verifier_model="deepseek-v4-flash",
        n_evaluations=2,
        pivots=1,
    )

    assert result.selected_index == 1
    assert result.is_self_verification is True
    assert result.is_certified is True
    assert result.validation_status == ValidationStatus.PASS
    assert result.ranking == [1, 0]
    assert result.scores == [0.25, 0.91]
    assert calls[0]["n_evaluations"] == 2
    assert calls[0]["pivots"] == 1


def test_deterministic_fail_candidate_is_never_sent_to_llm_selector():
    class Validator:
        def validate_plan(self, plan, **kwargs):
            status = ValidationStatus.FAIL if plan.plan_id == "bad" else ValidationStatus.PASS
            return PlanValidationResult(status=status)

    seen = []

    def fake_select(**kwargs):
        seen.extend(kwargs["candidates"])
        return SimpleNamespace(index=0, ranking=[0], scores=[0.8])

    selector = PlanSelfVerifier(
        registry=_registry(),
        validator=Validator(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
    )
    bad = PlanIR(plan_id="bad", goal_description="x", actions=[_action("bad", "noop.a")])
    good = PlanIR(plan_id="good", goal_description="x", actions=[_action("good", "noop.b")])

    result = selector.select([bad, good], verifier_model="v")
    assert result.selected_index == 1
    assert result.is_certified is True
    assert len(seen) == 1
    assert '"plan_id":"good"' in seen[0].replace(" ", "")


def test_unknown_candidates_may_be_ranked_for_rework_but_never_certified():
    class Validator:
        def validate_plan(self, plan, **kwargs):
            return PlanValidationResult(status=ValidationStatus.UNKNOWN, unknown_facts=["missing(x)"])

    def fake_select(**kwargs):
        return SimpleNamespace(index=1, ranking=[1, 0], scores=[0.1, 0.7])

    selector = PlanSelfVerifier(
        registry=_registry(),
        validator=Validator(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
    )
    plans = [
        PlanIR(plan_id="u0", goal_description="x", actions=[_action("u0", "noop.a")]),
        PlanIR(plan_id="u1", goal_description="x", actions=[_action("u1", "noop.b")]),
    ]
    result = selector.select(plans, verifier_model="v")

    assert result.selected_index == 1
    assert result.validation_status == ValidationStatus.UNKNOWN
    assert result.is_certified is False
    assert result.requires_rework is True


def test_unregistered_action_cannot_be_certified_even_if_verifier_prefers_it():
    def fake_select(**kwargs):
        return SimpleNamespace(index=0, ranking=[0], scores=[0.99])

    selector = PlanSelfVerifier(
        registry=_registry(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
    )
    unknown = PlanIR(
        plan_id="magic",
        goal_description="x",
        actions=[_action("m", "magic.capability")],
    )
    result = selector.select([unknown], verifier_model="v")

    assert result.is_certified is False
    assert result.validation_status == ValidationStatus.UNKNOWN
    assert result.requires_rework is True
