"""Tests for inherited probabilistic Best-of-N self-verification."""
from __future__ import annotations

from types import SimpleNamespace

from plan_mode.epistemic_validator import PlanValidationResult, ValidationStatus
from plan_mode.ir import ActionIR, PlanIR, Provenance, SourceType
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.self_verification import (
    DEFAULT_SELF_VERIFICATION_MODEL,
    DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    DEFAULT_SELF_VERIFICATION_PIVOTS,
    PlanSelfVerifier,
    ProbabilisticSelfVerifier,
)


def _action(action_id: str, capability: str) -> ActionIR:
    return ActionIR(action_id=action_id, capability_name=capability,
                    provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE))


def _registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(CapabilityEntry(name="noop.a", description="A", executor_command_template=["true"]))
    reg.register(CapabilityEntry(name="noop.b", description="B", executor_command_template=["true"]))
    return reg


def _plans() -> list[PlanIR]:
    return [
        PlanIR(plan_id="p0", goal_description="choose", actions=[_action("a0", "noop.a")]),
        PlanIR(plan_id="p1", goal_description="choose", actions=[_action("a1", "noop.b")]),
    ]


def test_default_select_inherits_gemini_same_model_verification():
    calls = []
    def fake_select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(index=1, ranking=[1, 0], scores=[0.25, 0.91])
    selector = PlanSelfVerifier(registry=_registry(), verifier=ProbabilisticSelfVerifier(select_fn=fake_select))
    result = selector.select(_plans())
    assert result.selected_index == 1
    assert result.generator_model == result.verifier_model == "gemini-3.7-flash"
    assert result.is_self_verification is True
    assert result.is_certified is True
    assert result.validation_status == ValidationStatus.PASS
    assert calls[0]["model"] == "gemini-3.7-flash"
    assert calls[0]["n_evaluations"] == 2
    assert calls[0]["pivots"] == 1


def test_inherited_defaults_match_prime_model_and_paper_settings():
    def fake_select(**kwargs):
        return SimpleNamespace(index=0, ranking=[0, 1], scores=[0.9, 0.2])
    selector = PlanSelfVerifier(registry=_registry(), verifier=ProbabilisticSelfVerifier(select_fn=fake_select))
    result = selector.select(_plans())
    assert DEFAULT_SELF_VERIFICATION_MODEL == "gemini-3.7-flash"
    assert DEFAULT_SELF_VERIFICATION_N_EVALUATIONS == 2
    assert DEFAULT_SELF_VERIFICATION_PIVOTS == 1
    assert result.n_evaluations == 2
    assert result.pivots == 1


def test_compatibility_alias_does_not_change_default_semantics():
    def fake_select(**kwargs):
        return SimpleNamespace(index=0, ranking=[0, 1], scores=[0.9, 0.2])
    selector = PlanSelfVerifier(registry=_registry(), verifier=ProbabilisticSelfVerifier(select_fn=fake_select))
    normal = selector.select(_plans())
    alias = selector.select_same_model(_plans())
    assert normal.generator_model == alias.generator_model == DEFAULT_SELF_VERIFICATION_MODEL
    assert normal.verifier_model == alias.verifier_model == DEFAULT_SELF_VERIFICATION_MODEL
    assert normal.is_self_verification is alias.is_self_verification is True


def test_deterministic_fail_candidate_is_never_sent_to_llm_selector():
    class Validator:
        def validate_plan(self, plan, **kwargs):
            return PlanValidationResult(status=ValidationStatus.FAIL if plan.plan_id == "bad" else ValidationStatus.PASS)
    seen = []
    def fake_select(**kwargs):
        seen.extend(kwargs["candidates"])
        return SimpleNamespace(index=0, ranking=[0], scores=[0.8])
    selector = PlanSelfVerifier(registry=_registry(), validator=Validator(),
                                verifier=ProbabilisticSelfVerifier(select_fn=fake_select))
    bad = PlanIR(plan_id="bad", goal_description="x", actions=[_action("bad", "noop.a")])
    good = PlanIR(plan_id="good", goal_description="x", actions=[_action("good", "noop.b")])
    result = selector.select([bad, good])
    assert result.selected_index == 1
    assert result.is_certified is True
    assert len(seen) == 1
    assert '\"plan_id\":\"good\"' in seen[0].replace(" ", "")


def test_unknown_candidates_may_be_ranked_for_rework_but_never_certified():
    class Validator:
        def validate_plan(self, plan, **kwargs):
            return PlanValidationResult(status=ValidationStatus.UNKNOWN, unknown_facts=["missing(x)"])
    def fake_select(**kwargs):
        return SimpleNamespace(index=1, ranking=[1, 0], scores=[0.1, 0.7])
    selector = PlanSelfVerifier(registry=_registry(), validator=Validator(),
                                verifier=ProbabilisticSelfVerifier(select_fn=fake_select))
    result = selector.select(_plans())
    assert result.selected_index == 1
    assert result.validation_status == ValidationStatus.UNKNOWN
    assert result.is_certified is False
    assert result.requires_rework is True


def test_unregistered_action_cannot_be_certified_even_if_verifier_prefers_it():
    def fake_select(**kwargs):
        return SimpleNamespace(index=0, ranking=[0], scores=[0.99])
    selector = PlanSelfVerifier(registry=_registry(), verifier=ProbabilisticSelfVerifier(select_fn=fake_select))
    unknown = PlanIR(plan_id="magic", goal_description="x", actions=[_action("m", "magic.capability")])
    result = selector.select([unknown])
    assert result.is_certified is False
    assert result.validation_status == ValidationStatus.UNKNOWN
    assert result.requires_rework is True
