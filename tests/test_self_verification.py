"""Tests for inherent same-model, same-thinking probabilistic self-verification."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from plan_mode.epistemic_validator import PlanValidationResult, ValidationStatus
from plan_mode.ir import ActionIR, PlanIR, Provenance, SourceType
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.self_verification import (
    DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    DEFAULT_SELF_VERIFICATION_PIVOTS,
    PlanSelfVerifier,
    ProbabilisticSelfVerifier,
    SelfVerificationUnavailableError,
    _wrap_client_for_thinking,
    normalize_thinking_profile,
    resolve_implementation_model,
    resolve_implementation_thinking,
)


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


def _plans() -> list[PlanIR]:
    return [
        PlanIR(plan_id="p0", goal_description="choose", actions=[_action("a0", "noop.a")]),
        PlanIR(plan_id="p1", goal_description="choose", actions=[_action("a1", "noop.b")]),
    ]


@pytest.mark.parametrize(
    ("model", "thinking"),
    [
        ("gemini-3.7-flash", "high"),
        ("deepseek-v4-flash", "high"),
        ("model-x", "medium"),
    ],
)
def test_select_inherits_same_model_and_same_thinking(model, thinking):
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(index=1, ranking=[1, 0], scores=[0.25, 0.91])

    selector = PlanSelfVerifier(
        registry=_registry(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
        implementation_model=model,
        implementation_thinking=thinking,
    )
    result = selector.select(_plans())

    expected = normalize_thinking_profile(thinking)
    assert result.selected_index == 1
    assert result.generator_model == model
    assert result.verifier_model == model
    assert result.generator_thinking == expected
    assert result.verifier_thinking == expected
    assert result.is_self_verification is True
    assert result.is_same_thinking is True
    assert result.is_certified is True
    assert result.validation_status == ValidationStatus.PASS
    assert calls[0]["model"] == model
    assert calls[0]["thinking_profile"] == expected
    assert calls[0]["n_evaluations"] == 2
    assert calls[0]["pivots"] == 1


def test_exact_thinking_budget_is_preserved_for_both_roles():
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(index=0, ranking=[0, 1], scores=[0.9, 0.2])

    selector = PlanSelfVerifier(
        registry=_registry(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
        implementation_model="gemini-2.5-flash",
        implementation_thinking={"thinking_budget": 8192, "max_tokens": 16000},
    )
    result = selector.select(_plans())
    expected = {"thinking_budget": 8192, "max_tokens": 16000, "mode": "budget"}
    assert result.generator_thinking == expected
    assert result.verifier_thinking == expected
    assert calls[0]["thinking_profile"] == expected


def test_resolution_prefers_runtime_then_session_then_environment():
    session = {
        "meta": {
            "implementation_model": "session-model",
            "implementation_thinking": "medium",
        }
    }
    assert resolve_implementation_model("explicit-model", session=session, environ={}) == "explicit-model"
    assert resolve_implementation_model(session=session, environ={}) == "session-model"
    assert resolve_implementation_model(environ={"PRIME_IMPLEMENTATION_MODEL": "env-model"}) == "env-model"

    assert resolve_implementation_thinking("high", session=session, environ={}) == normalize_thinking_profile("high")
    assert resolve_implementation_thinking(session=session, environ={}) == normalize_thinking_profile("medium")
    assert resolve_implementation_thinking(environ={"PRIME_IMPLEMENTATION_THINKING": "low"}) == normalize_thinking_profile("low")


def test_missing_thinking_metadata_means_same_provider_default_not_a_new_profile():
    assert resolve_implementation_thinking(environ={}) == {"mode": "default"}


def test_no_model_identity_does_not_silently_choose_a_verifier_model():
    def fake_select(**kwargs):
        raise AssertionError("backend should not be called without a model identity")

    selector = PlanSelfVerifier(
        registry=_registry(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
    )
    with pytest.raises(SelfVerificationUnavailableError, match="implementation-model identity"):
        selector.select(_plans())


def test_different_verifier_model_is_rejected():
    selector = PlanSelfVerifier(
        registry=_registry(),
        verifier=ProbabilisticSelfVerifier(select_fn=lambda **_: None),
        implementation_model="model-a",
        implementation_thinking="high",
    )
    with pytest.raises(ValueError, match="verifier_model == implementation model"):
        selector.select(_plans(), verifier_model="model-b")


def test_verification_settings_keep_paper_matched_k_and_pivots():
    assert DEFAULT_SELF_VERIFICATION_N_EVALUATIONS == 2
    assert DEFAULT_SELF_VERIFICATION_PIVOTS == 1


class _DummyCreate:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return "ok"


class _DummyCompletions:
    def __init__(self):
        self._create = _DummyCreate()

    def create(self, **kwargs):
        return self._create.create(**kwargs)


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


class _DummyOpenAIClient:
    def __init__(self, deepseek=False):
        self.chat = _DummyChat()
        if deepseek:
            self._llm_verifier_deepseek = True


def test_openai_proxy_replaces_upstream_disable_with_same_thinking_level():
    client = _DummyOpenAIClient()
    wrapped = _wrap_client_for_thinking(
        client,
        normalize_thinking_profile("high"),
        "gemini-3.7-flash",
    )
    wrapped.chat.completions.create(
        model="gemini-3.7-flash",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    sent = client.chat.completions._create.kwargs
    assert sent["reasoning_effort"] == "high"
    assert "enable_thinking" not in sent.get("extra_body", {}).get("chat_template_kwargs", {})


def test_deepseek_proxy_replaces_upstream_high_with_actual_implementation_level():
    client = _DummyOpenAIClient(deepseek=True)
    wrapped = _wrap_client_for_thinking(
        client,
        normalize_thinking_profile("low"),
        "deepseek-v4-flash",
    )
    wrapped.chat.completions.create(
        model="deepseek-v4-flash",
        extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    )
    sent = client.chat.completions._create.kwargs
    assert sent["extra_body"]["thinking"] == {"type": "enabled"}
    assert sent["extra_body"]["reasoning_effort"] == "low"


def test_deterministic_fail_candidate_is_never_sent_to_llm_selector():
    class Validator:
        def validate_plan(self, plan, **kwargs):
            return PlanValidationResult(
                status=ValidationStatus.FAIL if plan.plan_id == "bad" else ValidationStatus.PASS
            )

    seen = []

    def fake_select(**kwargs):
        seen.extend(kwargs["candidates"])
        return SimpleNamespace(index=0, ranking=[0], scores=[0.8])

    selector = PlanSelfVerifier(
        registry=_registry(),
        validator=Validator(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
        implementation_model="implementation-model",
        implementation_thinking="high",
    )
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

    selector = PlanSelfVerifier(
        registry=_registry(),
        validator=Validator(),
        verifier=ProbabilisticSelfVerifier(select_fn=fake_select),
        implementation_model="implementation-model",
        implementation_thinking="medium",
    )
    result = selector.select(_plans())

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
        implementation_model="implementation-model",
        implementation_thinking="high",
    )
    unknown = PlanIR(
        plan_id="magic",
        goal_description="x",
        actions=[_action("m", "magic.capability")],
    )
    result = selector.select([unknown])

    assert result.is_certified is False
    assert result.validation_status == ValidationStatus.UNKNOWN
    assert result.requires_rework is True
