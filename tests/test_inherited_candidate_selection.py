"""Integration tests for inherent self-verification in the public Prime API."""
from __future__ import annotations

from types import SimpleNamespace

import plan
import plan_mode


DRAFT_A = """# Goal
Goal: choose candidate A.

## Success criteria
- S1: produce one output file. Pass/fail: output exists. Deadline: within 1 day.

## Tasks
1. Create A. Depends on: none. Output: a.md. Exit criterion: file exists. Time: 1 min.
"""

DRAFT_B = """# Goal
Goal: choose candidate B.

## Success criteria
- S1: produce one output file. Pass/fail: output exists. Deadline: within 1 day.

## Tasks
1. Create B. Depends on: none. Output: b.md. Exit criterion: file exists. Time: 1 min.
"""


class FakeVerifier:
    def __init__(self, selected=1):
        self.selected = selected
        self.calls = []

    def select(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            selected_index=self.selected,
            ranking=[self.selected, 1 - self.selected],
            scores=[0.2, 0.9] if self.selected == 1 else [0.9, 0.2],
        )


def test_normal_assess_candidates_inherits_session_model_and_thinking(tmp_path):
    session = plan.start(
        "choose implementation",
        plans_dir=tmp_path,
        session_id="inherited-selector",
        meta={
            "implementation_model": "deepseek-v4-flash",
            "implementation_thinking": "high",
        },
    )
    verifier = FakeVerifier(selected=1)

    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=verifier,
    )

    expected_thinking = {
        "mode": "level",
        "thinking_level": "high",
        "reasoning_effort": "high",
    }
    assert result["selection_method"] == "inherited-same-model-same-thinking-self-verification"
    assert result["selected_candidate"] == 1
    assert result["implementation_model"] == "deepseek-v4-flash"
    assert result["generator_model"] == "deepseek-v4-flash"
    assert result["verifier_model"] == "deepseek-v4-flash"
    assert result["implementation_thinking"] == expected_thinking
    assert result["generator_thinking"] == expected_thinking
    assert result["verifier_thinking"] == expected_thinking
    assert result["is_self_verification"] is True
    assert result["is_same_thinking"] is True
    assert verifier.calls[0]["model"] == "deepseek-v4-flash"
    assert verifier.calls[0]["thinking_profile"] == expected_thinking
    assert verifier.calls[0]["n_evaluations"] == 2
    assert verifier.calls[0]["pivots"] == 1


def test_explicit_runtime_model_and_thinking_are_inherited_without_defaults(tmp_path):
    session = plan.start(
        "choose runtime model",
        plans_dir=tmp_path,
        session_id="runtime-selector",
    )
    verifier = FakeVerifier(selected=0)

    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=verifier,
        implementation_model="gemini-3.7-flash",
        implementation_thinking="medium",
    )

    assert result["implementation_model"] == "gemini-3.7-flash"
    assert result["verifier_model"] == "gemini-3.7-flash"
    assert result["implementation_thinking"]["thinking_level"] == "medium"
    assert result["verifier_thinking"] == result["implementation_thinking"]
    assert verifier.calls[0]["model"] == "gemini-3.7-flash"
    assert verifier.calls[0]["thinking_profile"] == result["implementation_thinking"]


def test_absent_thinking_override_inherits_same_provider_default(tmp_path):
    session = plan.start(
        "choose default-thinking runtime",
        plans_dir=tmp_path,
        session_id="default-thinking",
        meta={"implementation_model": "model-default"},
    )
    verifier = FakeVerifier(selected=0)
    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=verifier,
    )
    assert result["generator_thinking"] == {"mode": "default"}
    assert result["verifier_thinking"] == {"mode": "default"}
    assert verifier.calls[0]["thinking_profile"] == {"mode": "default"}


def test_unknown_model_identity_falls_back_without_calling_verifier(tmp_path):
    class MustNotRun:
        def select(self, **kwargs):
            raise AssertionError("verifier must not run when implementation model is unknown")

    session = plan.start(
        "choose unknown runtime",
        plans_dir=tmp_path,
        session_id="unknown-model",
    )
    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=MustNotRun(),
    )

    assert result["selection_method"] == "deterministic-fallback-no-model"
    assert result["self_verification_available"] is False
    assert "identity unavailable" in result["self_verification_error"]


def test_verifier_outage_inherits_pr1_deterministic_fallback(tmp_path):
    class DownVerifier:
        def select(self, **kwargs):
            raise RuntimeError("provider unavailable")

    session = plan.start(
        "choose with fallback",
        plans_dir=tmp_path,
        session_id="inherited-fallback",
        meta={
            "implementation_model": "model-z",
            "implementation_thinking": "low",
        },
    )
    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=DownVerifier(),
    )

    assert result["selection_method"] == "deterministic-fallback"
    assert result["implementation_model"] == "model-z"
    assert result["implementation_thinking"]["thinking_level"] == "low"
    assert result["self_verification_available"] is False
    assert "provider unavailable" in result["self_verification_error"]


def test_importing_plan_exposes_inherited_selector_through_plan_mode():
    assert plan_mode.assess_candidates is plan.assess_candidates
