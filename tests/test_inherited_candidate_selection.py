"""Integration tests for inherited self-verification in the public Prime API."""
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


def test_normal_plan_assess_candidates_inherits_gemini_self_verification(tmp_path):
    session = plan.start(
        "choose implementation",
        plans_dir=tmp_path,
        session_id="inherited-selector",
    )
    verifier = FakeVerifier(selected=1)

    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=verifier,
    )

    assert result["selection_method"] == "inherited-same-model-self-verification"
    assert result["selected_candidate"] == 1
    assert result["generator_model"] == "gemini-3.7-flash"
    assert result["verifier_model"] == "gemini-3.7-flash"
    assert result["is_self_verification"] is True
    assert result["n_evaluations"] == 2
    assert result["pivots"] == 1
    assert verifier.calls[0]["model"] == "gemini-3.7-flash"
    assert verifier.calls[0]["n_evaluations"] == 2
    assert verifier.calls[0]["pivots"] == 1


def test_verifier_outage_inherits_pr1_deterministic_fallback(tmp_path):
    class DownVerifier:
        def select(self, **kwargs):
            raise RuntimeError("provider unavailable")

    session = plan.start(
        "choose with fallback",
        plans_dir=tmp_path,
        session_id="inherited-fallback",
    )
    result = plan.assess_candidates(
        session,
        [DRAFT_A, DRAFT_B],
        plans_dir=tmp_path,
        verifier=DownVerifier(),
    )

    assert result["selection_method"] == "deterministic-fallback"
    assert result["self_verification_available"] is False
    assert "provider unavailable" in result["self_verification_error"]


def test_importing_plan_exposes_inherited_selector_through_plan_mode():
    # The global Prime entrypoint patches the existing PR #1 public symbol, so
    # callers in the same runtime do not need to choose a separate PR #2 API.
    assert plan_mode.assess_candidates is plan.assess_candidates
