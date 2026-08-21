"""Regression tests for bounded/safe plan.search runtime behavior."""
from __future__ import annotations

import asyncio
import time

import pytest

import plan_mode
import plan_mode.search_engine as se


PLAN = """# Goal
Goal: produce b.txt. In scope: local files. Out of scope: network.

## Success criteria
- S1: 1 output exists. Pass/fail. Deadline: within 1 day.

## Tasks
1. Create A. Output: a.txt.
2. Create B. Depends on 1. Inputs: a.txt. Output: b.txt.
"""


@pytest.mark.asyncio
async def test_bare_search_is_local_ast_rules_by_default(tmp_path, monkeypatch):
    session = plan_mode.start("safe default search", plans_dir=tmp_path)
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    async def must_not_call(*args, **kwargs):
        raise AssertionError("bare plan.search must never enter the network proposer")

    monkeypatch.setattr(se, "_legacy_deepseek_propose", must_not_call)
    result = await plan_mode.search(
        session,
        iterations=1,
        width=1,
        skip_if_converged=False,
        plans_dir=tmp_path,
    )

    assert result["mode"] == "ast"
    assert result["expansion"] == "rules"
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_already_converged_search_returns_before_any_llm_setup(tmp_path, monkeypatch):
    session = plan_mode.start("converged search", plans_dir=tmp_path)
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    def high_rollout(plan_text, rubric):
        return {
            "score": 99.87,
            "value": 0.9987,
            "verify_ok": True,
            "sim_ok": True,
            "critiques": [],
        }

    monkeypatch.setattr(se, "_rollout", high_rollout)

    result = await plan_mode.search(
        session,
        iterations=4,
        width=2,
        plans_dir=tmp_path,
    )

    assert result["termination_reason"] == "already-converged"
    assert result["best_score"] == 99.87
    assert result["timed_out"] is False
    assert result["rollouts"] == 0
    assert result["convergence_checks"] == {
        "verify_ok": True,
        "ground_ok": True,
        "sim_ok": True,
    }


@pytest.mark.asyncio
async def test_convergence_requires_grounded_feasibility(tmp_path, monkeypatch):
    session = plan_mode.start("grounded convergence", plans_dir=tmp_path)
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    def high_rollout(plan_text, rubric):
        return {
            "score": 99.87,
            "value": 0.9987,
            "verify_ok": True,
            "sim_ok": True,
            "critiques": [],
        }

    calls = []

    def failed_ground_check(plan_text, *, cwd=None):
        calls.append(cwd)
        return {"ok": False, "verified": [], "missing": ["required.bin"]}

    monkeypatch.setattr(se, "_rollout", high_rollout)
    monkeypatch.setattr(plan_mode, "ground_check", failed_ground_check)

    result = await plan_mode.search(
        session,
        iterations=0,
        width=1,
        cwd=tmp_path,
        plans_dir=tmp_path,
    )

    assert calls
    assert calls[0] == tmp_path
    assert result["termination_reason"] != "already-converged"
    assert result["convergence_checks"]["ground_ok"] is False


@pytest.mark.asyncio
async def test_explicit_llm_search_cannot_silently_choose_deepseek(tmp_path):
    session = plan_mode.start(
        "no silent provider switch",
        plans_dir=tmp_path,
        meta={"implementation_model": "gemini-3.7-flash", "implementation_thinking": "high"},
    )
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    with pytest.raises(ValueError, match="explicit runtime llm_proposer"):
        await plan_mode.search(
            session,
            mode="mcts",
            expansion="llm",
            iterations=1,
            width=1,
            skip_if_converged=False,
            plans_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_llm_search_inherits_same_model_and_thinking(tmp_path):
    session = plan_mode.start(
        "same model thinking search",
        plans_dir=tmp_path,
        meta={"implementation_model": "gemini-3.7-flash", "implementation_thinking": "high"},
    )
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)
    calls = []

    async def proposer(**kwargs):
        calls.append(kwargs)
        return [kwargs["plan_text"] + "\n\n## Verification\nVerify with a deterministic script."], 123

    result = await plan_mode.search(
        session,
        mode="mcts",
        expansion="llm",
        llm_proposer=proposer,
        iterations=1,
        width=1,
        skip_if_converged=False,
        plans_dir=tmp_path,
    )

    assert calls
    assert calls[0]["model"] == "gemini-3.7-flash"
    assert calls[0]["thinking_profile"]["thinking_level"] == "high"
    assert calls[0]["thinking_profile"]["reasoning_effort"] == "high"
    assert result["implementation_model"] == "gemini-3.7-flash"
    assert result["implementation_thinking"] == calls[0]["thinking_profile"]


@pytest.mark.asyncio
async def test_sync_runtime_proposer_is_rejected_before_it_can_block(tmp_path):
    session = plan_mode.start(
        "sync proposer rejection",
        plans_dir=tmp_path,
        meta={"implementation_model": "model-x", "implementation_thinking": "default"},
    )
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    def blocking_sync_proposer(**kwargs):
        time.sleep(5.0)
        return [kwargs["plan_text"]], 0

    started = time.monotonic()
    with pytest.raises(ValueError, match="async"):
        await plan_mode.search(
            session,
            mode="mcts",
            expansion="llm",
            llm_proposer=blocking_sync_proposer,
            iterations=1,
            width=1,
            skip_if_converged=False,
            plans_dir=tmp_path,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_proposal_timeout_falls_back_instead_of_hanging(tmp_path):
    session = plan_mode.start(
        "proposal timeout",
        plans_dir=tmp_path,
        meta={"implementation_model": "model-x", "implementation_thinking": "default"},
    )
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    async def slow_proposer(**kwargs):
        await asyncio.sleep(2.0)
        return [kwargs["plan_text"]], 0

    started = time.monotonic()
    result = await plan_mode.search(
        session,
        mode="mcts",
        expansion="llm",
        llm_proposer=slow_proposer,
        iterations=1,
        width=1,
        proposal_timeout_seconds=0.01,
        search_timeout_seconds=5.0,
        skip_if_converged=False,
        plans_dir=tmp_path,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.5
    assert any("timed out" in warning.lower() for warning in result["warnings"])
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_total_search_deadline_returns_partial_result(tmp_path):
    session = plan_mode.start(
        "total search timeout",
        plans_dir=tmp_path,
        meta={"implementation_model": "model-x", "implementation_thinking": "default"},
    )
    plan_mode.assess(session, PLAN, plans_dir=tmp_path)

    async def very_slow_proposer(**kwargs):
        await asyncio.sleep(2.0)
        return [kwargs["plan_text"]], 0

    started = time.monotonic()
    result = await plan_mode.search(
        session,
        mode="mcts",
        expansion="llm",
        llm_proposer=very_slow_proposer,
        iterations=4,
        width=2,
        proposal_timeout_seconds=5.0,
        search_timeout_seconds=0.05,
        skip_if_converged=False,
        plans_dir=tmp_path,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.5
    assert result["timed_out"] is True
    assert result["termination_reason"] == "search-timeout"
