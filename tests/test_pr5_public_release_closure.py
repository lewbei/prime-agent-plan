"""Second-audit regressions for public release atomicity."""
from __future__ import annotations

import json

import plan_mode


def _state(tmp_path, session_id: str):
    state = {
        "session_id": session_id,
        "plans_dir": str(tmp_path),
        "rounds": [{
            "version": 1,
            "plan_text": "1. Build. Output: out.txt.\n",
            "score": 100.0,
            "critiques": [],
        }],
        "best_version": 1,
        "best_score": 100.0,
        "status": "converged",
        "judge_log": [],
    }
    plan_mode._save_session(tmp_path, state)
    return state


def test_public_release_failed_cwd_precheck_never_invokes_raw_release(tmp_path, monkeypatch):
    state = _state(tmp_path, "public-release-precheck")
    calls = []

    def dangerous_raw(*args, **kwargs):
        calls.append(True)
        state["committed_version"] = 1
        plan_mode._save_session(tmp_path, state)
        return {"ok": True, "checks": [], "problems": []}

    monkeypatch.setattr(plan_mode, "_raw_release", dangerous_raw)
    monkeypatch.setattr(
        plan_mode,
        "ground_check",
        lambda *args, **kwargs: {
            "ok": False,
            "missing": ["forced execution workspace mismatch"],
            "verified": [],
        },
    )
    monkeypatch.setattr(
        plan_mode,
        "simulate",
        lambda *args, **kwargs: {
            "executable_plan": False,
            "problems": ["forced simulation failure"],
        },
    )

    gate = plan_mode.release(
        state,
        min_score=0,
        require_judge=False,
        execution_cwd=tmp_path,
        plans_dir=tmp_path,
    )

    assert gate["ok"] is False
    assert calls == []
    assert state.get("committed_version") is None
    persisted = json.loads((tmp_path / "public-release-precheck.json").read_text())
    assert persisted.get("committed_version") is None


def test_public_release_restores_commit_if_raw_gate_returns_failure(tmp_path, monkeypatch):
    state = _state(tmp_path, "public-release-rollback")

    monkeypatch.setattr(
        plan_mode,
        "ground_check",
        lambda *args, **kwargs: {"ok": True, "missing": [], "verified": []},
    )
    monkeypatch.setattr(
        plan_mode,
        "simulate",
        lambda *args, **kwargs: {"executable_plan": True, "problems": []},
    )

    def dangerous_raw(*args, **kwargs):
        state.update({
            "committed_version": 1,
            "committed_score": 100.0,
            "committed_at": "unsafe",
            "committed_plan_hash": "unsafe",
        })
        plan_mode._save_session(tmp_path, state)
        return {
            "ok": False,
            "checks": [{"name": "forced", "ok": False}],
            "problems": ["forced raw failure"],
        }

    monkeypatch.setattr(plan_mode, "_raw_release", dangerous_raw)
    gate = plan_mode.release(
        state,
        min_score=0,
        require_judge=False,
        execution_cwd=tmp_path,
        plans_dir=tmp_path,
    )

    assert gate["ok"] is False
    for key in (
        "committed_version",
        "committed_score",
        "committed_at",
        "committed_plan_hash",
    ):
        assert key not in state
    persisted = json.loads((tmp_path / "public-release-rollback.json").read_text())
    assert persisted.get("committed_version") is None
    assert persisted.get("committed_plan_hash") is None
