"""Execution-trace snapshot isolation and runtime-attestation regressions."""
from __future__ import annotations

import hashlib
import json
import sys

import plan_mode
from plan_mode.runtime_closure_context import workspace_identity


def _plan() -> str:
    contract = {
        "verification_commands": [[
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "assert Path('out.txt').read_text(encoding='utf-8') == 'ok'; "
                "Path('mutation.txt').write_text('snapshot-only', encoding='utf-8')"
            ),
        ]],
        "expected_artifacts": {
            "out.txt": {"min_bytes": 1},
        },
        "exit_criteria": [],
    }
    return (
        "# Goal\n"
        "Goal: verify one artifact without mutating the source workspace.\n\n"
        "## Tasks\n"
        "1. Build artifact. Output: out.txt.\n\n"
        "## Execution Contract\n"
        "```json\n"
        f"{json.dumps(contract)}\n"
        "```\n"
    )


def _evidence(**extra):
    return {
        "agent_id": "executor-a",
        "verifier_agent_id": "verifier-b",
        "tasks": [{
            "task_id": 1,
            "status": "done",
            "files_created": ["out.txt"],
        }],
        **extra,
    }


def test_trace_verification_runs_on_copy_and_attests_result(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "out.txt").write_text("ok", encoding="utf-8")
    workspace_id = workspace_identity(workspace)
    plan_text = _plan()

    result = plan_mode.verify_execution_trace(
        plan_text,
        _evidence(),
        cwd=workspace,
        require_independent_verifier=True,
        expected_session_id="session-a",
        expected_workspace_identity=workspace_id,
    )

    assert result["ok"] is True, result["errors"]
    assert not (workspace / "mutation.txt").exists()
    snapshot = result["workspace_reverification"]["verification_snapshot"]
    assert snapshot["isolated_copy"] is True
    plan_hash = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
    assert plan_mode._verify_execution_trace_runtime_attestation(
        result,
        plan_hash=plan_hash,
        session_id="session-a",
        workspace_id=workspace_id,
        certificate_id=None,
    ) is True


def test_trace_rejects_caller_binding_to_another_session(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "out.txt").write_text("ok", encoding="utf-8")

    result = plan_mode.verify_execution_trace(
        _plan(),
        _evidence(session_id="wrong-session"),
        cwd=workspace,
        expected_session_id="expected-session",
        expected_workspace_identity=workspace_identity(workspace),
    )

    assert result["ok"] is False
    assert any("another session" in error for error in result["errors"])
    assert "runtime_attestation" not in result


def test_strict_release_refuses_unattested_trace_before_raw_release(
    tmp_path,
    monkeypatch,
):
    state = {
        "session_id": "strict-trace-release",
        "plans_dir": str(tmp_path),
        "rounds": [{
            "version": 1,
            "plan_text": _plan(),
            "score": 100.0,
            "critiques": [],
        }],
        "best_version": 1,
        "best_score": 100.0,
        "status": "converged",
        "judge_log": [],
    }
    plan_mode._save_session(tmp_path, state)
    calls = []

    def dangerous_raw(*args, **kwargs):
        calls.append(True)
        state["committed_version"] = 1
        plan_mode._save_session(tmp_path, state)
        return {"ok": True, "checks": [], "problems": []}

    monkeypatch.setattr(plan_mode, "_raw_release", dangerous_raw)
    monkeypatch.setattr(
        plan_mode,
        "verify_execution_trace",
        lambda *args, **kwargs: {
            "ok": True,
            "errors": [],
            "warnings": [],
            "evidence": None,
            "alignment": None,
            "exit_criteria": None,
            "workspace_reverification": None,
        },
    )

    gate = plan_mode.release(
        state,
        min_score=0,
        require_judge=False,
        execution_cwd=tmp_path,
        execution_evidence=_evidence(),
        require_execution_evidence=True,
        plans_dir=tmp_path,
    )

    assert gate["ok"] is False
    assert calls == []
    assert state.get("committed_version") is None
    assert gate["execution_trace_checks"]["runtime_attested"] is False
