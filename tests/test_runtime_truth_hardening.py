"""Adversarial regressions for public runtime truth/safety boundaries."""
from __future__ import annotations

import sys

import pytest

import plan
import plan_mode
import plan_mode.self_verification as sv
from plan_mode.runtime.sandbox import ExecutionSandbox, SecurityProfile


def _contract_plan() -> str:
    return """# Goal
Goal: build runner and report. In scope: local files. Out of scope: network.

## Success criteria
- S1: 1 verification passes. Pass/fail. Deadline: within 1 day.

## Tasks
1. Build runner. Output: runner.py.
2. Write report. Depends on 1. Inputs: runner.py. Output: report.txt.

## Execution Contract
```json
{
  "verification_commands": [["python", "-c", "print('ok')"]],
  "expected_artifacts": {
    "runner.py": {"min_lines": 2},
    "report.txt": {"min_lines": 1}
  },
  "symbols": {
    "runner.py": {"functions": ["main"], "variables": ["PROFILES"]}
  }
}
```
"""


def _claimed_evidence() -> dict:
    return {
        "agent_id": "executor-a",
        "verifier_agent_id": "verifier-b",
        "tasks": [
            {
                "task_id": 1,
                "status": "done",
                "files_created": ["runner.py"],
                "symbols": {
                    "runner.py": {
                        "functions": ["main"],
                        "classes": [],
                        "variables": ["PROFILES"],
                    }
                },
                "commands": [],
            },
            {
                "task_id": 2,
                "status": "done",
                "files_created": ["report.txt"],
                "symbols": {},
                "commands": [],
            },
        ],
        "reports": [{"claim": "green"}],
    }


def test_fabricated_trace_cannot_pass_live_workspace_reverification(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", "1")
    result = plan_mode.verify_execution_trace(
        _contract_plan(),
        _claimed_evidence(),
        cwd=tmp_path,
    )
    assert result["ok"] is False
    assert result["workspace_reverification"] is not None
    assert any(
        "workspace artifact" in error or "workspace output" in error
        for error in result["errors"]
    )


def test_execution_trace_passes_only_when_live_workspace_agrees(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", "1")
    (tmp_path / "runner.py").write_text("PROFILES = 40\ndef main():\n    return PROFILES\n")
    (tmp_path / "report.txt").write_text("ok\n")
    result = plan_mode.verify_execution_trace(
        _contract_plan(),
        _claimed_evidence(),
        cwd=tmp_path,
    )
    assert result["ok"] is True, result["errors"]
    live = result["workspace_reverification"]
    assert live["artifact_audit"]["artifacts"]["runner.py"]["sha256"]
    assert live["symbol_audit"]["ok"] is True
    assert live["verification_commands"]["ok"] is True


def test_strict_execution_evidence_requires_independent_verifier_id(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", "1")
    (tmp_path / "runner.py").write_text("PROFILES = 40\ndef main():\n    return PROFILES\n")
    (tmp_path / "report.txt").write_text("ok\n")
    evidence = _claimed_evidence()
    evidence["verifier_agent_id"] = ""
    result = plan_mode.verify_execution_trace(_contract_plan(), evidence, cwd=tmp_path)
    assert result["ok"] is False
    assert any("verifier provenance" in error for error in result["errors"])


def test_self_verifier_call_budget_fails_before_backend_invocation():
    called = []

    def fake_select(**kwargs):
        called.append(kwargs)
        raise AssertionError("backend must not run after call-budget rejection")

    verifier = sv.ProbabilisticSelfVerifier(select_fn=fake_select)
    with pytest.raises(sv.SelfVerificationUnavailableError, match="exceeding budget"):
        verifier.select(
            problem="choose",
            candidates=[f"candidate-{i}" for i in range(7)],
            model="model-x",
            thinking_profile="high",
            max_verifier_calls=128,
        )
    assert called == []


def test_self_verifier_client_factory_receives_exact_model_family(monkeypatch):
    seen = []
    sentinel = object()

    def fake_factory(model, timeout):
        seen.append((model, timeout))
        return sentinel

    monkeypatch.setattr(sv, "_create_model_family_client", fake_factory)
    out = sv.ProbabilisticSelfVerifier._create_upstream_client(
        "gemini-3.7-flash", 17.0
    )
    assert out is sentinel
    assert seen == [("gemini-3.7-flash", 17.0)]
    assert sv._model_family("gemini-3.7-flash") == "gemini"
    assert sv._model_family("deepseek-v4-flash") == "deepseek"


@pytest.mark.asyncio
async def test_public_execute_plan_rejects_missing_handlers():
    text = "1. Build. Output: a.txt.\n2. Test. Depends on 1. Output: b.txt.\n"
    result = await plan.execute_plan(text, task_handlers={})
    assert result["ok"] is False
    assert "missing task handlers" in result["error"]
    assert result["executed_tasks"] == []


@pytest.mark.asyncio
async def test_public_execute_plan_rejects_sync_handler_before_invocation():
    text = "1. Build. Output: a.txt.\n"
    invoked = []

    def blocking_handler(ctx):
        invoked.append(True)
        return {"done": True}

    result = await plan.execute_plan(text, task_handlers={1: blocking_handler})
    assert result["ok"] is False
    assert "synchronous task handlers" in result["error"]
    assert invoked == []


@pytest.mark.asyncio
async def test_public_execute_plan_keeps_async_transactional_path():
    text = "1. Build. Output: a.txt.\n"

    async def handler(ctx):
        return {"done": True}

    result = await plan.execute_plan(text, task_handlers={1: handler}, timeout_per_task=1.0)
    assert result["ok"] is True
    assert result["executed_tasks"] == [1]


def test_public_assess_reports_dirty_max_round_stop_as_plateaued(tmp_path):
    session = plan.start("dirty convergence regression", plans_dir=tmp_path, max_rounds=1)
    result = plan.assess(
        session,
        "# Goal\nGoal: x.\n\n## Tasks\n1. Maybe do something. Inputs: missing.bin. Output: out.txt.\n",
        plans_dir=tmp_path,
    )
    assert result["status"] == "plateaued"
    assert result["clean_convergence"] is False
    assert result["requires_revision"] is True
    assert session["status"] == "plateaued"


@pytest.mark.asyncio
async def test_sandbox_pipeline_supplies_stdin_before_waiting(tmp_path):
    sandbox = ExecutionSandbox(
        policy=SecurityProfile.get_profile("PERMISSIVE_DEV").model_copy(
            update={"workspace_dir": str(tmp_path)}
        )
    )
    pipeline = [
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read() + '!')"],
    ]
    result = sandbox.execute_argv_pipeline(
        pipeline,
        cwd=str(tmp_path),
        input_data="abc",
        timeout_seconds=1.0,
    )
    assert result.returncode == 0, result.stderr
    assert result.timeout_exceeded is False
    assert result.stdout == "ABC!"


@pytest.mark.asyncio
async def test_legacy_judge_does_not_silently_select_deepseek(monkeypatch):
    monkeypatch.delenv("PLAN_JUDGE_MODEL", raising=False)
    from plan_mode.judge_client import judge

    result = await judge("# Goal\nGoal: x")
    assert result["ok"] is False
    assert "will not silently choose DeepSeek" in result["error"]
