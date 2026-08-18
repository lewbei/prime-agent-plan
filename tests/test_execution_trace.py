"""Regression tests for execution-trace alignment and negative constraints."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

import plan_mode
from plan_mode.ast_search import apply_execution_feedback
from plan_mode.causal_validator import PlanParser
from plan_mode.execution_contract import parse_execution_contract, run_exit_criteria


def _contract(extra=None):
    data = {
        "verification_commands": [[sys.executable, "-c", "print('ok')"]],
        "expected_artifacts": {
            "runner.py": {"min_lines": 1},
            "report.txt": {"min_lines": 1},
        },
        "symbols": {
            "runner.py": {
                "functions": ["main", "run_profile"],
                "variables": ["PROFILES"],
            }
        },
    }
    if extra:
        data.update(extra)
    return data


def _plan(contract):
    return (
        "# Goal\nGoal: Build runner. In scope: x. Out of scope: y.\n\n"
        "## Success criteria\n- S1: 1 test passes. Pass/fail. Deadline: within 1 day.\n\n"
        "## Tasks\n"
        "1. Build. Output: runner.py.\n"
        "2. Report. Depends on 1. Inputs: runner.py. Output: report.txt (verifies S1).\n\n"
        "## Execution Contract\n```json\n"
        + json.dumps(contract)
        + "\n```\n"
    )


def test_execution_trace_rejects_stub_with_missing_symbols_and_output():
    plan = _plan(_contract())
    evidence = {
        "agent_id": "executor-1",
        "verifier_agent_id": "verifier-2",
        "tasks": [
            {
                "task_id": 1,
                "status": "done",
                "files_created": ["stub.py"],
                "symbols": {"runner.py": {"functions": ["placeholder"], "variables": []}},
                "commands": [],
            },
            {
                "task_id": 2,
                "status": "done",
                "files_created": ["report.txt"],
                "commands": [],
                "symbols": {},
            },
        ],
    }
    result = plan_mode.verify_execution_trace(plan, evidence)
    assert result["ok"] is False
    assert any("runner.py" in e for e in result["errors"])
    assert any("missing functions" in e for e in result["errors"])


def test_exit_criteria_reject_green_but_wrong_stdout(tmp_path):
    (tmp_path / "runner.py").write_text("print('3 profiles')\n")
    plan = _plan(_contract({
        "exit_criteria": [
            {
                "task": 1,
                "command": [sys.executable, "runner.py"],
                "exit_code": 0,
                "must_contain": ["profile_40"],
            }
        ]
    }))
    contract = parse_execution_contract(plan)[0]
    result = run_exit_criteria(contract, cwd=tmp_path)
    assert result["ok"] is False
    assert any("stdout missing" in e for e in result["errors"])


def test_execution_feedback_mutates_only_failing_task():
    plan = (
        "# Goal\nGoal: x.\n\n## Tasks\n"
        "1. A. Output: a.md.\n"
        "2. B. Depends on 1. Inputs: a.md. Output: b.md.\n"
        "3. C. Depends on 2. Inputs: b.md. Output: c.md.\n"
    )
    ast = PlanParser.parse_plan(plan)
    original = {a.id: a.name for a in ast.actions}
    repaired = apply_execution_feedback(ast, [{
        "task_id": 2,
        "missing_outputs": ["b.md"],
        "missing_symbols": {"b.py": ["main"]},
        "detail": "stdout was 3 profiles, expected 40",
    }])
    by_id = {a.id: a for a in repaired.actions}
    assert "repair" in by_id[2].name
    assert by_id[1].name == original[1]
    assert by_id[3].name == original[3]
    assert by_id[2].parameters["execution_feedback"]["task_id"] == 2


def test_negative_constraints_detect_same_agent_and_green_report():
    plan = (
        _plan(_contract())
        + "\n## Negative Constraints\n"
        "- NF-1: executor and verifier are the same agent id.\n"
        "- NF-2: report says green while a declared verification command failed.\n"
    )
    evidence = {
        "agent_id": "same-agent",
        "verifier_agent_id": "same-agent",
        "tasks": [
            {
                "task_id": 1,
                "status": "done",
                "files_created": ["runner.py"],
                "symbols": {"runner.py": {"functions": ["main", "run_profile"], "variables": ["PROFILES"]}},
                "commands": [{"command": [sys.executable, "-c", "print('ok')"], "exit_code": 0, "stdout": "ok"}],
            },
            {
                "task_id": 2,
                "status": "done",
                "files_created": ["report.txt"],
                "symbols": {},
                "commands": [{"command": ["false"], "exit_code": 1, "stdout": "", "stderr": "fail"}],
            },
        ],
        "reports": [{"claim": "green", "author": "same-agent"}],
    }
    result = plan_mode.verify_negative_constraints(plan, evidence)
    assert result["ok"] is False
    assert any("same agent" in v.lower() or "green" in v.lower() for v in result["violations"])


def test_assess_emits_trace_critique_when_evidence_is_stub(tmp_path):
    plan = _plan(_contract())
    evidence = {
        "agent_id": "executor-1",
        "verifier_agent_id": "verifier-2",
        "tasks": [{"task_id": 1, "status": "done", "files_created": ["stub.py"]}],
    }
    session = plan_mode.start("trace assess", plans_dir=tmp_path)
    result = plan_mode.assess(
        session,
        plan,
        plans_dir=tmp_path,
        require_execution_contract=True,
        execution_evidence=evidence,
    )
    assert result["execution_trace"]["ok"] is False
    assert any(c["id"].startswith("mech:trace:") for c in result["critiques"])
