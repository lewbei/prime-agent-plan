"""Tests for execution contracts, probes, command isolation, and symbol audits."""
from __future__ import annotations

import json
import sys

import pytest

import plan_mode
from plan_mode.execution_contract import (
    MAX_CONTRACT_COMMANDS,
    run_command,
    run_verification_commands,
)


@pytest.fixture(autouse=True)
def _explicit_test_dev_execution(monkeypatch):
    """Most legacy unit tests exercise parsing/output semantics, not bwrap.

    Production defaults remain strict; tests that verify the secure boundary
    explicitly remove this opt-in.
    """
    monkeypatch.setenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", "1")


def _plan_with_contract(contract: dict) -> str:
    return (
        "# Goal\nGoal: Build a runner. In scope: runner. Out of scope: docs.\n\n"
        "## Success criteria\n- S1: runner passes. Pass/fail. Deadline: within 1 day.\n\n"
        "## Tasks\n"
        "1. Build. Output: runner.py.\n"
        "2. Verify. Depends on 1. Inputs: runner.py. Output: report.txt.\n\n"
        "## Execution Contract\n```json\n"
        + json.dumps(contract)
        + "\n```\n"
    )


def test_validate_contract_accepts_complete_contract():
    plan = _plan_with_contract({
        "verification_commands": [["python", "-m", "pytest", "-q"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "workspace_invariants": ["git status is clean except declared files"],
        "parity_checks": [{"left": "legacy", "right": "runner", "algorithm": "sha256"}],
        "symbols": {"runner.py": {"functions": ["main"], "variables": ["PROFILES"]}},
    })
    result = plan_mode.validate_execution_contract(plan)
    assert result["ok"], result["errors"]


def test_validate_contract_rejects_missing_commands_and_budgets():
    plan = _plan_with_contract({"symbols": {}})
    result = plan_mode.validate_execution_contract(plan)
    assert not result["ok"]
    assert any("verification command" in e for e in result["errors"])
    assert any("expected-artifact" in e for e in result["errors"])


def test_release_requires_contract_only_when_requested(tmp_path):
    plan = (
        "# Goal\nGoal: x. In scope: x. Out of scope: y.\n\n"
        "## Success criteria\n- S1: 1 test passes. Pass/fail. Deadline: within 1 day.\n\n"
        "## Tasks\n1. A. Output: a.md.\n2. B. Depends on 1. Inputs: a.md. Output: b.md (verifies S1).\n"
    )
    s = plan_mode.start("contract gate", plans_dir=tmp_path, max_rounds=1)
    plan_mode.assess(s, plan, plans_dir=tmp_path)
    gate_soft = plan_mode.release(s, min_score=0, require_judge=False, plans_dir=tmp_path)
    assert gate_soft["ok"] is True
    gate_hard = plan_mode.release(s, min_score=0, require_judge=False,
                                  require_execution_contract=True, plans_dir=tmp_path)
    assert gate_hard["ok"] is False
    assert any("execution contract missing" in p for p in gate_hard["problems"])


def test_invalid_contract_blocks_release_even_when_soft(tmp_path):
    plan = _plan_with_contract({"verification_commands": [], "symbols": {}})
    s = plan_mode.start("invalid contract", plans_dir=tmp_path, max_rounds=1)
    plan_mode.assess(s, plan, plans_dir=tmp_path)
    gate = plan_mode.release(s, min_score=0, require_judge=False, plans_dir=tmp_path)
    assert gate["ok"] is False
    contract_check = next(c for c in gate["checks"] if c["name"] == "execution_contract")
    assert contract_check["ok"] is False


def test_probe_success_and_failure(tmp_path):
    probe_file = tmp_path / "spike.py"
    probe_file.write_text("print('runner=40')\n")
    base = {
        "verification_commands": [["python", "spike.py"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"], "variables": ["PROFILES"]}},
    }
    good = _plan_with_contract({
        **base,
        "probe": {"command": ["python", "spike.py"], "expected_output": "runner=40"},
    })
    res = plan_mode.probe_contract(good, cwd=tmp_path)
    assert res["ok"] is True and res["matched"] is True

    bad = _plan_with_contract({
        **base,
        "probe": {"command": ["python", "spike.py"], "expected_output": "runner=999"},
    })
    res = plan_mode.probe_contract(bad, cwd=tmp_path)
    assert res["ok"] is False


def test_symbol_audit_detects_stub_loophole(tmp_path):
    (tmp_path / "runner.py").write_text("def run_one():\n    return 1\n\nTOKEN = 'x'\n")
    plan = _plan_with_contract({
        "verification_commands": [["python", "-m", "pytest", "-q"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"], "variables": ["PROFILES"]}},
    })
    audit = plan_mode.symbol_audit(plan, cwd=tmp_path)
    assert audit["ok"] is False
    assert "run_one" in audit["files"]["runner.py"]["undeclared_functions"]
    assert "main" in audit["files"]["runner.py"]["missing_functions"]
    assert "TOKEN" in audit["files"]["runner.py"]["undeclared_variables"]


def test_assess_with_contract_and_probe_critiques(tmp_path):
    (tmp_path / "spike.py").write_text("print('ok')\n")
    plan = _plan_with_contract({
        "probe": {"command": ["python", "spike.py"], "expected_output": "ok"},
        "verification_commands": [["python", "spike.py"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"], "variables": ["PROFILES"]}},
    })
    s = plan_mode.start("assess probe", plans_dir=tmp_path)
    result = plan_mode.assess(s, plan, plans_dir=tmp_path,
                              require_execution_contract=True, run_probe=True,
                              probe_cwd=tmp_path)
    assert result["execution_contract"]["ok"] is True
    assert result["probe"]["ok"] is True
    assert not any(c["id"].startswith("mech:contract") for c in result["critiques"])


def test_parity_audit_and_verification_commands(tmp_path):
    plan = _plan_with_contract({
        "verification_commands": [["python", "-c", "print('ok')"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "parity_checks": [{"left": "a.txt", "right": "b.txt", "algorithm": "sha256"}],
        "symbols": {"runner.py": {"functions": ["main"], "variables": ["PROFILES"]}},
    })
    contract = plan_mode.parse_execution_contract(plan)[0]
    (tmp_path / "a.txt").write_text("same")
    (tmp_path / "b.txt").write_text("same")
    parity = plan_mode.parity_audit(contract, cwd=tmp_path)
    assert parity["ok"] is True
    cmds = plan_mode.run_verification_commands(contract, cwd=tmp_path)
    assert cmds["ok"] is True


def test_release_requires_passed_probe(tmp_path):
    (tmp_path / "spike.py").write_text("print('runner=3')\n")
    plan = _plan_with_contract({
        "probe": {"command": ["python", "spike.py"], "expected_output": "runner=40"},
        "verification_commands": [["python", "-c", "print('ok')"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"], "variables": ["PROFILES"]}},
    })
    s = plan_mode.start("probe gate", plans_dir=tmp_path, max_rounds=1)
    result = plan_mode.assess(s, plan, plans_dir=tmp_path,
                              require_execution_contract=True, run_probe=True,
                              probe_cwd=tmp_path)
    assert result["probe"]["ok"] is False
    gate = plan_mode.release(s, min_score=0, require_judge=False,
                             require_execution_contract=True,
                             execution_cwd=tmp_path, plans_dir=tmp_path)
    assert gate["ok"] is False
    assert any("probe" in p.lower() for p in gate["problems"])


def test_contract_defensive_null_and_primitive_handling():
    raw_plan = """# Goal
Goal: Test null handling.

## Tasks
1. Task A. Output: out.txt.

## Execution Contract
```json
{
  "probe": null,
  "verification_commands": null,
  "expected_artifacts": null,
  "workspace_invariants": null,
  "parity_checks": null,
  "symbols": null
}
```
"""
    contract, _ = plan_mode.parse_execution_contract(raw_plan)
    assert contract is not None
    assert contract.probe == {}
    assert contract.verification_commands == []
    assert contract.expected_artifacts == {}
    assert contract.workspace_invariants == []
    assert contract.parity_checks == []
    assert contract.symbols == {}
    val = plan_mode.validate_execution_contract(raw_plan)
    assert val["ok"] is False
    assert any("verification command" in e for e in val["errors"])


def test_contract_indented_markdown_fences():
    raw_plan = """# Goal
Goal: Test indentation.

## Tasks
1. Task A. Output: a.py.

   ## Execution Contract
   ```json
   {
     "verification_commands": [["python", "-V"]],
     "expected_artifacts": {"a.py": {"min_lines": 1}},
     "symbols": {"a.py": {"functions": ["run"]}}
   }
   ```
"""
    contract, _ = plan_mode.parse_execution_contract(raw_plan)
    assert contract is not None
    assert len(contract.verification_commands) == 1
    val = plan_mode.validate_execution_contract(raw_plan)
    assert val["ok"] is True


def test_contract_invalid_budget_integers(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    plan = _plan_with_contract({
        "verification_commands": [["python", "-V"]],
        "expected_artifacts": {"a.py": {"min_bytes": "100KB", "min_lines": "ten"}, "report.txt": {"min_lines": 1}},
        "symbols": {"a.py": {"variables": ["x"]}},
    })
    val = plan_mode.validate_execution_contract(plan, cwd=tmp_path)
    assert val["ok"] is False
    assert any("not a valid integer" in e for e in val["errors"])


def test_probe_missing_binary_no_crash(tmp_path):
    plan = _plan_with_contract({
        "probe": {"command": ["non_existent_binary_xyz_12345", "--version"]},
        "verification_commands": [["python", "-V"]],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"]}},
    })
    probe = plan_mode.probe_contract(plan, cwd=tmp_path)
    assert probe["ok"] is False
    assert any("probe failed" in e for e in probe["errors"])


def test_symbol_audit_ignores_inner_locals_and_class_methods(tmp_path):
    src = """import sys

GLOBAL_VAR = "hello"

class Worker:
    def __init__(self, name):
        self.name = name

    def compute(self):
        temp_val = 42
        return temp_val * 2

def run():
    local_calc = [i * 2 for i in range(5)]
    return len(local_calc)
"""
    (tmp_path / "worker.py").write_text(src)
    plan = _plan_with_contract({
        "verification_commands": [["python", "-V"]],
        "expected_artifacts": {"worker.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"worker.py": {"functions": ["run"], "classes": ["Worker"], "variables": ["GLOBAL_VAR"]}},
    })
    audit = plan_mode.symbol_audit(plan, cwd=tmp_path)
    assert audit["ok"] is True, audit["errors"]


def test_symbol_audit_non_python_artifacts(tmp_path):
    (tmp_path / "config.json").write_text('{"key": "value"}\n')
    plan = _plan_with_contract({
        "verification_commands": [["python", "-V"]],
        "expected_artifacts": {"config.json": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"config.json": {"variables": []}},
    })
    audit = plan_mode.symbol_audit(plan, cwd=tmp_path)
    assert audit["files"]["config.json"].get("non_python") is True


def test_plan_declared_command_cannot_escape_workspace_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS", raising=False)
    outside = tmp_path.parent / "prime_contract_escape.txt"
    outside.unlink(missing_ok=True)
    result = run_command(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(outside)!r}).write_text('escape')"],
        cwd=tmp_path,
        timeout=2.0,
    )
    assert outside.exists() is False
    assert result["ok"] is False
    assert result["isolation"] in {"STRICT_SANDBOX", "REFUSED"}


def test_contract_rejects_unbounded_command_count():
    plan = _plan_with_contract({
        "verification_commands": [["python", "-V"] for _ in range(MAX_CONTRACT_COMMANDS + 1)],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"]}},
    })
    result = plan_mode.validate_execution_contract(plan)
    assert result["ok"] is False
    assert any("maximum" in error for error in result["errors"])


def test_verification_commands_have_total_wall_clock_budget(tmp_path):
    plan = _plan_with_contract({
        "verification_commands": [
            ["python", "-c", "import time; time.sleep(0.05)"],
            ["python", "-c", "import time; time.sleep(0.05)"],
        ],
        "expected_artifacts": {"runner.py": {"min_lines": 1}, "report.txt": {"min_lines": 1}},
        "symbols": {"runner.py": {"functions": ["main"]}},
    })
    contract = plan_mode.parse_execution_contract(plan)[0]
    result = run_verification_commands(
        contract,
        cwd=tmp_path,
        timeout=0.05,
        total_budget_seconds=0.02,
    )
    assert result["ok"] is False
    assert result["budget_exhausted"] is True or any(r.get("timeout") for r in result["results"])
