"""Plan-to-execution trace alignment with optional empirical re-verification.

A supplied JSON trace is a *claim*.  When ``cwd`` is provided, Prime binds that
claim to the live workspace: declared files must exist, symbol contracts are
re-audited, verification commands and exit criteria are re-run through the
strict execution-contract sandbox, and executor/verifier provenance must be
independent.  Only those observations can support a strict release gate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .causal_validator import PlanParser
from .execution_contract import (
    ExecutionContract,
    artifact_audit,
    parse_execution_contract,
    run_exit_criteria,
    run_verification_commands,
    symbol_audit,
)


@dataclass
class CommandResult:
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class TaskExecution:
    task_id: int = 0
    status: str = "done"
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    commands: list[CommandResult] = field(default_factory=list)
    symbols: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    note: str = ""


@dataclass
class ExecutionEvidence:
    agent_id: str = "executor"
    verifier_agent_id: str = ""
    tasks: list[TaskExecution] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _as_str_list(value: Any) -> list[str]:
    return [str(x) for x in value] if isinstance(value, list) else []


def _as_command_result(value: Any) -> CommandResult:
    if not isinstance(value, dict):
        return CommandResult()
    return CommandResult(
        command=_as_str_list(value.get("command")),
        exit_code=value.get("exit_code") if isinstance(value.get("exit_code"), int) else None,
        stdout=str(value.get("stdout") or ""),
        stderr=str(value.get("stderr") or ""),
    )


def parse_execution_evidence(evidence: str | dict[str, Any] | ExecutionEvidence | None) -> Optional[ExecutionEvidence]:
    if isinstance(evidence, ExecutionEvidence):
        return evidence
    if evidence is None:
        return None
    data: dict[str, Any]
    if isinstance(evidence, str):
        text = evidence.strip()
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            text = text[start:end + 1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        data = parsed
    elif isinstance(evidence, dict):
        data = evidence
    else:
        return None

    raw_tasks = data.get("tasks", []) if isinstance(data.get("tasks"), list) else []
    tasks: list[TaskExecution] = []
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            continue
        raw_symbols = raw.get("symbols", {}) if isinstance(raw.get("symbols"), dict) else {}
        symbols = {str(k): (v if isinstance(v, dict) else {}) for k, v in raw_symbols.items()}
        try:
            task_id = int(raw.get("task_id") or 0)
        except (TypeError, ValueError):
            task_id = 0
        tasks.append(TaskExecution(
            task_id=task_id,
            status=str(raw.get("status") or "done"),
            files_created=_as_str_list(raw.get("files_created")),
            files_modified=_as_str_list(raw.get("files_modified")),
            commands=[_as_command_result(c) for c in raw.get("commands", []) if isinstance(c, dict)],
            symbols=symbols,
            note=str(raw.get("note") or ""),
        ))
    reports = [r for r in data.get("reports", []) if isinstance(r, dict)] if isinstance(data.get("reports"), list) else []
    return ExecutionEvidence(
        agent_id=str(data.get("agent_id") or "executor"),
        verifier_agent_id=str(data.get("verifier_agent_id") or ""),
        tasks=tasks,
        reports=reports,
        raw=data,
    )


def _command_matches(result: CommandResult, criterion: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.exit_code != criterion.get("exit_code", 0):
        errors.append(f"command {result.command} exit_code={result.exit_code}, expected {criterion.get('exit_code', 0)}")
    stdout = result.stdout or ""
    for needle in _as_str_list(criterion.get("must_contain")):
        if needle not in stdout:
            errors.append(f"command {result.command} stdout missing {needle!r}")
    expected = criterion.get("expected_stdout")
    if expected is not None and stdout.strip() != str(expected).strip():
        errors.append(f"command {result.command} stdout mismatch: got {stdout[:80]!r}, expected {expected!r}")
    expected_count = criterion.get("expected_count")
    if expected_count is not None:
        try:
            count = len(re.findall(r"\d+(?:\.\d+)?", stdout))
            if count < int(expected_count):
                errors.append(f"command {result.command} reported {count} numeric outputs, expected at least {expected_count}")
        except (TypeError, ValueError):
            errors.append(f"command {result.command} has invalid expected_count={expected_count!r}")
    return errors


def extract_declared_obligations(plan_text: str) -> dict[str, Any]:
    ast_tree = PlanParser.parse_plan(plan_text)
    contract, _ = parse_execution_contract(plan_text)
    obligations: dict[int, dict[str, Any]] = {}
    for action in ast_tree.actions:
        obligations[action.id] = {
            "outputs": list(action.outputs),
            "inputs": list(action.inputs),
            "symbols": {},
        }
    if contract:
        for path, syms in contract.symbols.items():
            for action in ast_tree.actions:
                if path in action.outputs:
                    obligations.setdefault(action.id, {"outputs": [], "inputs": [], "symbols": {}})
                    obligations[action.id]["symbols"][path] = syms
                    break
    return obligations


def align_task_evidence(plan_text: str, evidence: ExecutionEvidence) -> dict[str, Any]:
    obligations = extract_declared_obligations(plan_text)
    by_task = {t.task_id: t for t in evidence.tasks}
    errors: list[str] = []
    warnings: list[str] = []
    matches: list[dict[str, Any]] = []

    for task_id in sorted(obligations):
        ob = obligations[task_id]
        if not ob.get("outputs"):
            continue
        entry = by_task.get(task_id)
        if entry is None:
            errors.append(f"task {task_id}: no execution evidence entry")
            continue
        if entry.status != "done":
            errors.append(f"task {task_id}: status is {entry.status!r}, expected done")
        produced = set(entry.files_created) | set(entry.files_modified)
        for out in ob["outputs"]:
            if out not in produced:
                errors.append(f"task {task_id}: declared output {out!r} is missing from execution trace")
        for path, declared in ob["symbols"].items():
            actual = entry.symbols.get(path)
            if not actual:
                errors.append(f"task {task_id}: no symbol audit evidence for {path}")
                continue
            exp_f = set(_as_str_list(declared.get("functions")))
            exp_c = set(_as_str_list(declared.get("classes")))
            exp_v = set(_as_str_list(declared.get("variables")))
            act_f = set(_as_str_list(actual.get("functions")))
            act_c = set(_as_str_list(actual.get("classes")))
            act_v = set(_as_str_list(actual.get("variables")))
            missing_f = exp_f - act_f
            missing_c = exp_c - act_c
            missing_v = exp_v - (act_v | act_c)
            if missing_f:
                errors.append(f"task {task_id}: {path} missing functions {sorted(missing_f)}")
            if missing_c:
                errors.append(f"task {task_id}: {path} missing classes {sorted(missing_c)}")
            if missing_v:
                errors.append(f"task {task_id}: {path} missing variables {sorted(missing_v)}")
            undeclared = (act_f - exp_f) | (act_c - exp_c) | (act_v - exp_v)
            if undeclared:
                warnings.append(f"task {task_id}: {path} has undeclared symbols {sorted(undeclared)}")
        matches.append({"task_id": task_id, "matched": True})
    return {"ok": not errors, "errors": errors, "warnings": warnings, "matches": matches}


def _trace_exit_errors(contract: ExecutionContract, parsed: ExecutionEvidence) -> list[str]:
    errors: list[str] = []
    for criterion in contract.exit_criteria:
        task_id = int(criterion.get("task") or criterion.get("task_id") or 0)
        entry = next((t for t in parsed.tasks if t.task_id == task_id), None)
        if entry is None:
            errors.append(f"exit criterion for task {task_id}: no task evidence")
            continue
        expected_command = _as_str_list(criterion.get("command"))
        result = next((c for c in entry.commands if c.command == expected_command), None)
        if result is None:
            errors.append(f"exit criterion for task {task_id}: command {criterion.get('command')!r} was not recorded")
        else:
            errors.extend(_command_matches(result, criterion))
    return errors


def _workspace_reverify(plan_text: str, contract: ExecutionContract, cwd: Path) -> dict[str, Any]:
    """Independently re-observe the workspace instead of trusting trace claims."""
    errors: list[str] = []

    artifacts = artifact_audit(contract, cwd=cwd)
    if not artifacts["ok"]:
        errors.extend(f"[workspace artifact] {e}" for e in artifacts["errors"])

    symbols = symbol_audit(plan_text, cwd=cwd)
    if contract.symbols and not symbols["ok"]:
        errors.extend(f"[workspace symbol] {e}" for e in symbols["errors"])

    verification = run_verification_commands(contract, cwd=cwd)
    if contract.verification_commands and not verification["ok"]:
        errors.extend(f"[workspace verification] {e}" for e in verification.get("errors", []))

    exit_criteria = run_exit_criteria(contract, cwd=cwd)
    if contract.exit_criteria and not exit_criteria["ok"]:
        errors.extend(f"[workspace exit] {e}" for e in exit_criteria.get("errors", []))

    obligations = extract_declared_obligations(plan_text)
    actual_outputs: dict[str, bool] = {}
    for ob in obligations.values():
        for raw in ob.get("outputs", []):
            p = Path(raw) if Path(raw).is_absolute() else cwd / raw
            actual_outputs[raw] = p.exists()
            if not p.exists():
                errors.append(f"[workspace output] declared output {raw!r} does not exist")

    return {
        "ok": not errors,
        "errors": errors,
        "artifact_audit": artifacts,
        "symbol_audit": symbols,
        "verification_commands": verification,
        "exit_criteria": exit_criteria,
        "actual_outputs": actual_outputs,
    }


def verify_execution_trace(
    plan_text: str,
    evidence: str | dict[str, Any] | ExecutionEvidence | None,
    *,
    cwd: str | Path | None = None,
    require_independent_verifier: bool | None = None,
) -> dict[str, Any]:
    """Verify trace consistency and, when ``cwd`` is supplied, empirical truth."""
    parsed = parse_execution_evidence(evidence)
    if parsed is None:
        return {
            "ok": False,
            "errors": ["execution evidence missing or unparseable"],
            "warnings": [],
            "evidence": None,
            "alignment": None,
            "exit_criteria": None,
            "workspace_reverification": None,
        }

    alignment = align_task_evidence(plan_text, parsed)
    errors = list(alignment["errors"])
    warnings = list(alignment["warnings"])

    strict_provenance = (cwd is not None) if require_independent_verifier is None else bool(require_independent_verifier)
    if strict_provenance and not parsed.verifier_agent_id:
        errors.append("independent verifier provenance is missing")
    elif parsed.verifier_agent_id and parsed.agent_id == parsed.verifier_agent_id:
        errors.append("executor and verifier provenance is identical; independent verification required")

    contract, _ = parse_execution_contract(plan_text)
    trace_exit_errors: list[str] = []
    if contract:
        trace_exit_errors = _trace_exit_errors(contract, parsed)
        errors.extend(trace_exit_errors)

    for report in parsed.reports:
        if str(report.get("claim") or "").lower() in ("green", "ok", "pass"):
            if any(c.exit_code not in (None, 0) for entry in parsed.tasks for c in entry.commands):
                errors.append("report claims green but execution trace contains a failing command")
                break

    workspace = None
    if cwd is not None:
        if contract is None:
            errors.append("workspace re-verification requires an execution contract")
        else:
            workspace = _workspace_reverify(plan_text, contract, Path(cwd).resolve())
            errors.extend(workspace["errors"])

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "evidence": parsed,
        "alignment": alignment,
        "exit_criteria": {"ok": not trace_exit_errors, "errors": trace_exit_errors},
        "workspace_reverification": workspace,
    }


def verify_negative_constraints(
    plan_text: str,
    evidence: str | dict[str, Any] | ExecutionEvidence | None = None,
    *,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    parsed = parse_execution_evidence(evidence)
    declared = re.findall(r"^\s*-\s*NF-\d+\s*:\s*(.+)$", plan_text, re.M)
    violations: list[str] = []
    checked = len(declared)
    if parsed:
        trace = verify_execution_trace(plan_text, parsed, cwd=cwd, require_independent_verifier=False)
        if not trace["ok"]:
            violations.extend(trace["errors"])
        for nf in declared:
            lower = nf.lower()
            if "same agent" in lower and parsed.agent_id == parsed.verifier_agent_id and parsed.verifier_agent_id:
                violations.append(f"NF violation: {nf}")
            if "report says green" in lower or "green while" in lower:
                for report in parsed.reports:
                    claim = str(report.get("claim") or "").lower()
                    if claim in ("green", "ok", "pass"):
                        failed = any(c.exit_code not in (None, 0) for t in parsed.tasks for c in t.commands)
                        if failed:
                            violations.append(f"NF violation: {nf}")
    return {
        "ok": not violations,
        "declared_constraints": declared,
        "checked": checked,
        "violations": violations,
    }
