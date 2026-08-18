"""Plan-to-execution trace alignment.

A plan is only as good as the execution that discharges it. This module
defines the execution-evidence schema and the deterministic verifier that
matches each plan obligation against a real execution trace.

The verifier specifically catches the "green report / stubbed work" failure:
- declared output files must appear in the trace
- declared functions/classes/variables must appear in the audited symbols
- exit criteria must have a command result with matching stdout content
- executor and verifier provenance must differ when both are declared

Literature:
- AgentRewind (2608.14380): aligned context/environment recovery.
- ACID-Agent (2608.13900): evidence obligations and validated-effect-only commits.
- Capability Sheaves (2608.13228): detect disagreement between components.
- FlowScout (2608.10039): execution feedback drives plan repair.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .causal_validator import PlanParser
from .execution_contract import ExecutionContract, parse_execution_contract


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


def parse_execution_evidence(evidence: str | dict[str, Any] | None) -> Optional[ExecutionEvidence]:
    """Parse execution evidence from a JSON string or dict. Unknown shapes are safe."""
    if evidence is None:
        return None
    data: dict[str, Any] = {}
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
        symbols = {
            str(k): (v if isinstance(v, dict) else {}) for k, v in raw_symbols.items()
        }
        tasks.append(TaskExecution(
            task_id=int(raw.get("task_id") or 0),
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
        except Exception:
            count = 0
        if count < int(expected_count):
            errors.append(f"command {result.command} reported {count} numeric outputs, expected at least {expected_count}")
    return errors


def extract_declared_obligations(plan_text: str) -> dict[str, Any]:
    """Extract outputs and symbols declared by the plan."""
    ast = PlanParser.parse_plan(plan_text)
    contract, _ = parse_execution_contract(plan_text)
    obligations: dict[int, dict[str, Any]] = {}
    for action in ast.actions:
        obligations[action.id] = {
            "outputs": list(action.outputs),
            "inputs": list(action.inputs),
            "symbols": {},
        }
    if contract:
        for path, syms in contract.symbols.items():
            obligations.setdefault(0, {})  # ensure key exists
            for action in ast.actions:
                if path in action.outputs:
                    obligations[action.id]["symbols"][path] = syms
                    break
    return obligations


def align_task_evidence(plan_text: str, evidence: ExecutionEvidence) -> dict[str, Any]:
    """Compare declared plan obligations with an execution trace."""
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


def verify_execution_trace(plan_text: str, evidence: str | dict[str, Any] | None,
                           *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Verify that an execution trace discharges the plan's obligations."""
    parsed = parse_execution_evidence(evidence)
    if parsed is None:
        return {"ok": False, "errors": ["execution evidence missing or unparseable"],
                "warnings": [], "evidence": None, "alignment": None, "exit_criteria": None}

    alignment = align_task_evidence(plan_text, parsed)
    errors = list(alignment["errors"])
    warnings = list(alignment["warnings"])

    # Executor/verifier independence
    if parsed.verifier_agent_id and parsed.agent_id == parsed.verifier_agent_id:
        errors.append("executor and verifier provenance is identical; independent verification required")

    # Exit criteria against recorded command results
    contract, _ = parse_execution_contract(plan_text)
    exit_errors: list[str] = []
    if contract:
        for criterion in contract.raw.get("exit_criteria", []):
            if not isinstance(criterion, dict):
                continue
            task_id = int(criterion.get("task") or criterion.get("task_id") or 0)
            entry = next((t for t in parsed.tasks if t.task_id == task_id), None)
            if entry is None:
                exit_errors.append(f"exit criterion for task {task_id}: no task evidence")
                continue
            result = next((c for c in entry.commands if c.command == _as_str_list(criterion.get("command"))), None)
            if result is None:
                exit_errors.append(f"exit criterion for task {task_id}: command {criterion.get('command')!r} was not recorded")
            else:
                exit_errors.extend(_command_matches(result, criterion))
        errors.extend(exit_errors)

    # Green report contradiction check
    for report in parsed.reports:
        if str(report.get("claim") or "").lower() in ("green", "ok", "pass"):
            for entry in parsed.tasks:
                failed = [c for c in entry.commands if c.exit_code not in (None, 0)]
                if failed:
                    errors.append(f"report claims green but task {entry.task_id} recorded failing commands")
                    break

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "evidence": parsed, "alignment": alignment,
            "exit_criteria": {"ok": not exit_errors, "errors": exit_errors}}


def verify_negative_constraints(plan_text: str,
                                evidence: str | dict[str, Any] | None = None) -> dict[str, Any]:
    """Check plan-declared falsifiers against execution evidence."""
    parsed = parse_execution_evidence(evidence)
    declared = re.findall(r"^\s*-\s*NF-\d+\s*:\s*(.+)$", plan_text, re.M)
    violations: list[str] = []
    checked = len(declared)
    if parsed:
        trace = verify_execution_trace(plan_text, parsed)
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
    return {"ok": not violations, "declared_constraints": declared,
            "checked": checked, "violations": violations}
