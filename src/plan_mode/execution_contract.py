"""Execution contracts: bind plan claims to bounded, isolated execution evidence.

Plan-declared commands are untrusted input.  Probes, verification commands and
exit criteria therefore run through Prime's STRICT sandbox by default.  The
only raw-host escape hatch is the explicit development environment variable
``PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS=1``; production never silently falls
back when kernel isolation is unavailable.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .causal_validator import PlanParser

MAX_CONTRACT_COMMANDS = 16
DEFAULT_TOTAL_COMMAND_BUDGET_SECONDS = 120.0
UNISOLATED_DEV_ENV = "PLAN_ALLOW_UNISOLATED_CONTRACT_COMMANDS"


@dataclass
class ExecutionContract:
    probe: dict[str, Any] = field(default_factory=dict)
    verification_commands: list[list[str]] = field(default_factory=list)
    expected_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    workspace_invariants: list[str] = field(default_factory=list)
    parity_checks: list[dict[str, str]] = field(default_factory=list)
    symbols: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    exit_criteria: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def parse_execution_contract(plan_text: str) -> tuple[Optional[ExecutionContract], list[str]]:
    """Extract the JSON execution contract from markdown safely."""
    errors: list[str] = []
    candidates: list[str] = []
    pattern1 = r"##\s*Execution\s+Contract.*?\n[ \t]*```json\s*\n(.*?)\n[ \t]*```"
    m = re.search(pattern1, plan_text, re.I | re.S)
    if m:
        candidates.append(m.group(1))
    pattern2 = r"[ \t]*```json\s*\n(.*?)\n[ \t]*```"
    for block in re.finditer(pattern2, plan_text, re.I | re.S):
        payload = block.group(1)
        if "verification_commands" in payload or "probe" in payload or "symbols" in payload:
            candidates.append(payload)
    if not candidates:
        return None, []

    data: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                data = parsed
                break
        except json.JSONDecodeError as exc:
            errors.append(f"execution contract JSON is invalid: {exc}")
    if data is None:
        return None, errors or ["execution contract JSON is invalid"]

    raw_probe = data.get("probe")
    raw_cmds = data.get("verification_commands")
    raw_artifacts = data.get("expected_artifacts")
    raw_invariants = data.get("workspace_invariants")
    raw_parity = data.get("parity_checks")
    raw_symbols = data.get("symbols")
    raw_exit = data.get("exit_criteria")

    return ExecutionContract(
        probe=raw_probe if isinstance(raw_probe, dict) else {},
        verification_commands=[cmd for cmd in raw_cmds if isinstance(cmd, list)] if isinstance(raw_cmds, list) else [],
        expected_artifacts={str(k): (v if isinstance(v, dict) else {}) for k, v in raw_artifacts.items()} if isinstance(raw_artifacts, dict) else {},
        workspace_invariants=[str(x) for x in raw_invariants] if isinstance(raw_invariants, list) else [],
        parity_checks=[p for p in raw_parity if isinstance(p, dict)] if isinstance(raw_parity, list) else [],
        symbols={str(k): (v if isinstance(v, dict) else {}) for k, v in raw_symbols.items()} if isinstance(raw_symbols, dict) else {},
        exit_criteria=[c for c in raw_exit if isinstance(c, dict)] if isinstance(raw_exit, list) else [],
        raw=data,
    ), errors


def validate_execution_contract(plan_text: str, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Validate contract shape and artifact budgets without executing commands."""
    contract, parse_errors = parse_execution_contract(plan_text)
    errors = list(parse_errors)
    if contract is None:
        return {
            "ok": False,
            "errors": ["execution contract missing; add `## Execution Contract` with a JSON block"],
            "contract": None,
        }

    if not contract.verification_commands:
        errors.append("execution contract must contain at least one verification command")
    if len(contract.verification_commands) > MAX_CONTRACT_COMMANDS:
        errors.append(
            f"execution contract has {len(contract.verification_commands)} verification commands; "
            f"maximum is {MAX_CONTRACT_COMMANDS}"
        )
    if len(contract.exit_criteria) > MAX_CONTRACT_COMMANDS:
        errors.append(
            f"execution contract has {len(contract.exit_criteria)} exit criteria; maximum is {MAX_CONTRACT_COMMANDS}"
        )
    for cmd in contract.verification_commands:
        if not cmd or not isinstance(cmd[0], str) or not all(isinstance(x, str) for x in cmd):
            errors.append(f"invalid verification command: {cmd!r}")
            break

    ast_tree = PlanParser.parse_plan(plan_text)
    declared_outputs = {out for action in ast_tree.actions for out in action.outputs}
    if declared_outputs:
        missing = sorted(out for out in declared_outputs if out not in contract.expected_artifacts)
        if missing:
            errors.append(f"declared task outputs have no expected-artifact budgets: {missing}")

    base = Path(cwd or Path.cwd())
    for path, budget in contract.expected_artifacts.items():
        p = Path(path) if Path(path).is_absolute() else base / path
        if p.exists():
            size = p.stat().st_size
            if budget.get("min_bytes") is not None:
                try:
                    min_bytes = int(budget["min_bytes"])
                    if size < min_bytes:
                        errors.append(f"{path}: actual {size} bytes < min_bytes {min_bytes}")
                except (ValueError, TypeError):
                    errors.append(f"{path}: min_bytes budget '{budget.get('min_bytes')}' is not a valid integer")
            if budget.get("min_lines") is not None:
                try:
                    min_lines = int(budget["min_lines"])
                    try:
                        lines = len(p.read_text(encoding="utf-8").splitlines())
                    except Exception:
                        lines = 0
                    if lines < min_lines:
                        errors.append(f"{path}: actual {lines} lines < min_lines {min_lines}")
                except (ValueError, TypeError):
                    errors.append(f"{path}: min_lines budget '{budget.get('min_lines')}' is not a valid integer")

    for path, syms in contract.symbols.items():
        functions = syms.get("functions", []) if isinstance(syms, dict) else []
        classes = syms.get("classes", []) if isinstance(syms, dict) else []
        variables = syms.get("variables", []) if isinstance(syms, dict) else []
        # Non-Python artifacts may legitimately have an empty symbol contract.
        if Path(path).suffix == ".py" and not functions and not classes and not variables:
            errors.append(f"symbol contract for {path} must declare at least one function, class, or variable")

    return {
        "ok": not errors,
        "errors": errors,
        "contract": contract,
        "declared_outputs": sorted(declared_outputs),
    }


def _symbols_from_source(source: str) -> dict[str, set[str]]:
    tree = ast.parse(source)
    funcs: set[str] = set()
    classes: set[str] = set()
    vars_: set[str] = set()

    def _extract_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            vars_.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _extract_target(elt)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _extract_target(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            _extract_target(node.target)
        elif isinstance(node, ast.NamedExpr):
            _extract_target(node.target)
    return {"functions": funcs, "classes": classes, "variables": vars_}


def scan_symbols(paths: list[str] | tuple[str, ...] | set[str], *,
                 cwd: str | Path | None = None) -> dict[str, dict[str, Any]]:
    base = Path(cwd or Path.cwd())
    out: dict[str, dict[str, Any]] = {}
    for raw in paths:
        p = Path(raw) if Path(raw).is_absolute() else base / raw
        if not p.exists():
            out[raw] = {"functions": [], "classes": [], "variables": [], "missing": True}
            continue
        if p.suffix != ".py":
            out[raw] = {"functions": [], "classes": [], "variables": [], "non_python": True}
            continue
        try:
            found = _symbols_from_source(p.read_text(encoding="utf-8"))
            out[raw] = {k: sorted(v) for k, v in found.items()}
        except SyntaxError as exc:
            out[raw] = {"functions": [], "classes": [], "variables": [], "syntax_error": str(exc)}
    return out


def symbol_audit(plan_text: str, *, cwd: str | Path | None = None) -> dict[str, Any]:
    contract, parse_errors = parse_execution_contract(plan_text)
    if contract is None:
        return {"ok": False, "errors": parse_errors or ["execution contract missing"], "files": {}}

    actual = scan_symbols(list(contract.symbols.keys()), cwd=cwd)
    files: dict[str, Any] = {}
    errors: list[str] = []
    for path, declared in contract.symbols.items():
        found = actual.get(path, {})
        if found.get("missing"):
            errors.append(f"{path}: declared symbol file is missing")
            files[path] = {"missing": True}
            continue
        if found.get("non_python"):
            files[path] = {"non_python": True}
            continue
        if found.get("syntax_error"):
            errors.append(f"{path}: syntax error: {found['syntax_error']}")
            files[path] = found
            continue
        expected_funcs = {str(x) for x in declared.get("functions", [])} if isinstance(declared, dict) else set()
        expected_classes = {str(x) for x in declared.get("classes", [])} if isinstance(declared, dict) else set()
        expected_vars = {str(x) for x in declared.get("variables", [])} if isinstance(declared, dict) else set()
        actual_funcs = set(found.get("functions", []))
        actual_classes = set(found.get("classes", []))
        actual_vars = set(found.get("variables", []))
        missing_funcs = sorted(expected_funcs - actual_funcs)
        missing_classes = sorted(expected_classes - actual_classes)
        missing_vars = sorted(expected_vars - (actual_vars | actual_classes))
        undeclared_funcs = sorted(actual_funcs - expected_funcs)
        undeclared_classes = sorted(actual_classes - (expected_classes | expected_vars))
        undeclared_vars = sorted(actual_vars - (expected_vars | expected_classes))
        files[path] = {
            "missing_functions": missing_funcs,
            "missing_classes": missing_classes,
            "missing_variables": missing_vars,
            "undeclared_functions": undeclared_funcs,
            "undeclared_classes": undeclared_classes,
            "undeclared_variables": undeclared_vars,
            "actual_functions": sorted(actual_funcs),
            "actual_classes": sorted(actual_classes),
            "actual_variables": sorted(actual_vars),
        }
        if missing_funcs:
            errors.append(f"{path}: missing declared functions: {missing_funcs}")
        if missing_classes:
            errors.append(f"{path}: missing declared classes: {missing_classes}")
        if missing_vars:
            errors.append(f"{path}: missing declared variables: {missing_vars}")
        if undeclared_funcs:
            errors.append(f"{path}: undeclared functions not listed in contract: {undeclared_funcs}")
        if undeclared_classes:
            errors.append(f"{path}: undeclared classes not listed in contract: {undeclared_classes}")
        if undeclared_vars:
            errors.append(f"{path}: undeclared variables not listed in contract: {undeclared_vars}")
    return {"ok": not errors, "errors": errors, "files": files, "contract": contract}


def artifact_audit(contract: ExecutionContract, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Empirically verify declared artifact existence, budgets and SHA-256 hashes."""
    base = Path(cwd or Path.cwd())
    errors: list[str] = []
    artifacts: dict[str, Any] = {}
    for raw, budget in contract.expected_artifacts.items():
        p = Path(raw) if Path(raw).is_absolute() else base / raw
        if not p.exists() or not p.is_file():
            errors.append(f"{raw}: expected artifact is missing")
            artifacts[raw] = {"exists": False}
            continue
        data = p.read_bytes()
        info: dict[str, Any] = {
            "exists": True,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        try:
            info["lines"] = len(data.decode("utf-8").splitlines())
        except Exception:
            info["lines"] = None
        if budget.get("min_bytes") is not None:
            try:
                minimum = int(budget["min_bytes"])
                if len(data) < minimum:
                    errors.append(f"{raw}: actual {len(data)} bytes < min_bytes {minimum}")
            except (TypeError, ValueError):
                errors.append(f"{raw}: invalid min_bytes budget")
        if budget.get("min_lines") is not None:
            try:
                minimum = int(budget["min_lines"])
                lines = info["lines"] if isinstance(info["lines"], int) else 0
                if lines < minimum:
                    errors.append(f"{raw}: actual {lines} lines < min_lines {minimum}")
            except (TypeError, ValueError):
                errors.append(f"{raw}: invalid min_lines budget")
        artifacts[raw] = info
    return {"ok": not errors, "errors": errors, "artifacts": artifacts}


def parse_exit_criteria(plan_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    contract, errors = parse_execution_contract(plan_text)
    if contract is None:
        return [], errors or ["execution contract missing"]
    return contract.exit_criteria, []


def validate_exit_criteria(plan_text: str) -> dict[str, Any]:
    criteria, errors = parse_exit_criteria(plan_text)
    if errors:
        return {"ok": False, "errors": errors, "criteria": []}
    problems: list[str] = []
    if len(criteria) > MAX_CONTRACT_COMMANDS:
        problems.append(f"too many exit criteria: {len(criteria)} > {MAX_CONTRACT_COMMANDS}")
    for criterion in criteria:
        cmd = criterion.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            problems.append(f"invalid exit criterion command: {cmd!r}")
        if criterion.get("must_contain") is not None and not isinstance(criterion.get("must_contain"), list):
            problems.append("must_contain must be a list of strings")
        if criterion.get("expected_count") is not None and not isinstance(criterion.get("expected_count"), int):
            problems.append("expected_count must be an integer")
    return {"ok": not problems, "errors": problems, "criteria": criteria}


def _dev_unisolated_allowed() -> bool:
    return os.environ.get(UNISOLATED_DEV_ENV, "").strip().lower() in {"1", "true", "yes"}


def _run_unisolated_dev(cmd: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
    """Explicit development-only compatibility path; never selected silently."""
    resolved = list(cmd)
    if resolved and resolved[0] in ("python", "python3"):
        resolved[0] = sys.executable
    try:
        proc = subprocess.run(resolved, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "timeout": False,
            "isolation": "UNISOLATED_DEV_EXPLICIT",
        }
    except subprocess.TimeoutExpired:
        return {"command": cmd, "exit_code": None, "ok": False, "stdout": "", "stderr": "timeout", "timeout": True, "isolation": "UNISOLATED_DEV_EXPLICIT"}
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return {"command": cmd, "exit_code": None, "ok": False, "stdout": "", "stderr": f"command execution failed: {exc}", "timeout": False, "isolation": "UNISOLATED_DEV_EXPLICIT"}


def run_command(cmd: list[str], *, cwd: str | Path | None = None,
                timeout: float = 60.0) -> dict[str, Any]:
    """Run an untrusted plan-declared argv in a strict sandbox.

    The function fails closed when Bubblewrap is unavailable.  Raw host
    execution requires the explicit development opt-in environment variable.
    """
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
        return {"command": cmd, "exit_code": None, "ok": False, "stdout": "", "stderr": "invalid argv", "timeout": False, "isolation": "REFUSED"}
    base = Path(cwd or Path.cwd()).resolve()
    if _dev_unisolated_allowed():
        return _run_unisolated_dev(cmd, cwd=base, timeout=float(timeout))

    resolved = list(cmd)
    if resolved[0] in ("python", "python3"):
        resolved[0] = sys.executable
    try:
        from .runtime.sandbox import ExecutionSandbox, SecurityProfile
        policy = SecurityProfile.get_profile("STRICT").model_copy(
            update={"workspace_dir": str(base)}
        )
        sandbox = ExecutionSandbox(policy=policy)
        result = sandbox.execute_argv_pipeline(
            [resolved],
            cwd=str(base),
            timeout_seconds=float(timeout),
        )
        return {
            "command": cmd,
            "exit_code": result.returncode,
            "ok": result.returncode == 0,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "timeout": bool(result.timeout_exceeded),
            "isolation": "STRICT_SANDBOX",
        }
    except Exception as exc:
        return {
            "command": cmd,
            "exit_code": None,
            "ok": False,
            "stdout": "",
            "stderr": f"strict sandbox execution refused/failed: {exc}",
            "timeout": False,
            "isolation": "REFUSED",
        }


def _bounded_command_timeout(started: float, total_budget_seconds: float, per_command_timeout: float) -> float:
    remaining = total_budget_seconds - (time.monotonic() - started)
    return max(0.0, min(float(per_command_timeout), remaining))


def run_exit_criteria(contract: ExecutionContract, *, cwd: str | Path | None = None,
                      timeout: float = 60.0,
                      total_budget_seconds: float = DEFAULT_TOTAL_COMMAND_BUDGET_SECONDS) -> dict[str, Any]:
    """Run exit criteria under a count cap and a total wall-clock budget."""
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    criteria = contract.exit_criteria
    if len(criteria) > MAX_CONTRACT_COMMANDS:
        return {"ok": False, "errors": [f"too many exit criteria: {len(criteria)} > {MAX_CONTRACT_COMMANDS}"], "results": [], "total": len(criteria), "passed": 0, "budget_exhausted": False}
    started = time.monotonic()
    budget_exhausted = False
    for criterion in criteria:
        cmd = criterion.get("command")
        if not isinstance(cmd, list) or not cmd:
            errors.append(f"invalid criterion: {criterion!r}")
            results.append({"criterion": criterion, "ok": False})
            continue
        call_timeout = _bounded_command_timeout(started, total_budget_seconds, timeout)
        if call_timeout <= 0:
            budget_exhausted = True
            errors.append("exit-criteria total execution budget exhausted")
            break
        run = run_command(cmd, cwd=cwd, timeout=call_timeout)
        criterion_errors: list[str] = []
        if run.get("ok") is False:
            criterion_errors.append(f"exit_code={run.get('exit_code')}, stderr={(run.get('stderr') or '')[:160]}")
        stdout = run.get("stdout") or ""
        for needle in (criterion.get("must_contain") or []):
            if str(needle) not in stdout:
                criterion_errors.append(f"stdout missing {needle!r}")
        if criterion.get("expected_stdout") is not None and stdout.strip() != str(criterion["expected_stdout"]).strip():
            criterion_errors.append(f"stdout mismatch: {stdout[:80]!r}")
        expected_count = criterion.get("expected_count")
        if isinstance(expected_count, int) and len(re.findall(r"\d+(?:\.\d+)?", stdout)) < expected_count:
            criterion_errors.append(f"fewer than {expected_count} numeric outputs")
        ok = not criterion_errors
        if not ok:
            errors.append(f"criterion {cmd!r} failed: {'; '.join(criterion_errors)}")
        results.append({"criterion": criterion, "run": run, "ok": ok, "errors": criterion_errors})
    return {
        "ok": not errors and not budget_exhausted and len(results) == len(criteria),
        "errors": errors,
        "results": results,
        "total": len(criteria),
        "passed": sum(bool(r.get("ok")) for r in results),
        "budget_exhausted": budget_exhausted,
    }


def run_verification_commands(contract: ExecutionContract, *, cwd: str | Path | None = None,
                              timeout: float = 60.0,
                              total_budget_seconds: float = DEFAULT_TOTAL_COMMAND_BUDGET_SECONDS) -> dict[str, Any]:
    """Run verification commands with strict isolation and aggregate bounds."""
    commands = contract.verification_commands
    if len(commands) > MAX_CONTRACT_COMMANDS:
        return {"ok": False, "results": [], "total": len(commands), "passed": 0, "errors": [f"too many verification commands: {len(commands)} > {MAX_CONTRACT_COMMANDS}"], "budget_exhausted": False}
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    started = time.monotonic()
    budget_exhausted = False
    for cmd in commands:
        call_timeout = _bounded_command_timeout(started, total_budget_seconds, timeout)
        if call_timeout <= 0:
            budget_exhausted = True
            errors.append("verification-command total execution budget exhausted")
            break
        result = run_command(cmd, cwd=cwd, timeout=call_timeout)
        results.append(result)
        if not result.get("ok"):
            errors.append(f"verification command {cmd!r} failed: {(result.get('stderr') or '')[:160]}")
    return {
        "ok": not errors and not budget_exhausted and len(results) == len(commands),
        "results": results,
        "total": len(commands),
        "passed": sum(bool(r.get("ok")) for r in results),
        "errors": errors,
        "budget_exhausted": budget_exhausted,
    }


def parity_audit(contract: ExecutionContract, *, cwd: str | Path | None = None) -> dict[str, Any]:
    base = Path(cwd or Path.cwd())
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for check in contract.parity_checks:
        left = Path(str(check.get("left", "")))
        right = Path(str(check.get("right", "")))
        left_p = left if left.is_absolute() else base / left
        right_p = right if right.is_absolute() else base / right
        if not left_p.exists() or not right_p.exists():
            errors.append(f"parity files missing: {left} / {right}")
            results.append({"left": str(left), "right": str(right), "ok": False})
            continue
        algo = (check.get("algorithm") or "sha256").lower()
        if algo == "sha256":
            lh = hashlib.sha256(left_p.read_bytes()).hexdigest()
            rh = hashlib.sha256(right_p.read_bytes()).hexdigest()
        elif algo == "md5":
            lh = hashlib.md5(left_p.read_bytes()).hexdigest()
            rh = hashlib.md5(right_p.read_bytes()).hexdigest()
        else:
            errors.append(f"unsupported parity algorithm: {algo}")
            results.append({"left": str(left), "right": str(right), "ok": False})
            continue
        ok = lh == rh
        if not ok:
            errors.append(f"parity mismatch for {left} vs {right}")
        results.append({"left": str(left), "right": str(right), "algorithm": algo, "left_hash": lh, "right_hash": rh, "ok": ok})
    return {"ok": not errors, "errors": errors, "results": results}


def probe_contract(plan_text: str, *, cwd: str | Path | None = None,
                   timeout: float | None = None) -> dict[str, Any]:
    """Run the minimal feasibility spike through the strict command runner."""
    contract, parse_errors = parse_execution_contract(plan_text)
    if contract is None:
        return {"ok": False, "configured": False, "errors": parse_errors or ["execution contract missing"], "result": None}
    probe = contract.probe
    if not probe or not probe.get("command"):
        return {"ok": True, "configured": False, "error": None, "result": None, "message": "no probe configured; full verification deferred"}
    cmd = probe["command"]
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
        return {"ok": False, "configured": True, "errors": [f"invalid probe command: {cmd!r}"], "result": None}
    try:
        configured_timeout = float(probe.get("timeout_seconds", 30))
    except (TypeError, ValueError):
        configured_timeout = 30.0
    result = run_command(cmd, cwd=cwd, timeout=float(timeout or configured_timeout))
    expected = str(probe.get("expected_output", "")).strip()
    matched = not expected or expected in (result.get("stdout") or "")
    ok = bool(result.get("ok")) and matched
    errors = [] if ok else [
        f"probe failed (exit={result.get('exit_code')}, expected_output={'present' if expected else 'none'}, "
        f"stderr={(result.get('stderr') or '')[:200]})"
    ]
    return {
        "ok": ok,
        "configured": True,
        "errors": errors,
        "result": result,
        "expected_output": expected,
        "matched": matched,
        "probe": probe,
    }
