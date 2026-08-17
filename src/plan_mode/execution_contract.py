"""Execution contracts: make plans release only against real-world evidence.

This module operationalizes the distinction between *plan verification* and
*execution verification*. A plan may be causally valid while the work product
is stubbed. The contract therefore carries:

- ``probe``: a minimal executable spike that must produce the expected output
  before the full plan is trusted (feasibility gate).
- ``verification_commands``: commands an independent verifier must run.
- ``expected_artifacts``: filesystem artifacts with minimum size/line budgets.
- ``symbols``: declared functions/variables per source file. After execution,
  `symbol_audit` detects missing declarations and undeclared helpers, closing
  the "write a plausible stub" loophole.
- ``parity_checks``: equality/hash comparisons against the old behavior.
- ``workspace_invariants``: natural-language invariants enforced by the
  harness-level verifier.

Literature grounding:
- ACID-Agent (2608.13900): evidence obligations and validated-effect-only commits.
- STAIR (2608.09524): validate execution effects before experience reuse.
- FlowScout (2608.10039): execution-feedback-guided repair.
- AgentRewind (2608.14380): checkpoint before execution; rewind on failed probe.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .causal_validator import PlanParser


@dataclass
class ExecutionContract:
    probe: dict[str, Any] = field(default_factory=dict)
    verification_commands: list[list[str]] = field(default_factory=list)
    expected_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    workspace_invariants: list[str] = field(default_factory=list)
    parity_checks: list[dict[str, str]] = field(default_factory=list)
    symbols: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def parse_execution_contract(plan_text: str) -> tuple[Optional[ExecutionContract], list[str]]:
    """Extract the JSON execution contract from a plan.

    Accepted forms:
    1. `## Execution Contract` followed by a fenced ```json block.
    2. Any fenced ```json block containing ``verification_commands`` or ``probe``.
    """
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
    probe = raw_probe if isinstance(raw_probe, dict) else {}

    raw_cmds = data.get("verification_commands")
    verification_commands = [cmd for cmd in raw_cmds if isinstance(cmd, list)] if isinstance(raw_cmds, list) else []

    raw_artifacts = data.get("expected_artifacts")
    expected_artifacts = {str(k): (v if isinstance(v, dict) else {}) for k, v in raw_artifacts.items()} if isinstance(raw_artifacts, dict) else {}

    raw_invariants = data.get("workspace_invariants")
    workspace_invariants = [str(x) for x in raw_invariants] if isinstance(raw_invariants, list) else []

    raw_parity = data.get("parity_checks")
    parity_checks = [p for p in raw_parity if isinstance(p, dict)] if isinstance(raw_parity, list) else []

    raw_symbols = data.get("symbols")
    symbols = {str(k): (v if isinstance(v, dict) else {}) for k, v in raw_symbols.items()} if isinstance(raw_symbols, dict) else {}

    contract = ExecutionContract(
        probe=probe,
        verification_commands=verification_commands,
        expected_artifacts=expected_artifacts,
        workspace_invariants=workspace_invariants,
        parity_checks=parity_checks,
        symbols=symbols,
        raw=data,
    )
    return contract, errors


def validate_execution_contract(plan_text: str, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Validate the contract against the plan AST without executing it."""
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
    for cmd in contract.verification_commands:
        if not cmd or not isinstance(cmd[0], str):
            errors.append(f"invalid verification command: {cmd!r}")
            break

    ast_tree = PlanParser.parse_plan(plan_text)
    declared_outputs = {out for action in ast_tree.actions for out in action.outputs}
    if declared_outputs:
        missing = sorted(out for out in declared_outputs if out not in contract.expected_artifacts)
        if missing:
            errors.append(f"declared task outputs have no expected-artifact budgets: {missing}")

    for path, budget in contract.expected_artifacts.items():
        p = Path(cwd or Path.cwd()) / path
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
        if not functions and not classes and not variables:
            errors.append(f"symbol contract for {path} must declare at least one function, class, or variable")

    return {"ok": not errors, "errors": errors, "contract": contract,
            "declared_outputs": sorted(declared_outputs)}


def _symbols_from_source(source: str) -> dict[str, set[str]]:
    """Extract top-level module functions, classes, and variables from Python source."""
    tree = ast.parse(source)
    funcs: set[str] = set()
    classes: set[str] = set()
    vars_: set[str] = set()

    def _extract_target(target):
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
    """Return actual functions/classes/variables for each existing Python file."""
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
    """Compare declared symbol contracts with the actual source tree."""
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


def run_verification_commands(contract: ExecutionContract, *, cwd: str | Path | None = None,
                              timeout: float = 60.0) -> dict[str, Any]:
    """Run every verification command declared by the contract.

    The verdict is true only when all commands exit 0.
    """
    results = [run_command(cmd, cwd=cwd, timeout=timeout) for cmd in contract.verification_commands]
    ok = all(r.get("ok") for r in results)
    return {"ok": ok, "results": results, "total": len(results),
            "passed": sum(bool(r.get("ok")) for r in results)}


def parity_audit(contract: ExecutionContract, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Execute parity checks by hashing the declared left/right artifacts."""
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
        results.append({"left": str(left), "right": str(right), "algorithm": algo,
                        "left_hash": lh, "right_hash": rh, "ok": ok})
    return {"ok": not errors, "errors": errors, "results": results}


def run_command(cmd: list[str], *, cwd: str | Path | None = None,
                timeout: float = 60.0) -> dict[str, Any]:
    resolved = list(cmd)
    if resolved and resolved[0] in ("python", "python3"):
        resolved[0] = sys.executable
    try:
        proc = subprocess.run(resolved, cwd=str(cwd or Path.cwd()), capture_output=True,
                              text=True, timeout=timeout)
        return {"command": cmd, "exit_code": proc.returncode, "ok": proc.returncode == 0,
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
                "timeout": False}
    except subprocess.TimeoutExpired:
        return {"command": cmd, "exit_code": None, "ok": False, "stdout": "",
                "stderr": "timeout", "timeout": True}
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return {"command": cmd, "exit_code": None, "ok": False, "stdout": "",
                "stderr": f"command execution failed: {exc}", "timeout": False}


def probe_contract(plan_text: str, *, cwd: str | Path | None = None,
                   timeout: float | None = None) -> dict[str, Any]:
    """Run the contract's minimal feasibility spike.

    If the spike fails, the plan should be revised before full implementation.
    """
    contract, parse_errors = parse_execution_contract(plan_text)
    if contract is None:
        return {"ok": False, "configured": False,
                "errors": parse_errors or ["execution contract missing"], "result": None}
    probe = contract.probe
    if not probe or not probe.get("command"):
        return {"ok": True, "configured": False, "error": None, "result": None,
                "message": "no probe configured; full verification deferred"}
    cmd = probe["command"]
    if not isinstance(cmd, list) or not cmd:
        return {"ok": False, "configured": True,
                "errors": [f"invalid probe command: {cmd!r}"], "result": None}
    result = run_command(cmd, cwd=cwd, timeout=timeout or float(probe.get("timeout_seconds", 30)))
    expected = str(probe.get("expected_output", "")).strip()
    matched = not expected or expected in (result.get("stdout") or "")
    ok = bool(result.get("ok")) and matched
    errors = [] if ok else [f"probe failed (exit={result.get('exit_code')}, expected_output={'present' if expected else 'none'}, stderr={result.get('stderr')[:200]})"]
    return {"ok": ok, "configured": True, "errors": errors, "result": result,
            "expected_output": expected, "matched": matched, "probe": probe}
