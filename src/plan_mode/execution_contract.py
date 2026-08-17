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

    m = re.search(r"##\s*Execution\s+Contract.*?\n```json\s*\n(.*?)\n```", plan_text, re.I | re.S)
    if m:
        candidates.append(m.group(1))
    for block in re.finditer(r"```json\s*\n(.*?)\n```", plan_text, re.I | re.S):
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

    contract = ExecutionContract(
        probe=data.get("probe", {}) if isinstance(data.get("probe"), dict) else {},
        verification_commands=[cmd for cmd in data.get("verification_commands", []) if isinstance(cmd, list)],
        expected_artifacts={str(k): (v if isinstance(v, dict) else {}) for k, v in data.get("expected_artifacts", {}).items()},
        workspace_invariants=[str(x) for x in data.get("workspace_invariants", [])],
        parity_checks=[p for p in data.get("parity_checks", []) if isinstance(p, dict)],
        symbols={str(k): (v if isinstance(v, dict) else {}) for k, v in data.get("symbols", {}).items()},
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

    ast = PlanParser.parse_plan(plan_text)
    declared_outputs = {out for action in ast.actions for out in action.outputs}
    if declared_outputs:
        missing = sorted(out for out in declared_outputs if out not in contract.expected_artifacts)
        if missing:
            errors.append(f"declared task outputs have no expected-artifact budgets: {missing}")

    for path, budget in contract.expected_artifacts.items():
        p = Path(cwd or Path.cwd()) / path
        if p.exists():
            size = p.stat().st_size
            if budget.get("min_bytes") and size < int(budget["min_bytes"]):
                errors.append(f"{path}: actual {size} bytes < min_bytes {budget['min_bytes']}")
            if budget.get("min_lines"):
                try:
                    lines = len(p.read_text(encoding="utf-8").splitlines())
                except Exception:
                    lines = 0
                if lines < int(budget["min_lines"]):
                    errors.append(f"{path}: actual {lines} lines < min_lines {budget['min_lines']}")

    for path, syms in contract.symbols.items():
        functions = syms.get("functions", [])
        variables = syms.get("variables", [])
        if not functions and not variables:
            errors.append(f"symbol contract for {path} must declare at least one function or variable")

    return {"ok": not errors, "errors": errors, "contract": contract,
            "declared_outputs": sorted(declared_outputs)}


def _symbols_from_source(source: str) -> dict[str, set[str]]:
    tree = ast.parse(source)
    funcs: set[str] = set()
    classes: set[str] = set()
    vars_: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802
            funcs.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            funcs.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node):  # noqa: N802
            classes.add(node.name)
            self.generic_visit(node)

        def _add_target(self, target):
            if isinstance(target, ast.Name):
                vars_.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self._add_target(elt)

        def visit_Assign(self, node):  # noqa: N802
            for target in node.targets:
                self._add_target(target)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):  # noqa: N802
            self._add_target(node.target)
            self.generic_visit(node)

        def visit_AugAssign(self, node):  # noqa: N802
            self._add_target(node.target)
            self.generic_visit(node)

        def visit_NamedExpr(self, node):  # noqa: N802
            self._add_target(node.target)
            self.generic_visit(node)

    Visitor().visit(tree)
    return {"functions": funcs, "classes": classes, "variables": vars_}


def scan_symbols(paths: list[str] | tuple[str, ...] | set[str], *,
                 cwd: str | Path | None = None) -> dict[str, dict[str, list[str]]]:
    """Return actual functions/classes/variables for each existing Python file."""
    base = Path(cwd or Path.cwd())
    out: dict[str, dict[str, list[str]]] = {}
    for raw in paths:
        p = Path(raw) if Path(raw).is_absolute() else base / raw
        if not p.exists() or p.suffix != ".py":
            out[raw] = {"functions": [], "classes": [], "variables": [], "missing": True}
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
        if found.get("syntax_error"):
            errors.append(f"{path}: syntax error: {found['syntax_error']}")
            files[path] = found
            continue
        expected_funcs = {str(x) for x in declared.get("functions", [])}
        expected_vars = {str(x) for x in declared.get("variables", [])}
        actual_funcs = set(found.get("functions", []))
        actual_vars = set(found.get("variables", [])) | set(found.get("classes", []))
        missing_funcs = sorted(expected_funcs - actual_funcs)
        missing_vars = sorted(expected_vars - actual_vars)
        undeclared_funcs = sorted(actual_funcs - expected_funcs)
        undeclared_vars = sorted(actual_vars - expected_vars)
        files[path] = {
            "missing_functions": missing_funcs,
            "missing_variables": missing_vars,
            "undeclared_functions": undeclared_funcs,
            "undeclared_variables": undeclared_vars,
            "actual_functions": sorted(actual_funcs),
            "actual_variables": sorted(actual_vars),
        }
        if missing_funcs:
            errors.append(f"{path}: missing declared functions: {missing_funcs}")
        if missing_vars:
            errors.append(f"{path}: missing declared variables: {missing_vars}")
        if undeclared_funcs:
            errors.append(f"{path}: undeclared functions not listed in contract: {undeclared_funcs}")
        if undeclared_vars:
            errors.append(f"{path}: undeclared variables not listed in contract: {undeclared_vars}")
    return {"ok": not errors, "errors": errors, "files": files, "contract": contract}


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
    errors = [] if ok else [f"probe failed (exit={result.get('exit_code')}, expected_output={'present' if expected else 'none'})"]
    return {"ok": ok, "configured": True, "errors": errors, "result": result,
            "expected_output": expected, "matched": matched, "probe": probe}
