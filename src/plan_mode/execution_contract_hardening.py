"""Workspace-containment hardening for execution-contract file observations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

from .runtime.sandbox import (
    PathTraversalEscapeError,
    SymlinkEscapeError,
    validate_path_within_workspace,
)


def install_execution_contract_hardening(ns: MutableMapping[str, Any]) -> None:
    """Replace host-reading contract helpers with workspace-contained variants."""
    legacy_validate = ns["validate_execution_contract"]
    legacy_scan_symbols = ns["scan_symbols"]
    legacy_artifact_audit = ns["artifact_audit"]
    legacy_parity_audit = ns["parity_audit"]

    def _safe_path(raw: str, base: Path) -> Path:
        candidate = Path(raw) if Path(raw).is_absolute() else base / raw
        try:
            resolved = validate_path_within_workspace(str(candidate), str(base))
        except (PathTraversalEscapeError, SymlinkEscapeError) as exc:
            raise ValueError(f"contract path escapes workspace: {raw!r}: {exc}") from exc
        return Path(resolved)

    def _path_errors(contract: Any, base: Path, *, plan_text: str | None = None) -> list[str]:
        errors: list[str] = []
        seen: set[tuple[str, str]] = set()

        def check(kind: str, raw: Any) -> None:
            text = str(raw)
            marker = (kind, text)
            if marker in seen:
                return
            seen.add(marker)
            try:
                _safe_path(text, base)
            except ValueError as exc:
                errors.append(f"{kind} {text!r}: {exc}")

        for raw in getattr(contract, "expected_artifacts", {}).keys():
            check("expected artifact", raw)
        for raw in getattr(contract, "symbols", {}).keys():
            check("symbol file", raw)
        for item in getattr(contract, "parity_checks", []):
            if isinstance(item, dict):
                if item.get("left"):
                    check("parity left", item["left"])
                if item.get("right"):
                    check("parity right", item["right"])

        if plan_text:
            ast_tree = ns["PlanParser"].parse_plan(plan_text)
            for action in ast_tree.actions:
                for raw in action.outputs:
                    check(f"task {action.id} output", raw)
                for raw in action.inputs:
                    # Internal and environmental artifacts are still contract
                    # paths once an execution workspace is selected.
                    check(f"task {action.id} input", raw)
        return errors

    def validate_execution_contract(plan_text: str, *, cwd: str | Path | None = None) -> dict[str, Any]:
        base = Path(cwd or Path.cwd()).resolve()
        contract, parse_errors = ns["parse_execution_contract"](plan_text)
        if contract is None:
            return legacy_validate(plan_text, cwd=base)
        containment = _path_errors(contract, base, plan_text=plan_text)
        if containment:
            return {
                "ok": False,
                "errors": list(parse_errors) + containment,
                "contract": contract,
                "declared_outputs": sorted(
                    out
                    for action in ns["PlanParser"].parse_plan(plan_text).actions
                    for out in action.outputs
                ),
            }
        return legacy_validate(plan_text, cwd=base)

    def scan_symbols(paths: list[str] | tuple[str, ...] | set[str], *,
                     cwd: str | Path | None = None) -> dict[str, dict[str, Any]]:
        base = Path(cwd or Path.cwd()).resolve()
        safe: list[str] = []
        refused: dict[str, dict[str, Any]] = {}
        for raw in paths:
            try:
                _safe_path(str(raw), base)
                safe.append(str(raw))
            except ValueError as exc:
                refused[str(raw)] = {
                    "functions": [],
                    "classes": [],
                    "variables": [],
                    "missing": True,
                    "security_error": str(exc),
                }
        out = legacy_scan_symbols(safe, cwd=base) if safe else {}
        out.update(refused)
        return out

    def artifact_audit(contract: Any, *, cwd: str | Path | None = None) -> dict[str, Any]:
        base = Path(cwd or Path.cwd()).resolve()
        errors = _path_errors(contract, base)
        if errors:
            return {
                "ok": False,
                "errors": errors,
                "artifacts": {
                    str(raw): {"exists": False, "security_error": "path outside workspace"}
                    for raw in getattr(contract, "expected_artifacts", {}).keys()
                },
            }
        return legacy_artifact_audit(contract, cwd=base)

    def parity_audit(contract: Any, *, cwd: str | Path | None = None) -> dict[str, Any]:
        base = Path(cwd or Path.cwd()).resolve()
        errors = _path_errors(contract, base)
        if errors:
            return {"ok": False, "errors": errors, "results": []}
        return legacy_parity_audit(contract, cwd=base)

    ns["validate_execution_contract"] = validate_execution_contract
    ns["scan_symbols"] = scan_symbols
    ns["artifact_audit"] = artifact_audit
    ns["parity_audit"] = parity_audit
