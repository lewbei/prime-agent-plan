"""Isolated execution-trace re-verification and runtime attestation."""
from __future__ import annotations

import copy
import hashlib
import hmac
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, MutableMapping

from .runtime_closure_context import canonical_json, workspace_identity

_TRACE_ATTESTATION_KEY = secrets.token_bytes(32)
_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
_MAX_SNAPSHOT_ENTRIES = 10_000


def _map_workspace_path(raw: str, source: Path, snapshot: Path) -> str:
    candidate = Path(raw)
    if not candidate.is_absolute():
        return raw
    try:
        relative = candidate.resolve(strict=False).relative_to(source)
    except ValueError:
        return raw
    return str(snapshot / relative)


def _map_token(token: str, source: Path, snapshot: Path) -> str:
    source_text = str(source)
    if token == source_text:
        return str(snapshot)
    if token.startswith(source_text + os.sep):
        return str(snapshot) + token[len(source_text):]
    return token


def _snapshot_size(root: Path) -> tuple[int, int]:
    entries = 0
    total = 0
    for path in root.rglob("*"):
        entries += 1
        if entries > _MAX_SNAPSHOT_ENTRIES:
            raise ValueError(
                f"verification workspace has more than {_MAX_SNAPSHOT_ENTRIES} entries"
            )
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
            if total > _MAX_SNAPSHOT_BYTES:
                raise ValueError(
                    f"verification workspace exceeds {_MAX_SNAPSHOT_BYTES} bytes"
                )
    return entries, total


def _mapped_contract(contract: Any, source: Path, snapshot: Path) -> Any:
    mapped = copy.deepcopy(contract)
    mapped.expected_artifacts = {
        _map_workspace_path(str(path), source, snapshot): copy.deepcopy(budget)
        for path, budget in contract.expected_artifacts.items()
    }
    mapped.symbols = {
        _map_workspace_path(str(path), source, snapshot): copy.deepcopy(symbols)
        for path, symbols in contract.symbols.items()
    }
    mapped.parity_checks = [
        {
            **copy.deepcopy(check),
            "left": _map_workspace_path(str(check.get("left", "")), source, snapshot),
            "right": _map_workspace_path(str(check.get("right", "")), source, snapshot),
        }
        for check in contract.parity_checks
    ]
    mapped.verification_commands = [
        [_map_token(str(token), source, snapshot) for token in command]
        for command in contract.verification_commands
    ]
    mapped.exit_criteria = []
    for criterion in contract.exit_criteria:
        item = copy.deepcopy(criterion)
        command = item.get("command")
        if isinstance(command, list):
            item["command"] = [
                _map_token(str(token), source, snapshot)
                for token in command
            ]
        mapped.exit_criteria.append(item)
    return mapped


def _json_safe(value: Any) -> Any:
    """Convert runtime result objects into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except TypeError:
            return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        normalized = [_json_safe(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return repr(value)


def _evidence_digest(result: dict[str, Any]) -> str:
    evidence = result.get("evidence")
    raw = getattr(evidence, "raw", None) if evidence is not None else None
    payload = {
        "ok": bool(result.get("ok")),
        "errors": list(result.get("errors") or []),
        "warnings": list(result.get("warnings") or []),
        "alignment": _json_safe(result.get("alignment")),
        "exit_criteria": _json_safe(result.get("exit_criteria")),
        "workspace_reverification": _json_safe(result.get("workspace_reverification")),
        "raw_evidence": _json_safe(raw) if isinstance(raw, dict) else None,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _attestation_payload(
    result: dict[str, Any],
    *,
    plan_hash: str,
    session_id: str,
    workspace_id: str,
    certificate_id: str | None,
    issued_at: float,
) -> dict[str, Any]:
    return {
        "plan_hash": plan_hash,
        "session_id": session_id,
        "workspace_identity": workspace_id,
        "certificate_id": certificate_id,
        "evidence_digest": _evidence_digest(result),
        "issued_at": round(float(issued_at), 6),
    }


def _sign(payload: dict[str, Any]) -> str:
    return hmac.new(
        _TRACE_ATTESTATION_KEY,
        canonical_json(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_runtime_trace_attestation(
    result: dict[str, Any],
    *,
    plan_hash: str,
    session_id: str,
    workspace_id: str,
    certificate_id: str | None = None,
) -> bool:
    attestation = result.get("runtime_attestation")
    if not isinstance(attestation, dict):
        return False
    signature = attestation.get("signature")
    issued_at = attestation.get("issued_at")
    if not isinstance(signature, str) or not isinstance(issued_at, (int, float)):
        return False
    payload = _attestation_payload(
        result,
        plan_hash=plan_hash,
        session_id=session_id,
        workspace_id=workspace_id,
        certificate_id=certificate_id,
        issued_at=float(issued_at),
    )
    stored = {key: attestation.get(key) for key in payload}
    return stored == payload and hmac.compare_digest(signature, _sign(payload))


def install_trace_runtime_closure(ns: MutableMapping[str, Any]) -> None:
    from . import execution_trace as trace

    raw_workspace_reverify = trace._workspace_reverify
    if not getattr(raw_workspace_reverify, "_runtime_snapshot_closure", False):
        def workspace_reverify(plan_text, contract, cwd):
            source = Path(cwd).resolve()
            entries, total = _snapshot_size(source)
            temporary_root = Path(tempfile.mkdtemp(
                prefix=".prime_verify_snapshot_",
                dir=str(source.parent),
            ))
            snapshot = temporary_root / "workspace"
            try:
                shutil.copytree(
                    source,
                    snapshot,
                    symlinks=True,
                    copy_function=shutil.copy2,
                )
                mapped_contract = _mapped_contract(contract, source, snapshot)
                mapped_plan = str(plan_text).replace(str(source), str(snapshot))
                result = raw_workspace_reverify(
                    mapped_plan,
                    mapped_contract,
                    snapshot,
                )
                result["verification_snapshot"] = {
                    "isolated_copy": True,
                    "source_workspace": str(source),
                    "entries": entries,
                    "bytes": total,
                }
                return result
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)

        workspace_reverify._runtime_snapshot_closure = True
        trace._workspace_reverify = workspace_reverify

    raw_verify = trace.verify_execution_trace
    if getattr(raw_verify, "_runtime_trace_attestation_closure", False):
        ns["_verify_execution_trace_runtime_attestation"] = verify_runtime_trace_attestation
        return

    def verify_execution_trace(
        plan_text,
        evidence,
        *,
        cwd=None,
        require_independent_verifier=None,
        require_bound_identity=False,
        expected_session_id=None,
        expected_certificate_id=None,
        expected_workspace_identity=None,
    ):
        try:
            result = raw_verify(
                plan_text,
                evidence,
                cwd=cwd,
                require_independent_verifier=require_independent_verifier,
                require_bound_identity=require_bound_identity,
                expected_session_id=expected_session_id,
                expected_certificate_id=expected_certificate_id,
                expected_workspace_identity=expected_workspace_identity,
            )
        except (OSError, ValueError) as exc:
            return {
                "ok": False,
                "errors": [f"workspace verification snapshot failed: {exc}"],
                "warnings": [],
                "evidence": None,
                "alignment": None,
                "exit_criteria": None,
                "workspace_reverification": None,
            }

        errors = list(result.get("errors") or [])
        parsed = result.get("evidence")
        raw = getattr(parsed, "raw", None) if parsed is not None else None
        raw = raw if isinstance(raw, dict) else {}

        for field, expected, label in (
            ("session_id", expected_session_id, "session"),
            ("certificate_id", expected_certificate_id, "authorization certificate"),
            ("workspace_identity", expected_workspace_identity, "workspace"),
        ):
            supplied = raw.get(field)
            if expected is not None and supplied is not None and supplied != expected:
                errors.append(
                    f"execution evidence is bound to another {label}: "
                    f"{supplied!r} != {expected!r}"
                )

        actual_workspace_id = workspace_identity(cwd) if cwd is not None else ""
        if (
            expected_workspace_identity is not None
            and actual_workspace_id
            and actual_workspace_id != expected_workspace_identity
        ):
            errors.append("live workspace identity does not match the release boundary")

        result["errors"] = list(dict.fromkeys(errors))
        result["ok"] = not result["errors"]
        result.pop("runtime_attestation", None)

        if result["ok"] and cwd is not None:
            plan_hash = hashlib.sha256(str(plan_text).encode("utf-8")).hexdigest()
            bound_session = str(expected_session_id or raw.get("session_id") or "")
            bound_workspace = str(
                expected_workspace_identity
                or raw.get("workspace_identity")
                or actual_workspace_id
            )
            issued_at = time.time()
            payload = _attestation_payload(
                result,
                plan_hash=plan_hash,
                session_id=bound_session,
                workspace_id=bound_workspace,
                certificate_id=expected_certificate_id,
                issued_at=issued_at,
            )
            result["runtime_attestation"] = {
                **payload,
                "signature": _sign(payload),
            }
        return result

    verify_execution_trace._runtime_trace_attestation_closure = True
    trace.verify_execution_trace = verify_execution_trace
    ns["verify_execution_trace"] = verify_execution_trace
    ns["_verify_execution_trace_runtime_attestation"] = verify_runtime_trace_attestation
