"""Shared state and canonical helpers for runtime-integrity closure."""
from __future__ import annotations

import contextvars
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

ACTIVE_TRANSACTION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prime_active_transaction_id", default=None
)
ACTIVE_WORKSPACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prime_active_workspace_id", default=None
)
ACTIVE_WORKSPACE_PATH: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prime_active_workspace_path", default=None
)
CERTIFICATE_WORKSPACES: dict[str, str] = {}
CERTIFICATE_WORKSPACES_LOCK = threading.RLock()
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_session_id(session_id: Any) -> str:
    sid = str(session_id or "")
    if (
        not SESSION_ID_RE.fullmatch(sid)
        or sid in {".", ".."}
        or "/" in sid
        or "\\" in sid
        or "\x00" in sid
    ):
        raise ValueError(
            "session_id must contain only letters, digits, '.', '_' or '-' "
            "and must not contain path traversal components"
        )
    return sid


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def workspace_identity(path: str | None) -> str:
    if not path:
        return "unbound-test-workspace"
    resolved = Path(path).resolve(strict=False)
    payload: dict[str, Any] = {"path": str(resolved)}
    try:
        stat = resolved.stat()
        payload.update({"device": stat.st_dev, "inode": stat.st_ino})
    except OSError:
        payload.update({"device": None, "inode": None})
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
