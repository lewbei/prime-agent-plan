"""Canonical cryptographic identity for execution-isolation policy.

The ephemeral workspace path is deliberately excluded: it is an instance
identifier created after authorization, not a privilege.  Every security-
relevant capability/limit remains bound into the fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_isolation_policy(policy: Any) -> dict[str, Any]:
    """Return the security-relevant, deterministic policy representation."""
    if hasattr(policy, "model_dump"):
        data = dict(policy.model_dump())
    elif isinstance(policy, dict):
        data = dict(policy)
    else:
        raise TypeError("isolation policy must be a mapping or Pydantic model")

    # workspace_dir is an ephemeral runtime instance and is separately enforced
    # as mandatory by TransactionalExecutionManager.  It must not change the
    # authorization privilege identity.
    data.pop("workspace_dir", None)

    # These collections are semantically sets for policy purposes.
    if isinstance(data.get("blocked_paths"), list):
        data["blocked_paths"] = sorted(data["blocked_paths"])
    if isinstance(data.get("env_whitelist"), list):
        data["env_whitelist"] = sorted(data["env_whitelist"])
    return data


def compute_isolation_policy_hash(policy: Any) -> str:
    """SHA-256 fingerprint of all security-relevant isolation privileges."""
    payload = json.dumps(
        canonical_isolation_policy(policy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
