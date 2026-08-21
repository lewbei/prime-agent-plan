"""Session path, checkpoint, rewind, and world-state identity hardening."""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, MutableMapping

from .runtime_closure_context import canonical_json, validate_session_id

_FALLBACK_CHECKPOINT_FIELDS = (
    "rounds", "best_version", "best_score", "status", "completed_at",
    "execution_log", "world_state", "replan_pending", "replan_task",
    "replan_tier", "replan_scope", "search_tree", "release_gate",
    "committed_version", "committed_score", "committed_at",
    "committed_plan_hash",
)


def _snapshot_digest(session_id: str, present_fields: list[str], snapshot: dict[str, Any]) -> str:
    payload = {
        "schema": 2,
        "session_id": session_id,
        "present_fields": present_fields,
        "snapshot": snapshot,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def install_session_closure(ns: MutableMapping[str, Any]) -> None:
    raw_lock = ns["session_lock"]
    raw_save = ns["_save_session"]
    raw_load = ns["_load_session"]
    checkpoint_fields = tuple(
        ns.get("_CHECKPOINT_FIELDS") or _FALLBACK_CHECKPOINT_FIELDS
    )

    def session_path(plans_dir: str | Path, session_id: str) -> Path:
        sid = validate_session_id(session_id)
        root = Path(plans_dir).resolve()
        candidate = (root / f"{sid}.json").resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("session path escapes plans_dir") from exc
        return candidate

    @contextmanager
    def session_lock(plans_dir: str | Path, session_id: str, timeout: float = 10.0):
        sid = validate_session_id(session_id)
        with raw_lock(plans_dir, sid, timeout=timeout):
            yield

    def save_session(plans_dir: str | Path, session: dict[str, Any]) -> None:
        if not isinstance(session, dict):
            raise TypeError("session must be a mapping")
        validate_session_id(session.get("session_id"))
        raw_save(Path(plans_dir), session)

    def load_session(plans_dir: str | Path, session_id: str) -> dict[str, Any]:
        return raw_load(Path(plans_dir), validate_session_id(session_id))

    def plans_dir_for(session: dict[str, Any] | str, plans_dir: str | Path | None) -> Path:
        if plans_dir is not None:
            return Path(plans_dir)
        if isinstance(session, dict) and session.get("plans_dir"):
            return Path(session["plans_dir"])
        return Path(ns["DEFAULT_PLANS_DIR"])

    def checkpoint(
        session: dict[str, Any] | str,
        *,
        plans_dir: str | Path | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        pdir = plans_dir_for(session, plans_dir)
        sid = validate_session_id(
            session if isinstance(session, str) else session.get("session_id")
        )
        with session_lock(pdir, sid):
            state = load_session(pdir, sid) if isinstance(session, str) else session
            present_fields = [key for key in checkpoint_fields if key in state]
            snapshot = {
                key: copy.deepcopy(state[key])
                for key in present_fields
            }
            digest = _snapshot_digest(sid, present_fields, snapshot)
            cp_id = f"cp-{ns['_now']().replace(':', '')}-{len(state.get('checkpoints', [])) + 1}"
            entry = {
                "id": cp_id,
                "ts": ns["_now"](),
                "note": note,
                "snapshot_schema": 2,
                "present_fields": present_fields,
                "snapshot_hash": digest,
                "session_hash": digest,
                "snapshot": snapshot,
            }
            state.setdefault("checkpoints", []).append(entry)
            save_session(pdir, state)
            return {
                "checkpoint_id": cp_id,
                "note": note,
                "best_version": state.get("best_version"),
                "rounds": len(state.get("rounds", [])),
                "snapshot_hash": digest,
            }

    def rewind(
        session: dict[str, Any] | str,
        checkpoint_id: str | None = None,
        *,
        plans_dir: str | Path | None = None,
        note: str = "manual rewind",
    ) -> dict[str, Any]:
        pdir = plans_dir_for(session, plans_dir)
        sid = validate_session_id(
            session if isinstance(session, str) else session.get("session_id")
        )
        with session_lock(pdir, sid):
            state = load_session(pdir, sid) if isinstance(session, str) else session
            checkpoints = state.get("checkpoints") or []
            if not checkpoints:
                raise ValueError("session has no checkpoints")
            if checkpoint_id is None:
                selected = checkpoints[-1]
            else:
                selected = next(
                    (item for item in checkpoints if item.get("id") == checkpoint_id),
                    None,
                )
                if selected is None:
                    raise ValueError(f"checkpoint {checkpoint_id!r} not found in session")

            snapshot = selected.get("snapshot")
            if not isinstance(snapshot, dict):
                raise ValueError("checkpoint snapshot is missing or invalid")

            if int(selected.get("snapshot_schema") or 1) >= 2:
                present_fields = selected.get("present_fields")
                if not isinstance(present_fields, list) or not all(
                    isinstance(key, str) for key in present_fields
                ):
                    raise ValueError("checkpoint present_fields is invalid")
                expected = _snapshot_digest(sid, present_fields, snapshot)
                if expected != selected.get("snapshot_hash"):
                    raise ValueError("checkpoint integrity verification failed")
            else:
                present_fields = list(snapshot)
                old_hash = selected.get("session_hash")
                if old_hash:
                    calculated = hashlib.sha256(
                        json.dumps(
                            snapshot.get("rounds", []),
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    if calculated != old_hash:
                        raise ValueError("legacy checkpoint integrity verification failed")

            present = set(present_fields)
            for key in checkpoint_fields:
                if key in present:
                    state[key] = copy.deepcopy(snapshot.get(key))
                else:
                    state.pop(key, None)

            state.setdefault("rewind_log", []).append({
                "ts": ns["_now"](),
                "checkpoint_id": selected.get("id"),
                "note": note,
                "restored_rounds": len(state.get("rounds", [])),
                "snapshot_hash": selected.get("snapshot_hash") or selected.get("session_hash"),
            })
            save_session(pdir, state)
            return state

    ns["_validate_session_id"] = validate_session_id
    ns["_session_path"] = session_path
    ns["session_lock"] = session_lock
    ns["_save_session"] = save_session
    ns["_load_session"] = load_session
    ns["checkpoint"] = checkpoint
    ns["rewind"] = rewind


def install_world_state_identity() -> None:
    from . import session as session_mod
    from .runtime import executor as executor_mod

    def compute_world_state_hash(facts: list[Any]) -> str:
        entries: list[str] = []
        for fact in facts:
            provenance = getattr(fact, "provenance", None)
            source_type = getattr(getattr(provenance, "source_type", None), "value", None)
            entry = {
                "predicate": fact.predicate,
                "args": fact.args,
                "truth": fact.truth.value,
                "witnessability": fact.witnessability.value,
                "ttl_seconds": fact.ttl_seconds,
                "provenance": {
                    "source_type": source_type,
                    "source_id": getattr(provenance, "source_id", None),
                },
                "attestation_id": (
                    fact.metadata.get("attestation_id")
                    if isinstance(getattr(fact, "metadata", None), dict)
                    else None
                ),
            }
            entries.append(canonical_json(entry))
        return hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()

    session_mod.compute_world_state_hash = compute_world_state_hash
    executor_mod.compute_world_state_hash = compute_world_state_hash
