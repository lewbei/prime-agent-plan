"""Paper-grounded semantic isolation (ACID-Agent 2608.13900, Section 2.2.3).

Conflict detection is based on *currently active* artifact ownership, not every
historical access ever recorded.  A release event terminates that agent's
active lease for the path/workspace while the append-only record remains
available for audit.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentIsolation(str, Enum):
    INDEPENDENT = "independent"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"


class OperationIsolation(str, Enum):
    SERIALIZABLE = "serializable"
    SNAPSHOT = "snapshot"
    OPTIMISTIC = "optimistic"


@dataclass
class ArtifactVersion:
    path: str
    agent_id: str
    workspace: str = "main"
    version: int = 1
    operation: str = "write"
    effect: str = "modify"
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ConflictReport:
    ok: bool
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ArtifactVersion] = field(default_factory=list)


class IsolationManager:
    """Versioned workspace registry used to detect active shared-artifact conflicts."""

    def __init__(self) -> None:
        self.records: list[ArtifactVersion] = []
        self._versions: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    def acquire_artifact(self, path: str, agent_id: str, *,
                         operation: str = "write",
                         effect: str = "modify",
                         workspace: str = "main",
                         isolation: OperationIsolation = OperationIsolation.SERIALIZABLE,
                         content: str | None = None) -> ArtifactVersion:
        del isolation  # recorded by conflict evaluation; acquisition itself is neutral.
        with self._lock:
            key = (path, workspace)
            version = self._versions.get(key, 0) + 1
            self._versions[key] = version
            record = ArtifactVersion(
                path=path,
                agent_id=agent_id,
                workspace=workspace,
                version=version,
                operation=operation,
                effect=effect,
                content_hash=hashlib.sha256(
                    (content or f"{path}:{agent_id}:{version}").encode("utf-8")
                ).hexdigest(),
            )
            self.records.append(record)
            return record

    def release_artifact(self, path: str, agent_id: str, *, workspace: str = "main") -> None:
        with self._lock:
            self.records.append(ArtifactVersion(
                path=path,
                agent_id=agent_id,
                workspace=workspace,
                version=self._versions.get((path, workspace), 0),
                operation="release",
                effect="ownership released",
            ))

    def _active_records(self) -> list[ArtifactVersion]:
        active: dict[tuple[str, str, str], ArtifactVersion] = {}
        for record in self.records:
            key = (record.path, record.workspace, record.agent_id)
            if record.operation == "release":
                active.pop(key, None)
            else:
                active[key] = record
        return list(active.values())

    def detect_conflicts(self, *, isolation: OperationIsolation = OperationIsolation.SERIALIZABLE) -> ConflictReport:
        with self._lock:
            active = self._active_records()
            conflicts: list[dict[str, Any]] = []
            seen: set[tuple[str, ...]] = set()
            writes = [r for r in active if r.operation in ("write", "delete")]

            for i, a in enumerate(writes):
                for b in writes[i + 1:]:
                    if a.path != b.path or a.workspace != b.workspace or a.agent_id == b.agent_id:
                        continue
                    if isolation == OperationIsolation.OPTIMISTIC and a.version != b.version:
                        continue
                    key = tuple(sorted((a.agent_id, b.agent_id))) + (a.path, a.workspace)
                    if key in seen:
                        continue
                    seen.add(key)
                    conflicts.append({
                        "path": a.path,
                        "workspace": a.workspace,
                        "agents": sorted({a.agent_id, b.agent_id}),
                        "operations": [a.operation, b.operation],
                        "versions": [a.version, b.version],
                        "isolation": isolation.value,
                    })

            if isolation == OperationIsolation.SERIALIZABLE:
                reads = [r for r in active if r.operation == "read"]
                for read in reads:
                    later_writers = [
                        writer for writer in writes
                        if writer.path == read.path
                        and writer.workspace == read.workspace
                        and writer.agent_id != read.agent_id
                        and read.created_at < writer.created_at
                    ]
                    for writer in later_writers:
                        conflicts.append({
                            "path": read.path,
                            "workspace": read.workspace,
                            "agents": sorted({read.agent_id, writer.agent_id}),
                            "operations": ["read", writer.operation],
                            "versions": [read.version, writer.version],
                            "isolation": isolation.value,
                            "reason": "write-after-read ordering conflict",
                        })

            return ConflictReport(
                ok=not conflicts,
                conflicts=conflicts,
                artifacts=list(self.records),
            )


_DEFAULT_MANAGER = IsolationManager()


def acquire_artifact(path: str, agent_id: str, **kwargs) -> ArtifactVersion:
    return _DEFAULT_MANAGER.acquire_artifact(path, agent_id, **kwargs)


def release_artifact(path: str, agent_id: str, **kwargs) -> None:
    _DEFAULT_MANAGER.release_artifact(path, agent_id, **kwargs)


def detect_conflicts(**kwargs) -> ConflictReport:
    return _DEFAULT_MANAGER.detect_conflicts(**kwargs)
