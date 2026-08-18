"""Paper-grounded semantic isolation (ACID-Agent 2608.13900, Section 2.2.3).

The paper distinguishes two levels of isolation:

1. Agent-level isolation:
   - independent agents operate on disjoint resources and run fully parallel;
   - collaborative agents keep independent workspace branches and merge with
     a Git-like workflow;
   - competitive agents explore alternatives in isolated environments.

2. Operation-level isolation:
   - effect annotations and inference;
   - versioned workspaces;
   - snapshot-based execution;
   - optimistic validation.

This module implements the deterministic subset needed by a planning engine:
artifact versions, workspace branches, effect annotations, and write/read
conflict detection.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


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
    operation: str = "write"       # write | read | delete
    effect: str = "modify"          # free-text effect annotation
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ConflictReport:
    ok: bool
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ArtifactVersion] = field(default_factory=list)


class IsolationManager:
    """Versioned workspace registry used to detect shared-artifact conflicts."""

    def __init__(self) -> None:
        self.records: list[ArtifactVersion] = []
        self._versions: dict[tuple[str, str], int] = {}

    def acquire_artifact(self, path: str, agent_id: str, *,
                         operation: str = "write",
                         effect: str = "modify",
                         workspace: str = "main",
                         isolation: OperationIsolation = OperationIsolation.SERIALIZABLE,
                         content: str | None = None) -> ArtifactVersion:
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
            content_hash=hashlib.sha256((content or f"{path}:{agent_id}:{version}").encode("utf-8")).hexdigest(),
        )
        self.records.append(record)
        return record

    def release_artifact(self, path: str, agent_id: str, *, workspace: str = "main") -> None:
        self.records.append(ArtifactVersion(
            path=path, agent_id=agent_id, workspace=workspace,
            operation="release", effect="ownership released",
        ))

    def detect_conflicts(self, *, isolation: OperationIsolation = OperationIsolation.SERIALIZABLE) -> ConflictReport:
        conflicts: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        writes = [r for r in self.records if r.operation in ("write", "delete")]
        for i, a in enumerate(writes):
            for b in writes[i + 1:]:
                if a.path != b.path or a.workspace != b.workspace:
                    continue
                # Collaborative branches are versioned; only same-version writes conflict.
                if isolation == OperationIsolation.OPTIMISTIC and a.version != b.version:
                    continue
                if a.agent_id == b.agent_id:
                    continue
                key = tuple(sorted((a.agent_id, b.agent_id, a.path, str(a.version), str(b.version))))
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
        # Read-after-write is fine; write-after-read under serializable is a conflict
        # only when the read predates the write and both agents are distinct.
        reads = [r for r in self.records if r.operation == "read"]
        writes_by_key = {(r.path, r.workspace): r for r in writes}
        for read in reads:
            writer = writes_by_key.get((read.path, read.workspace))
            if writer and writer.agent_id != read.agent_id and isolation == OperationIsolation.SERIALIZABLE:
                conflicts.append({
                    "path": read.path,
                    "workspace": read.workspace,
                    "agents": sorted({read.agent_id, writer.agent_id}),
                    "operations": ["read", writer.operation],
                    "versions": [read.version, writer.version],
                    "isolation": isolation.value,
                    "reason": "read-write ordering conflict",
                })
        return ConflictReport(ok=not conflicts, conflicts=conflicts, artifacts=list(self.records))


_DEFAULT_MANAGER = IsolationManager()


def acquire_artifact(path: str, agent_id: str, **kwargs) -> ArtifactVersion:
    return _DEFAULT_MANAGER.acquire_artifact(path, agent_id, **kwargs)


def release_artifact(path: str, agent_id: str, **kwargs) -> None:
    _DEFAULT_MANAGER.release_artifact(path, agent_id, **kwargs)


def detect_conflicts(**kwargs) -> ConflictReport:
    return _DEFAULT_MANAGER.detect_conflicts(**kwargs)
