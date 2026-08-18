"""Five-node drift recovery graph (2608.14109).

Paper routing rules:
    n1 -> n2 when the audited step is not aligned
    n1 -> n5 when the last K alignment verdicts are all aligned
    otherwise walk backward one step and re-run n1
    n2 -> n1 when no write operations were detected
    n2 -> n3 when writes were detected
    n3 -> n4 unconditionally
    n4 -> n1 unconditionally

n1: {step, is_aligned, why}
n2: {write_operations, read_operations}
n3: {apps_name}
n4: {write_reversible, write_not_reversible}
n5: {action, arguments}

This implementation is deterministic. An optional `policy` callable can be
injected later for the learned small-model specialisation described in the
paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class DriftEvidence:
    task: str = ""
    step: int = 1
    suspected_onset: int = 1
    is_aligned: Optional[bool] = None
    why: str = ""
    write_operations: list[str] = field(default_factory=list)
    read_operations: list[str] = field(default_factory=list)
    apps_name: str = ""
    write_reversible: list[str] = field(default_factory=list)
    write_not_reversible: list[str] = field(default_factory=list)
    checkpoint_available: bool = True
    risk: float = 0.0


@dataclass
class RecoveryDecision:
    action: str            # retry | rewind | replan | abort
    node_path: list[str]
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def classify_drift(evidence: DriftEvidence, *, policy: Optional[Callable[[DriftEvidence], dict[str, Any]]] = None) -> dict[str, Any]:
    """Node n1: alignment classification."""
    if policy is not None:
        out = policy(evidence)
        if isinstance(out, dict):
            return out
    if evidence.is_aligned is None:
        evidence.is_aligned = not bool(evidence.write_not_reversible) and evidence.risk < 0.5
    return {
        "node": "n1",
        "step": evidence.step,
        "is_aligned": evidence.is_aligned,
        "why": evidence.why or ("risk low" if evidence.is_aligned else "drift signal present"),
    }


def recovery_decision(evidence: DriftEvidence, *, K: int = 2,
                      policy: Optional[Callable[[DriftEvidence], dict[str, Any]]] = None) -> RecoveryDecision:
    """Route evidence through n1..n5 and return the terminal recovery action."""
    path: list[str] = []
    aligned_streak: list[bool] = []
    current = evidence
    current.step = max(1, int(current.step))
    current.suspected_onset = max(1, int(current.suspected_onset))

    for _ in range(max(1, current.step - current.suspected_onset + 1) + 3):
        n1 = classify_drift(current, policy=policy)
        path.append(n1["node"])
        aligned_streak.append(bool(n1.get("is_aligned")))
        if not n1.get("is_aligned"):
            if not current.write_operations:
                path.append("n2")
                current.step -= 1
                continue
            path.append("n2")
            path.append("n3")
            path.append("n4")
            break
        if len(aligned_streak) >= K and all(aligned_streak[-K:]):
            break
        current.step -= 1

    path.append("n5")
    if evidence.write_not_reversible and evidence.risk >= 0.5:
        action = "rewind" if evidence.checkpoint_available else "abort"
        reason = "non-reversible writes with high drift risk"
    elif evidence.risk >= 0.5:
        action = "replan"
        reason = "high drift risk despite reversible writes"
    elif evidence.write_operations:
        action = "retry"
        reason = "drift is reversible and risk is bounded"
    else:
        action = "retry"
        reason = "no write operations detected; retry the aligned step"

    return RecoveryDecision(
        action=action,
        node_path=path,
        arguments={"step": evidence.step, "onset": evidence.suspected_onset,
                   "apps_name": evidence.apps_name, "K": K},
        reason=reason,
    )


class RecoveryGraph:
    """Public wrapper around the paper's five-node recovery graph."""

    def __init__(self, K: int = 2, policy: Optional[Callable[[DriftEvidence], dict[str, Any]]] = None):
        self.K = K
        self.policy = policy
        self.history: list[RecoveryDecision] = []

    def decide(self, evidence: DriftEvidence) -> RecoveryDecision:
        decision = recovery_decision(evidence, K=self.K, policy=self.policy)
        self.history.append(decision)
        return decision
