"""Canonical Plan Intermediate Representation (IR) and Provenance Tracking."""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class FactTruth(str, Enum):
    """4-state epistemic truth lattice."""
    VERIFIED_TRUE = "VERIFIED_TRUE"
    VERIFIED_FALSE = "VERIFIED_FALSE"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class WitnessabilityStatus(str, Enum):
    """Observability / witnessability classification of world state facts."""
    WITNESSABLE = "WITNESSABLE"
    UNWITNESSABLE = "UNWITNESSABLE"


class SourceType(str, Enum):
    """Provenance source categorization for epistemic grounding."""
    USER_REQUIREMENT = "USER_REQUIREMENT"
    OBSERVED_WORLD_STATE = "OBSERVED_WORLD_STATE"
    CAPABILITY_REGISTRY = "CAPABILITY_REGISTRY"
    DOMAIN_POLICY = "DOMAIN_POLICY"
    PLANNER_INFERENCE = "PLANNER_INFERENCE"
    EXPLICIT_ASSUMPTION = "EXPLICIT_ASSUMPTION"


class Provenance(BaseModel):
    """Metadata tracking origin and confidence of facts and decisions."""
    source_type: SourceType
    source_id: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)
    rationale: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0 inclusive.")
        return float(v)


class WorldFact(BaseModel):
    """Atomic ground truth or epistemic assertion about the environment."""
    predicate: str
    args: List[Any] = Field(default_factory=list)
    truth: FactTruth = FactTruth.VERIFIED_TRUE
    witnessability: WitnessabilityStatus = WitnessabilityStatus.WITNESSABLE
    ttl_seconds: Optional[float] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: Optional[float] = None
    provenance: Provenance
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.updated_at is None:
            self.updated_at = self.created_at

    @property
    def fact_key(self) -> str:
        """Deterministic canonical string key for the predicate and arguments."""
        args_str = ",".join(str(a) for a in self.args)
        return f"{self.predicate}({args_str})"

    def is_fresh(self, current_time: Optional[float] = None) -> bool:
        """Evaluate if the fact remains fresh according to its TTL."""
        if self.ttl_seconds is None:
            return True
        now = current_time if current_time is not None else time.time()
        effective_updated = self.updated_at if self.updated_at is not None else self.created_at
        return (now - effective_updated) <= self.ttl_seconds


class PredicateCondition(BaseModel):
    """Precondition, effect, or invariant condition."""
    predicate: str
    args: List[Any] = Field(default_factory=list)
    expected_truth: FactTruth = FactTruth.VERIFIED_TRUE
    active_until_action_id: Optional[str] = None

    @property
    def fact_key(self) -> str:
        """Deterministic canonical string key matching WorldFact."""
        args_str = ",".join(str(a) for a in self.args)
        return f"{self.predicate}({args_str})"


class HardConstraint(BaseModel):
    """System, safety, or domain invariant that must never be violated."""
    constraint_id: str
    description: str
    condition: PredicateCondition
    enforcement_level: str = "ERROR"
    provenance: Provenance


class SuccessCriterion(BaseModel):
    """Target terminal condition defining plan completion."""
    criterion_id: str
    description: str
    condition: PredicateCondition
    is_mandatory: bool = True


class ActionIR(BaseModel):
    """Single discrete capability invocation step in the plan."""
    action_id: str
    capability_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[PredicateCondition] = Field(default_factory=list)
    positive_effects: List[PredicateCondition] = Field(default_factory=list)
    negative_effects: List[PredicateCondition] = Field(default_factory=list)
    compensation_action_id: Optional[str] = None
    is_idempotent: bool = False
    timeout_seconds: float = 60.0
    provenance: Provenance


class PlanIR(BaseModel):
    """Canonical Plan Representation."""
    plan_id: str
    goal_description: str
    version: int = 1
    initial_state: List[WorldFact] = Field(default_factory=list)
    actions: List[ActionIR] = Field(default_factory=list)
    hard_constraints: List[HardConstraint] = Field(default_factory=list)
    success_criteria: List[SuccessCriterion] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of canonical plan representation."""
        # Serialize fields deterministically excluding dynamic timestamps
        dump_data = {
            "plan_id": self.plan_id,
            "goal_description": self.goal_description,
            "version": self.version,
            "initial_state": [
                {
                    "key": f.fact_key,
                    "truth": f.truth.value,
                    "witnessability": f.witnessability.value,
                    "source": f.provenance.source_type.value,
                }
                for f in self.initial_state
            ],
            "actions": [
                {
                    "action_id": a.action_id,
                    "capability_name": a.capability_name,
                    "parameters": a.parameters,
                    "preconditions": [
                        {"key": p.fact_key, "truth": p.expected_truth.value}
                        for p in a.preconditions
                    ],
                    "positive_effects": [
                        {"key": p.fact_key, "truth": p.expected_truth.value}
                        for p in a.positive_effects
                    ],
                    "negative_effects": [
                        {"key": p.fact_key, "truth": p.expected_truth.value}
                        for p in a.negative_effects
                    ],
                    "is_idempotent": a.is_idempotent,
                }
                for a in self.actions
            ],
            "hard_constraints": [
                {
                    "constraint_id": hc.constraint_id,
                    "key": hc.condition.fact_key,
                    "expected": hc.condition.expected_truth.value,
                }
                for hc in self.hard_constraints
            ],
            "success_criteria": [
                {
                    "criterion_id": sc.criterion_id,
                    "key": sc.condition.fact_key,
                    "expected": sc.condition.expected_truth.value,
                }
                for sc in self.success_criteria
            ],
        }
        serialized = json.dumps(dump_data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def render_markdown_view(plan_ir: PlanIR) -> str:
    """Render canonical PlanIR into structured Markdown for inspection and reporting."""
    lines: List[str] = [
        f"# Plan IR: `{plan_ir.plan_id}` (v{plan_ir.version})",
        f"**Goal**: {plan_ir.goal_description}\n",
        f"**Canonical Hash**: `{plan_ir.compute_hash()}`\n",
        "## Initial World State",
    ]

    if not plan_ir.initial_state:
        lines.append("*No initial facts registered.*")
    else:
        lines.append("| Fact Key | Truth Status | Witnessability | Source | Confidence |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for f in plan_ir.initial_state:
            lines.append(
                f"| `{f.fact_key}` | **{f.truth.value}** | {f.witnessability.value} | {f.provenance.source_type.value} | {f.provenance.confidence:.2f} |"
            )

    lines.append("\n## Action Sequence")
    if not plan_ir.actions:
        lines.append("*No actions scheduled.*")
    else:
        for idx, act in enumerate(plan_ir.actions, 1):
            lines.append(f"### Step {idx}: `{act.action_id}` - `{act.capability_name}`")
            lines.append(f"- **Idempotent**: {act.is_idempotent} | **Timeout**: {act.timeout_seconds}s")
            if act.parameters:
                params_str = ", ".join(f"`{k}={v}`" for k, v in act.parameters.items())
                lines.append(f"- **Parameters**: {params_str}")
            if act.preconditions:
                pre_str = ", ".join(f"`{p.fact_key} == {p.expected_truth.value}`" for p in act.preconditions)
                lines.append(f"- **Preconditions**: {pre_str}")
            if act.positive_effects:
                pos_str = ", ".join(f"`{p.fact_key} := {p.expected_truth.value}`" for p in act.positive_effects)
                lines.append(f"- **Positive Effects**: {pos_str}")
            if act.negative_effects:
                neg_str = ", ".join(f"`{p.fact_key} := {p.expected_truth.value}`" for p in act.negative_effects)
                lines.append(f"- **Negative Effects**: {neg_str}")
            if act.compensation_action_id:
                lines.append(f"- **Compensation Action**: `{act.compensation_action_id}`")
            lines.append("")

    lines.append("## Invariant Hard Constraints")
    if not plan_ir.hard_constraints:
        lines.append("*No hard constraints specified.*")
    else:
        for hc in plan_ir.hard_constraints:
            lines.append(
                f"- **[{hc.constraint_id}]** {hc.description} -> Requires `{hc.condition.fact_key} == {hc.condition.expected_truth.value}`"
            )

    lines.append("\n## Success Criteria")
    if not plan_ir.success_criteria:
        lines.append("*No success criteria specified.*")
    else:
        for sc in plan_ir.success_criteria:
            mandatory_tag = "[MANDATORY]" if sc.is_mandatory else "[OPTIONAL]"
            lines.append(
                f"- **{mandatory_tag} [{sc.criterion_id}]** {sc.description} -> Requires `{sc.condition.fact_key} == {sc.condition.expected_truth.value}`"
            )

    return "\n".join(lines)
