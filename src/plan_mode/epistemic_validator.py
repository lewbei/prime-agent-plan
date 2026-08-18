"""4-State Fact Lattice and Causal Validator for Canonical Plan IR."""

from __future__ import annotations

import copy
import time
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import (
    ActionIR,
    FactTruth,
    HardConstraint,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WorldFact,
)
from plan_mode.registry import CapabilityRegistry


class ValidationStatus(str, Enum):
    """Feasibility status resulting from causal validation."""
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


def merge_fact_truth(a: FactTruth, b: FactTruth) -> FactTruth:
    """Merge two truth values in the 4-state lattice."""
    if a == b:
        return a
    if a == FactTruth.UNKNOWN:
        return b
    if b == FactTruth.UNKNOWN:
        return a
    if a == FactTruth.CONFLICT or b == FactTruth.CONFLICT:
        return FactTruth.CONFLICT
    # One is TRUE, one is FALSE
    return FactTruth.CONFLICT


class PlanValidationResult(BaseModel):
    """Detailed outcome of causal forward validation."""
    status: ValidationStatus
    failed_step_id: Optional[str] = None
    failed_predicate: Optional[str] = None
    blocker_reasons: List[str] = Field(default_factory=list)
    unknown_facts: List[str] = Field(default_factory=list)
    invariants_violated: List[str] = Field(default_factory=list)
    criteria_satisfied: List[str] = Field(default_factory=list)
    criteria_unmet: List[str] = Field(default_factory=list)
    intermediate_states: List[Dict[str, WorldFact]] = Field(default_factory=list)


class EpistemicCausalValidator:
    """Deterministic forward simulator and invariant checker over 4-state fact lattice."""

    def __init__(self, default_ttl_decay_to_unknown: bool = True):
        self.default_ttl_decay_to_unknown = default_ttl_decay_to_unknown

    def validate_plan(
        self,
        plan_ir: PlanIR,
        registry: Optional[CapabilityRegistry] = None,
        current_time: Optional[float] = None,
    ) -> PlanValidationResult:
        now = current_time if current_time is not None else time.time()
        
        # 1. Initialize world state W_0 from initial facts with 4-state lattice merging
        current_state: Dict[str, WorldFact] = {}
        blocker_reasons: List[str] = []
        unknown_facts: List[str] = []
        invariants_violated: List[str] = []
        first_unknown_step: Optional[str] = None

        for fact in plan_ir.initial_state:
            f_clone = fact.model_copy(deep=True)
            if self.default_ttl_decay_to_unknown and not f_clone.is_fresh(now):
                f_clone.truth = FactTruth.UNKNOWN
                f_clone.provenance = Provenance(
                    source_type=SourceType.OBSERVED_WORLD_STATE,
                    confidence=0.0,
                    rationale="Fact TTL expired; decayed to UNKNOWN",
                )
            key = f_clone.fact_key
            if key in current_state:
                merged_truth = merge_fact_truth(current_state[key].truth, f_clone.truth)
                current_state[key].truth = merged_truth
                if merged_truth == FactTruth.CONFLICT:
                    current_state[key].provenance = Provenance(
                        source_type=SourceType.OBSERVED_WORLD_STATE,
                        confidence=0.0,
                        rationale=f"Contradictory duplicate fact definitions for '{key}'",
                    )
            else:
                current_state[key] = f_clone

        # If any initial fact is in CONFLICT, fail immediately
        for key, f in current_state.items():
            if f.truth == FactTruth.CONFLICT:
                blocker_reasons.append(f"Initial state conflict / contradiction for fact '{key}'")
                return PlanValidationResult(
                    status=ValidationStatus.FAIL,
                    failed_step_id="INITIAL_STATE",
                    failed_predicate=key,
                    blocker_reasons=blocker_reasons,
                    intermediate_states=[copy.deepcopy(current_state)],
                )

        intermediate_states: List[Dict[str, WorldFact]] = [copy.deepcopy(current_state)]

        # Check invariants on initial state W_0
        self._check_invariants(
            current_state,
            plan_ir.hard_constraints,
            action_idx=0,
            plan_actions=plan_ir.actions,
            invariants_violated=invariants_violated,
            blocker_reasons=blocker_reasons,
        )
        if invariants_violated:
            return PlanValidationResult(
                status=ValidationStatus.FAIL,
                failed_step_id="INITIAL_STATE",
                blocker_reasons=blocker_reasons,
                invariants_violated=invariants_violated,
                intermediate_states=intermediate_states,
            )

        # 2. Iterate through each action step
        for idx, action in enumerate(plan_ir.actions):
            step_id = action.action_id
            cap_entry = None

            # If registry provided, validate action schema and effect declarations
            if registry is not None:
                try:
                    registry.validate_action(action)
                    cap_entry = registry.get(action.capability_name)
                except Exception as e:
                    return PlanValidationResult(
                        status=ValidationStatus.FAIL,
                        failed_step_id=step_id,
                        blocker_reasons=[f"Capability/schema validation failed: {str(e)}"],
                        intermediate_states=intermediate_states,
                    )

            # Check preconditions against current state W_t
            step_unknown = False

            for pre in action.preconditions:
                key = pre.fact_key
                fact = current_state.get(key)
                
                # If not present in state, it is implicitly UNKNOWN
                fact_truth = fact.truth if fact is not None else FactTruth.UNKNOWN

                if fact_truth == FactTruth.CONFLICT:
                    reason = f"Precondition conflict for '{key}' on action '{step_id}'"
                    blocker_reasons.append(reason)
                    return PlanValidationResult(
                        status=ValidationStatus.FAIL,
                        failed_step_id=step_id,
                        failed_predicate=key,
                        blocker_reasons=blocker_reasons,
                        intermediate_states=intermediate_states,
                    )
                elif fact_truth == FactTruth.UNKNOWN:
                    step_unknown = True
                    if key not in unknown_facts:
                        unknown_facts.append(key)
                    if first_unknown_step is None:
                        first_unknown_step = step_id
                elif fact_truth != pre.expected_truth:
                    reason = f"Precondition failed on '{step_id}': expected {key} == {pre.expected_truth.value}, got {fact_truth.value}"
                    blocker_reasons.append(reason)
                    return PlanValidationResult(
                        status=ValidationStatus.FAIL,
                        failed_step_id=step_id,
                        failed_predicate=key,
                        blocker_reasons=blocker_reasons,
                        intermediate_states=intermediate_states,
                    )

            # 3. Transition: W_{t+1}
            # INVARIANT: If precondition is UNKNOWN or CONFLICT, effects MUST NOT be applied as VERIFIED
            next_state = copy.deepcopy(current_state)
            
            if not step_unknown:
                # Apply negative effects
                for neg in action.negative_effects:
                    neg_key = neg.fact_key
                    next_state[neg_key] = WorldFact(
                        predicate=neg.predicate,
                        args=neg.args,
                        truth=neg.expected_truth if neg.expected_truth != FactTruth.VERIFIED_TRUE else FactTruth.VERIFIED_FALSE,
                        created_at=now,
                        updated_at=now,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            rationale=f"Negative effect of {step_id}",
                        ),
                    )

                # Apply positive effects (checking verifier presence if registry available)
                for pos in action.positive_effects:
                    pos_key = pos.fact_key
                    # If registry is active, check if a verifier exists for this effect
                    has_verifier = True
                    if cap_entry is not None:
                        has_verifier = any(v.predicate == pos.predicate for v in cap_entry.verifiers)

                    if has_verifier:
                        effect_truth = pos.expected_truth
                        rationale = f"Positive effect of {step_id}"
                    else:
                        effect_truth = FactTruth.UNKNOWN
                        rationale = f"Positive effect of {step_id} (missing observation verifier in capability)"
                        if pos_key not in unknown_facts:
                            unknown_facts.append(pos_key)
                        if first_unknown_step is None:
                            first_unknown_step = step_id

                    next_state[pos_key] = WorldFact(
                        predicate=pos.predicate,
                        args=pos.args,
                        truth=effect_truth,
                        created_at=now,
                        updated_at=now,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            rationale=rationale,
                        ),
                    )
            else:
                # Precondition was UNKNOWN: action cannot produce verified effects
                for pos in action.positive_effects:
                    pos_key = pos.fact_key
                    if pos_key not in unknown_facts:
                        unknown_facts.append(pos_key)
                    next_state[pos_key] = WorldFact(
                        predicate=pos.predicate,
                        args=pos.args,
                        truth=FactTruth.UNKNOWN,
                        created_at=now,
                        updated_at=now,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            confidence=0.0,
                            rationale=f"Uncertain effect due to UNKNOWN precondition on {step_id}",
                        ),
                    )

            current_state = next_state
            intermediate_states.append(copy.deepcopy(current_state))

            # Check invariants on intermediate state W_{t+1}
            self._check_invariants(
                current_state,
                plan_ir.hard_constraints,
                action_idx=idx + 1,
                plan_actions=plan_ir.actions,
                invariants_violated=invariants_violated,
                blocker_reasons=blocker_reasons,
            )
            if invariants_violated:
                return PlanValidationResult(
                    status=ValidationStatus.FAIL,
                    failed_step_id=step_id,
                    blocker_reasons=blocker_reasons,
                    invariants_violated=invariants_violated,
                    intermediate_states=intermediate_states,
                )

        # 4. Check Success Criteria on final state W_T
        criteria_satisfied: List[str] = []
        criteria_unmet: List[str] = []

        for crit in plan_ir.success_criteria:
            key = crit.condition.fact_key
            fact = current_state.get(key)
            fact_truth = fact.truth if fact is not None else FactTruth.UNKNOWN

            if fact_truth == crit.condition.expected_truth:
                criteria_satisfied.append(crit.criterion_id)
            else:
                criteria_unmet.append(crit.criterion_id)
                if crit.is_mandatory:
                    if fact_truth == FactTruth.UNKNOWN:
                        if key not in unknown_facts:
                            unknown_facts.append(key)
                    else:
                        blocker_reasons.append(
                            f"Mandatory success criterion '{crit.criterion_id}' unmet: expected {key} == {crit.condition.expected_truth.value}, got {fact_truth.value}"
                        )

        # 5. Determine overall verdict
        if blocker_reasons or invariants_violated:
            return PlanValidationResult(
                status=ValidationStatus.FAIL,
                failed_step_id=first_unknown_step,
                blocker_reasons=blocker_reasons,
                unknown_facts=unknown_facts,
                invariants_violated=invariants_violated,
                criteria_satisfied=criteria_satisfied,
                criteria_unmet=criteria_unmet,
                intermediate_states=intermediate_states,
            )

        if unknown_facts:
            return PlanValidationResult(
                status=ValidationStatus.UNKNOWN,
                failed_step_id=first_unknown_step,
                unknown_facts=unknown_facts,
                criteria_satisfied=criteria_satisfied,
                criteria_unmet=criteria_unmet,
                intermediate_states=intermediate_states,
            )

        if criteria_unmet and any(c.is_mandatory for c in plan_ir.success_criteria if c.criterion_id in criteria_unmet):
            return PlanValidationResult(
                status=ValidationStatus.FAIL,
                blocker_reasons=["One or more mandatory success criteria unmet."],
                criteria_satisfied=criteria_satisfied,
                criteria_unmet=criteria_unmet,
                intermediate_states=intermediate_states,
            )

        return PlanValidationResult(
            status=ValidationStatus.PASS,
            criteria_satisfied=criteria_satisfied,
            criteria_unmet=criteria_unmet,
            intermediate_states=intermediate_states,
        )

    def _check_invariants(
        self,
        state: Dict[str, WorldFact],
        constraints: List[HardConstraint],
        action_idx: int,
        plan_actions: List[ActionIR],
        invariants_violated: List[str],
        blocker_reasons: List[str],
    ) -> None:
        for hc in constraints:
            if hc.active_until_action_id:
                until_idx = None
                for i, act in enumerate(plan_actions):
                    if act.action_id == hc.active_until_action_id:
                        until_idx = i
                        break
                # Constraint is active from step 0 up to execution of until_idx
                # When action_idx (1-based after step execution) > until_idx, the constraint is inactive
                if until_idx is not None and action_idx > until_idx:
                    continue

            key = hc.condition.fact_key
            fact = state.get(key)
            fact_truth = fact.truth if fact is not None else FactTruth.UNKNOWN
            if fact_truth != hc.condition.expected_truth:
                invariants_violated.append(hc.constraint_id)
                blocker_reasons.append(
                    f"Hard constraint '{hc.constraint_id}' violated: expected {key} == {hc.condition.expected_truth.value}, got {fact_truth.value}"
                )


EpistemicValidator = EpistemicCausalValidator
CausalValidator = EpistemicCausalValidator
