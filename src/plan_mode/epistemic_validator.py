"""4-State Fact Lattice and Causal Validator for Canonical Plan IR."""

from __future__ import annotations

import copy
import time
from enum import Enum
from typing import Any, Dict, List, Optional
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
    WitnessabilityStatus,
    WorldFact,
)
from plan_mode.registry import (
    CapabilityEntry,
    CapabilityNotFoundError,
    CapabilityRegistry,
    SchemaMismatchError,
    typed_args_equal,
)


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


def _matches_verifier(
    cond: PredicateCondition,
    cap_entry: Optional[CapabilityEntry],
    params: Dict[str, Any],
    declared_effects: List[PredicateCondition],
) -> bool:
    """Check whether cond has a matching observation verifier bound to its exact predicate and arguments."""
    if cap_entry is None:
        return False

    for v in cap_entry.verifiers:
        if v.predicate != cond.predicate:
            continue

        # If verifier has explicit target_args_mapping, resolve it and compare
        if v.target_args_mapping:
            resolved_args: List[Any] = []
            for arg in v.target_args_mapping:
                if isinstance(arg, str):
                    if arg.startswith("{") and arg.endswith("}"):
                        var_name = arg[1:-1]
                        resolved_args.append(params.get(var_name, arg))
                    elif arg.startswith("$"):
                        var_name = arg[1:]
                        resolved_args.append(params.get(var_name, arg))
                    elif arg in params:
                        resolved_args.append(params[arg])
                    else:
                        resolved_args.append(arg)
                else:
                    resolved_args.append(arg)

            if typed_args_equal(resolved_args, cond.args):
                return True
        else:
            # target_args_mapping omitted: check if cond.args is empty or matches declared instantiated effect
            if not cond.args:
                return True
            for decl in declared_effects:
                if decl.predicate == cond.predicate:
                    inst_args: List[Any] = []
                    for arg in decl.args:
                        if isinstance(arg, str):
                            if arg.startswith("{") and arg.endswith("}"):
                                var_name = arg[1:-1]
                                inst_args.append(params.get(var_name, arg))
                            elif arg.startswith("$"):
                                var_name = arg[1:]
                                inst_args.append(params.get(var_name, arg))
                            elif arg in params:
                                inst_args.append(params[arg])
                            else:
                                inst_args.append(arg)
                        else:
                            inst_args.append(arg)
                    if typed_args_equal(inst_args, cond.args):
                        return True
    return False


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
            unknown_facts=unknown_facts,
        )
        if invariants_violated or blocker_reasons:
            return PlanValidationResult(
                status=ValidationStatus.FAIL,
                failed_step_id="INITIAL_STATE",
                blocker_reasons=blocker_reasons,
                unknown_facts=unknown_facts,
                invariants_violated=invariants_violated,
                intermediate_states=intermediate_states,
            )

        current_sim_time = now

        # 2. Iterate through each action step
        for idx, action in enumerate(plan_ir.actions):
            step_id = action.action_id
            cap_entry = None
            step_unknown = False

            # Advance simulated time by duration of previous action step
            if idx > 0:
                current_sim_time += plan_ir.actions[idx - 1].timeout_seconds

            # Recheck fact freshness at point-of-use (t = current_sim_time)
            for key, f in list(current_state.items()):
                if self.default_ttl_decay_to_unknown and not f.is_fresh(current_sim_time):
                    if f.truth != FactTruth.UNKNOWN:
                        f_decayed = f.model_copy(deep=True)
                        f_decayed.truth = FactTruth.UNKNOWN
                        f_decayed.provenance = Provenance(
                            source_type=SourceType.OBSERVED_WORLD_STATE,
                            confidence=0.0,
                            rationale=f"Fact TTL expired at t={current_sim_time}; decayed to UNKNOWN",
                        )
                        current_state[key] = f_decayed

            # Validate capability against registry
            has_effects = bool(action.positive_effects or action.negative_effects)
            if registry is not None:
                try:
                    registry.validate_action(action)
                    cap_entry = registry.get(action.capability_name)
                except CapabilityNotFoundError:
                    # Unregistered capability is lack of knowledge (UNKNOWN), not a proven contradiction (FAIL)
                    step_unknown = True
                    if first_unknown_step is None:
                        first_unknown_step = step_id
                    for pos in action.positive_effects:
                        if pos.fact_key not in unknown_facts:
                            unknown_facts.append(pos.fact_key)
                    for neg in action.negative_effects:
                        if neg.fact_key not in unknown_facts:
                            unknown_facts.append(neg.fact_key)
                    for pre in action.preconditions:
                        if pre.fact_key not in unknown_facts:
                            unknown_facts.append(pre.fact_key)
                except SchemaMismatchError as e:
                    # Known malformed capability contract is FAIL
                    return PlanValidationResult(
                        status=ValidationStatus.FAIL,
                        failed_step_id=step_id,
                        blocker_reasons=[f"Capability contract validation failed on '{step_id}': {str(e)}"],
                        intermediate_states=intermediate_states,
                    )
            elif has_effects:
                # Effectful action with registry=None lacks grounding -> UNKNOWN
                step_unknown = True
                if first_unknown_step is None:
                    first_unknown_step = step_id
                missing_reg_tag = f"missing_registry_grounding({step_id})"
                if missing_reg_tag not in unknown_facts:
                    unknown_facts.append(missing_reg_tag)
                for pos in action.positive_effects:
                    if pos.fact_key not in unknown_facts:
                        unknown_facts.append(pos.fact_key)
                for neg in action.negative_effects:
                    if neg.fact_key not in unknown_facts:
                        unknown_facts.append(neg.fact_key)

            # Check preconditions against current state W_t
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
            next_state = copy.deepcopy(current_state)

            if not step_unknown:
                # Apply negative effects (checking verifier presence and arg binding)
                for neg in action.negative_effects:
                    neg_key = neg.fact_key
                    has_neg_verifier = _matches_verifier(
                        neg,
                        cap_entry,
                        action.parameters,
                        cap_entry.negative_effects if cap_entry else [],
                    )

                    expected_truth_val = (
                        FactTruth.VERIFIED_FALSE.value
                        if neg.expected_truth == FactTruth.VERIFIED_TRUE
                        else neg.expected_truth.value
                    )

                    if has_neg_verifier:
                        neg_truth = FactTruth.UNKNOWN
                        neg_witness = WitnessabilityStatus.WITNESSABLE
                        neg_rationale = f"Negative effect of {step_id} (witnessable, pending runtime execution)"
                    else:
                        neg_truth = FactTruth.UNKNOWN
                        neg_witness = WitnessabilityStatus.UNWITNESSABLE
                        neg_rationale = f"Negative effect of {step_id} (missing observation verifier)"
                        if neg_key not in unknown_facts:
                            unknown_facts.append(neg_key)
                        if first_unknown_step is None:
                            first_unknown_step = step_id

                    next_state[neg_key] = WorldFact(
                        predicate=neg.predicate,
                        args=neg.args,
                        truth=neg_truth,
                        witnessability=neg_witness,
                        created_at=current_sim_time,
                        updated_at=current_sim_time,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            confidence=1.0 if has_neg_verifier else 0.0,
                            rationale=neg_rationale,
                        ),
                        metadata={"predicted_truth": expected_truth_val},
                    )

                # Apply positive effects (checking verifier presence and arg binding)
                for pos in action.positive_effects:
                    pos_key = pos.fact_key
                    has_pos_verifier = _matches_verifier(
                        pos,
                        cap_entry,
                        action.parameters,
                        cap_entry.positive_effects if cap_entry else [],
                    )

                    if has_pos_verifier:
                        pos_truth = FactTruth.UNKNOWN
                        pos_witness = WitnessabilityStatus.WITNESSABLE
                        pos_rationale = f"Positive effect of {step_id} (witnessable, pending runtime execution)"
                    else:
                        pos_truth = FactTruth.UNKNOWN
                        pos_witness = WitnessabilityStatus.UNWITNESSABLE
                        pos_rationale = f"Positive effect of {step_id} (missing observation verifier)"
                        if pos_key not in unknown_facts:
                            unknown_facts.append(pos_key)
                        if first_unknown_step is None:
                            first_unknown_step = step_id

                    next_state[pos_key] = WorldFact(
                        predicate=pos.predicate,
                        args=pos.args,
                        truth=pos_truth,
                        witnessability=pos_witness,
                        created_at=current_sim_time,
                        updated_at=current_sim_time,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            confidence=1.0 if has_pos_verifier else 0.0,
                            rationale=pos_rationale,
                        ),
                        metadata={"predicted_truth": pos.expected_truth.value},
                    )
            else:
                # Precondition was UNKNOWN or capability ungrounded: action cannot produce verified effects
                for pos in action.positive_effects:
                    pos_key = pos.fact_key
                    if pos_key not in unknown_facts:
                        unknown_facts.append(pos_key)
                    next_state[pos_key] = WorldFact(
                        predicate=pos.predicate,
                        args=pos.args,
                        truth=FactTruth.UNKNOWN,
                        witnessability=WitnessabilityStatus.UNWITNESSABLE,
                        created_at=current_sim_time,
                        updated_at=current_sim_time,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            confidence=0.0,
                            rationale=f"Uncertain effect due to UNKNOWN precondition / ungrounded capability on {step_id}",
                        ),
                        metadata={"predicted_truth": pos.expected_truth.value},
                    )
                for neg in action.negative_effects:
                    neg_key = neg.fact_key
                    if neg_key not in unknown_facts:
                        unknown_facts.append(neg_key)
                    next_state[neg_key] = WorldFact(
                        predicate=neg.predicate,
                        args=neg.args,
                        truth=FactTruth.UNKNOWN,
                        witnessability=WitnessabilityStatus.UNWITNESSABLE,
                        created_at=current_sim_time,
                        updated_at=current_sim_time,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            confidence=0.0,
                            rationale=f"Uncertain effect due to UNKNOWN precondition / ungrounded capability on {step_id}",
                        ),
                        metadata={"predicted_truth": neg.expected_truth.value},
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
                unknown_facts=unknown_facts,
            )
            if invariants_violated:
                return PlanValidationResult(
                    status=ValidationStatus.FAIL,
                    failed_step_id=step_id,
                    blocker_reasons=blocker_reasons,
                    unknown_facts=unknown_facts,
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
        unknown_facts: List[str],
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
            predicted_truth = fact.metadata.get("predicted_truth") if fact is not None else None

            if fact_truth == FactTruth.UNKNOWN:
                if predicted_truth is not None and predicted_truth != hc.condition.expected_truth.value:
                    invariants_violated.append(hc.constraint_id)
                    blocker_reasons.append(
                        f"Hard constraint '{hc.constraint_id}' violated: expected {key} == {hc.condition.expected_truth.value}, predicted {predicted_truth}"
                    )
                else:
                    if key not in unknown_facts:
                        unknown_facts.append(key)
            elif fact_truth != hc.condition.expected_truth:
                invariants_violated.append(hc.constraint_id)
                blocker_reasons.append(
                    f"Hard constraint '{hc.constraint_id}' violated: expected {key} == {hc.condition.expected_truth.value}, got {fact_truth.value}"
                )


EpistemicValidator = EpistemicCausalValidator
CausalValidator = EpistemicCausalValidator
