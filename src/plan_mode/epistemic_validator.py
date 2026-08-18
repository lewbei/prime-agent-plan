"""4-State Fact Lattice, Projected Causal State, and Epistemic Causal Validator for Plan IR."""

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
    ProjectedTruth,
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


def normalize_trusted_snapshot(
    observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]],
    default_ttl_decay_to_unknown: bool = True,
    now: Optional[float] = None,
) -> Dict[str, WorldFact]:
    """Normalize and canonicalize trusted observations using 4-state lattice merging and TTL decay."""
    if observed_world_state is None:
        return {}

    current_time = now if now is not None else time.time()
    trusted_map: Dict[str, WorldFact] = {}

    # Extract all facts whether supplied as list or dict (canonicalizing by fact_key to prevent aliasing)
    facts_iterable = (
        observed_world_state.values()
        if isinstance(observed_world_state, dict)
        else observed_world_state
    )

    for f in facts_iterable:
        f_clone = f.model_copy(deep=True)
        key = f_clone.fact_key

        # Check TTL decay
        if default_ttl_decay_to_unknown and not f_clone.is_fresh(current_time):
            f_clone.truth = FactTruth.UNKNOWN
            f_clone.projected_truth = ProjectedTruth.UNSUPPORTED
            f_clone.provenance = Provenance(
                source_type=SourceType.OBSERVED_WORLD_STATE,
                confidence=0.0,
                rationale="Fact TTL expired; decayed to UNKNOWN",
            )
        else:
            if f_clone.truth == FactTruth.VERIFIED_TRUE:
                f_clone.projected_truth = ProjectedTruth.SUPPORTED_TRUE
            elif f_clone.truth == FactTruth.VERIFIED_FALSE:
                f_clone.projected_truth = ProjectedTruth.SUPPORTED_FALSE
            elif f_clone.truth == FactTruth.CONFLICT:
                f_clone.projected_truth = ProjectedTruth.CONFLICT
            else:
                f_clone.projected_truth = ProjectedTruth.UNSUPPORTED

        if key in trusted_map:
            # 4-state lattice merge across duplicate trusted observations
            existing = trusted_map[key]
            merged_truth = merge_fact_truth(existing.truth, f_clone.truth)
            existing.truth = merged_truth
            if merged_truth == FactTruth.VERIFIED_TRUE:
                existing.projected_truth = ProjectedTruth.SUPPORTED_TRUE
            elif merged_truth == FactTruth.VERIFIED_FALSE:
                existing.projected_truth = ProjectedTruth.SUPPORTED_FALSE
            elif merged_truth == FactTruth.CONFLICT:
                existing.projected_truth = ProjectedTruth.CONFLICT
                existing.provenance = Provenance(
                    source_type=SourceType.OBSERVED_WORLD_STATE,
                    confidence=0.0,
                    rationale=f"Contradictory duplicate observations in trusted snapshot for '{key}'",
                )
            else:
                existing.projected_truth = ProjectedTruth.UNSUPPORTED
        else:
            trusted_map[key] = f_clone

    return trusted_map


def _matches_verifier(
    cond: PredicateCondition,
    cap_entry: Optional[CapabilityEntry],
    params: Dict[str, Any],
) -> bool:
    """Check whether cond has a matching observation verifier bound to its exact predicate and arguments."""
    if cap_entry is None:
        return False

    for v in cap_entry.verifiers:
        if v.predicate != cond.predicate:
            continue

        # If condition has arguments: target_args_mapping MUST be non-empty and match
        if cond.args:
            if not v.target_args_mapping:
                # Argument-bearing predicate without explicit target_args_mapping is UNWITNESSABLE
                continue
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
            # Zero-argument predicate: matches if target_args_mapping is empty or resolves to empty
            if not v.target_args_mapping:
                return True
            resolved_args = []
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
            if typed_args_equal(resolved_args, []):
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
    """Deterministic forward simulator and invariant checker over dual empirical & projected lattices."""

    def __init__(self, default_ttl_decay_to_unknown: bool = True):
        self.default_ttl_decay_to_unknown = default_ttl_decay_to_unknown

    def validate_plan(
        self,
        plan_ir: PlanIR,
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        current_time: Optional[float] = None,
    ) -> PlanValidationResult:
        now = current_time if current_time is not None else time.time()

        # 1. Normalize trusted snapshot with lattice duplicate merging and canonical fact_key indexing
        trusted_map: Dict[str, WorldFact] = normalize_trusted_snapshot(
            observed_world_state,
            default_ttl_decay_to_unknown=self.default_ttl_decay_to_unknown,
            now=now,
        )

        current_state: Dict[str, WorldFact] = copy.deepcopy(trusted_map)
        blocker_reasons: List[str] = []
        unknown_facts: List[str] = []
        invariants_violated: List[str] = []
        first_unknown_step: Optional[str] = None

        # Check if trusted snapshot itself contained conflicting facts
        for key, f in current_state.items():
            if f.truth == FactTruth.CONFLICT or f.projected_truth == ProjectedTruth.CONFLICT:
                reason = f"Trusted world state conflict / contradiction for fact '{key}'"
                if reason not in blocker_reasons:
                    blocker_reasons.append(reason)

        if blocker_reasons:
            return PlanValidationResult(
                status=ValidationStatus.FAIL,
                failed_step_id="INITIAL_STATE",
                failed_predicate=blocker_reasons[0].split("'")[1] if "'" in blocker_reasons[0] else None,
                blocker_reasons=blocker_reasons,
                intermediate_states=[copy.deepcopy(current_state)],
            )

        # 2. Process initial facts from plan_ir against trusted boundary
        for fact in plan_ir.initial_state:
            f_clone = fact.model_copy(deep=True)
            key = f_clone.fact_key

            if observed_world_state is not None:
                if key in trusted_map:
                    trusted_f = trusted_map[key]
                    merged_truth = merge_fact_truth(trusted_f.truth, f_clone.truth)
                    if merged_truth == FactTruth.CONFLICT:
                        current_state[key].truth = FactTruth.CONFLICT
                        current_state[key].projected_truth = ProjectedTruth.CONFLICT
                        current_state[key].provenance = Provenance(
                            source_type=SourceType.OBSERVED_WORLD_STATE,
                            confidence=0.0,
                            rationale=f"Contradictory fact definitions for '{key}' against trusted world state",
                        )
                    else:
                        current_state[key].truth = trusted_f.truth
                        current_state[key].provenance = trusted_f.provenance
                        current_state[key].projected_truth = trusted_f.projected_truth
                else:
                    # Omitted from trusted snapshot -> untrusted assumption
                    f_clone.truth = FactTruth.UNKNOWN
                    f_clone.projected_truth = ProjectedTruth.UNSUPPORTED
                    f_clone.provenance = Provenance(
                        source_type=SourceType.EXPLICIT_ASSUMPTION,
                        confidence=0.0,
                        rationale="Planner assumption ungrounded by trusted observation snapshot",
                    )
                    current_state[key] = f_clone
            else:
                # No trusted snapshot provided: all initial claims are untrusted
                f_clone.truth = FactTruth.UNKNOWN
                f_clone.projected_truth = ProjectedTruth.UNSUPPORTED
                f_clone.provenance = Provenance(
                    source_type=SourceType.EXPLICIT_ASSUMPTION,
                    confidence=0.0,
                    rationale="Initial state fact ungrounded by trusted observed world state snapshot",
                )
                current_state[key] = f_clone

        # If any initial fact in current_state is in CONFLICT, fail immediately
        for key, f in current_state.items():
            if f.truth == FactTruth.CONFLICT or f.projected_truth == ProjectedTruth.CONFLICT:
                reason = f"Initial state conflict / contradiction for fact '{key}'"
                if reason not in blocker_reasons:
                    blocker_reasons.append(reason)

        if blocker_reasons:
            return PlanValidationResult(
                status=ValidationStatus.FAIL,
                failed_step_id="INITIAL_STATE",
                failed_predicate=blocker_reasons[0].split("'")[1] if "'" in blocker_reasons[0] else None,
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

        # 3. Iterate through each action step
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
                        f_decayed.projected_truth = ProjectedTruth.UNSUPPORTED
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
                    return PlanValidationResult(
                        status=ValidationStatus.FAIL,
                        failed_step_id=step_id,
                        blocker_reasons=[f"Capability contract validation failed on '{step_id}': {str(e)}"],
                        intermediate_states=intermediate_states,
                    )
            elif has_effects:
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

            # Check preconditions against current state W_t (dual empirical/projected check)
            for pre in action.preconditions:
                key = pre.fact_key
                fact = current_state.get(key)

                emp_truth = fact.truth if fact is not None else FactTruth.UNKNOWN
                proj_truth = fact.projected_truth if fact is not None else ProjectedTruth.UNSUPPORTED

                if emp_truth == FactTruth.CONFLICT or proj_truth == ProjectedTruth.CONFLICT:
                    reason = f"Precondition conflict for '{key}' on action '{step_id}'"
                    blocker_reasons.append(reason)
                    return PlanValidationResult(
                        status=ValidationStatus.FAIL,
                        failed_step_id=step_id,
                        failed_predicate=key,
                        blocker_reasons=blocker_reasons,
                        intermediate_states=intermediate_states,
                    )

                if pre.expected_truth == FactTruth.VERIFIED_TRUE:
                    if proj_truth == ProjectedTruth.SUPPORTED_TRUE or emp_truth == FactTruth.VERIFIED_TRUE:
                        # Precondition is causally supported at plan-time
                        continue
                    elif proj_truth == ProjectedTruth.SUPPORTED_FALSE or emp_truth == FactTruth.VERIFIED_FALSE:
                        reason = f"Precondition failed on '{step_id}': expected {key} == VERIFIED_TRUE, got {emp_truth.value}/{proj_truth.value}"
                        blocker_reasons.append(reason)
                        return PlanValidationResult(
                            status=ValidationStatus.FAIL,
                            failed_step_id=step_id,
                            failed_predicate=key,
                            blocker_reasons=blocker_reasons,
                            intermediate_states=intermediate_states,
                        )
                    else:
                        step_unknown = True
                        if key not in unknown_facts:
                            unknown_facts.append(key)
                        if first_unknown_step is None:
                            first_unknown_step = step_id

                elif pre.expected_truth == FactTruth.VERIFIED_FALSE:
                    if proj_truth == ProjectedTruth.SUPPORTED_FALSE or emp_truth == FactTruth.VERIFIED_FALSE:
                        continue
                    elif proj_truth == ProjectedTruth.SUPPORTED_TRUE or emp_truth == FactTruth.VERIFIED_TRUE:
                        reason = f"Precondition failed on '{step_id}': expected {key} == VERIFIED_FALSE, got {emp_truth.value}/{proj_truth.value}"
                        blocker_reasons.append(reason)
                        return PlanValidationResult(
                            status=ValidationStatus.FAIL,
                            failed_step_id=step_id,
                            failed_predicate=key,
                            blocker_reasons=blocker_reasons,
                            intermediate_states=intermediate_states,
                        )
                    else:
                        step_unknown = True
                        if key not in unknown_facts:
                            unknown_facts.append(key)
                        if first_unknown_step is None:
                            first_unknown_step = step_id

            # 4. Transition: W_{t+1}
            next_state = copy.deepcopy(current_state)

            if not step_unknown:
                # Apply negative effects
                for neg in action.negative_effects:
                    neg_key = neg.fact_key
                    has_neg_verifier = _matches_verifier(
                        neg,
                        cap_entry,
                        action.parameters,
                    )

                    if has_neg_verifier:
                        neg_truth = FactTruth.UNKNOWN
                        neg_proj = ProjectedTruth.SUPPORTED_FALSE
                        neg_witness = WitnessabilityStatus.WITNESSABLE
                        neg_rationale = f"Negative effect of {step_id} (witnessable, pending runtime execution)"
                    else:
                        neg_truth = FactTruth.UNKNOWN
                        neg_proj = ProjectedTruth.UNSUPPORTED
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
                        projected_truth=neg_proj,
                        witnessability=neg_witness,
                        created_at=current_sim_time,
                        updated_at=current_sim_time,
                        provenance=Provenance(
                            source_type=SourceType.PLANNER_INFERENCE,
                            source_id=step_id,
                            confidence=1.0 if has_neg_verifier else 0.0,
                            rationale=neg_rationale,
                        ),
                        metadata={"predicted_truth": FactTruth.VERIFIED_FALSE.value},
                    )

                # Apply positive effects
                for pos in action.positive_effects:
                    pos_key = pos.fact_key
                    has_pos_verifier = _matches_verifier(
                        pos,
                        cap_entry,
                        action.parameters,
                    )

                    if has_pos_verifier:
                        pos_truth = FactTruth.UNKNOWN
                        pos_proj = ProjectedTruth.SUPPORTED_TRUE
                        pos_witness = WitnessabilityStatus.WITNESSABLE
                        pos_rationale = f"Positive effect of {step_id} (witnessable, pending runtime execution)"
                    else:
                        pos_truth = FactTruth.UNKNOWN
                        pos_proj = ProjectedTruth.UNSUPPORTED
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
                        projected_truth=pos_proj,
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
                for pos in action.positive_effects:
                    pos_key = pos.fact_key
                    if pos_key not in unknown_facts:
                        unknown_facts.append(pos_key)
                    next_state[pos_key] = WorldFact(
                        predicate=pos.predicate,
                        args=pos.args,
                        truth=FactTruth.UNKNOWN,
                        projected_truth=ProjectedTruth.UNSUPPORTED,
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
                        projected_truth=ProjectedTruth.UNSUPPORTED,
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

        # 5. Check Success Criteria on final state W_T
        criteria_satisfied: List[str] = []
        criteria_unmet: List[str] = []

        for crit in plan_ir.success_criteria:
            key = crit.condition.fact_key
            fact = current_state.get(key)
            emp_truth = fact.truth if fact is not None else FactTruth.UNKNOWN
            proj_truth = fact.projected_truth if fact is not None else ProjectedTruth.UNSUPPORTED

            is_satisfied = False
            if crit.condition.expected_truth == FactTruth.VERIFIED_TRUE:
                if proj_truth == ProjectedTruth.SUPPORTED_TRUE or emp_truth == FactTruth.VERIFIED_TRUE:
                    is_satisfied = True
            elif crit.condition.expected_truth == FactTruth.VERIFIED_FALSE:
                if proj_truth == ProjectedTruth.SUPPORTED_FALSE or emp_truth == FactTruth.VERIFIED_FALSE:
                    is_satisfied = True

            if is_satisfied:
                criteria_satisfied.append(crit.criterion_id)
            else:
                criteria_unmet.append(crit.criterion_id)
                if crit.is_mandatory:
                    if (proj_truth == ProjectedTruth.UNSUPPORTED and emp_truth == FactTruth.UNKNOWN) or fact is None:
                        if key not in unknown_facts:
                            unknown_facts.append(key)
                    else:
                        blocker_reasons.append(
                            f"Mandatory success criterion '{crit.criterion_id}' unmet: expected {key} == {crit.condition.expected_truth.value}, got {emp_truth.value}/{proj_truth.value}"
                        )

        # 6. Determine overall verdict
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
                if until_idx is not None and action_idx > until_idx:
                    continue

            key = hc.condition.fact_key
            fact = state.get(key)
            emp_truth = fact.truth if fact is not None else FactTruth.UNKNOWN
            proj_truth = fact.projected_truth if fact is not None else ProjectedTruth.UNSUPPORTED

            if hc.condition.expected_truth == FactTruth.VERIFIED_TRUE:
                if (
                    proj_truth == ProjectedTruth.SUPPORTED_FALSE
                    or emp_truth == FactTruth.VERIFIED_FALSE
                    or proj_truth == ProjectedTruth.CONFLICT
                    or emp_truth == FactTruth.CONFLICT
                ):
                    invariants_violated.append(hc.constraint_id)
                    blocker_reasons.append(
                        f"Hard constraint '{hc.constraint_id}' violated: expected {key} == VERIFIED_TRUE, got {emp_truth.value}/{proj_truth.value}"
                    )
                elif proj_truth == ProjectedTruth.SUPPORTED_TRUE or emp_truth == FactTruth.VERIFIED_TRUE:
                    continue
                else:
                    if key not in unknown_facts:
                        unknown_facts.append(key)
            elif hc.condition.expected_truth == FactTruth.VERIFIED_FALSE:
                if (
                    proj_truth == ProjectedTruth.SUPPORTED_TRUE
                    or emp_truth == FactTruth.VERIFIED_TRUE
                    or proj_truth == ProjectedTruth.CONFLICT
                    or emp_truth == FactTruth.CONFLICT
                ):
                    invariants_violated.append(hc.constraint_id)
                    blocker_reasons.append(
                        f"Hard constraint '{hc.constraint_id}' violated: expected {key} == VERIFIED_FALSE, got {emp_truth.value}/{proj_truth.value}"
                    )
                elif proj_truth == ProjectedTruth.SUPPORTED_FALSE or emp_truth == FactTruth.VERIFIED_FALSE:
                    continue
                else:
                    if key not in unknown_facts:
                        unknown_facts.append(key)


EpistemicValidator = EpistemicCausalValidator
CausalValidator = EpistemicCausalValidator
