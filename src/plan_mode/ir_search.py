"""IR-Native Search Operators, Causal Crossover, and Epistemic Plan Optimizer."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import (
    CausalValidator,
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.ir import ActionIR, PlanIR, PredicateCondition, Provenance, SourceType, WorldFact
from plan_mode.registry import CapabilityRegistry


def mutate_action_parameters(
    plan_ir: PlanIR,
    action_index: int,
    parameter_updates: Dict[str, Any],
) -> PlanIR:
    """Return a new PlanIR with mutated parameters at the given action index."""
    if action_index < 0 or action_index >= len(plan_ir.actions):
        return plan_ir.model_copy(deep=True)

    new_plan = plan_ir.model_copy(deep=True)
    target_action = new_plan.actions[action_index]
    new_params = dict(target_action.parameters)
    new_params.update(parameter_updates)
    
    mutated_action = target_action.model_copy(
        update={
            "parameters": new_params,
            "provenance": Provenance(
                source_type=SourceType.PLANNER_INFERENCE,
                rationale="Mutated parameters by IR-native search",
            ),
        }
    )
    new_plan.actions[action_index] = mutated_action
    return new_plan


def insert_disambiguation_action(
    plan_ir: PlanIR,
    target_action_index: int,
    probe_capability_name: str,
    parameters: Dict[str, Any],
    positive_effects: List[PredicateCondition],
) -> PlanIR:
    """Insert an epistemic grounding or probing action ahead of an ungrounded step."""
    new_plan = plan_ir.model_copy(deep=True)
    idx = max(0, min(target_action_index, len(new_plan.actions)))

    probe_action = ActionIR(
        action_id=f"probe_{probe_capability_name}_{idx}",
        capability_name=probe_capability_name,
        parameters=parameters,
        preconditions=[],
        positive_effects=positive_effects,
        is_idempotent=True,
        provenance=Provenance(
            source_type=SourceType.PLANNER_INFERENCE,
            confidence=1.0,
            rationale="Inserted disambiguation probe",
        ),
    )
    new_plan.actions.insert(idx, probe_action)
    return new_plan


def causal_crossover(
    parent_a: PlanIR,
    parent_b: PlanIR,
    split_index_a: int = 1,
) -> PlanIR:
    """Splice action subgraphs of two parent plans preserving provenance and schema."""
    new_plan = parent_a.model_copy(deep=True)
    actions_prefix = parent_a.actions[:split_index_a]
    actions_suffix = parent_b.actions[split_index_a:] if split_index_a < len(parent_b.actions) else parent_b.actions

    combined_actions: List[ActionIR] = []
    seen_ids = set()
    for act in actions_prefix + actions_suffix:
        cloned = act.model_copy(deep=True)
        if cloned.action_id in seen_ids:
            cloned.action_id = f"{cloned.action_id}_cross"
        seen_ids.add(cloned.action_id)
        combined_actions.append(cloned)

    new_plan.actions = combined_actions
    return new_plan


class EpistemicPlanSearch:
    """Beam-search epistemic optimizer operating directly on PlanIR space."""

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        validator: Optional[CausalValidator] = None,
    ):
        self.registry = registry or CapabilityRegistry()
        self.validator = validator or CausalValidator()

    def search_best_plan(
        self,
        seed_plan: PlanIR,
        max_iterations: int = 10,
        beam_width: int = 5,
    ) -> PlanIR:
        """Search the space of plans to resolve UNKNOWN and FAIL conditions."""
        beam = [seed_plan.model_copy(deep=True)]

        for _ in range(max_iterations):
            next_beam: List[PlanIR] = []
            
            for candidate in beam:
                val_res = self.validator.validate_plan(candidate, registry=self.registry)
                if val_res.status == ValidationStatus.PASS:
                    return candidate

                if val_res.status == ValidationStatus.UNKNOWN and val_res.unknown_facts:
                    # Look up capability in registry that produces the missing fact
                    target_unknown = val_res.unknown_facts[0]
                    resolved = False
                    for cap_name, cap in self.registry.capabilities.items():
                        for eff in cap.positive_effects:
                            if eff.predicate in target_unknown:
                                mutated = insert_disambiguation_action(
                                    plan_ir=candidate,
                                    target_action_index=0,
                                    probe_capability_name=cap_name,
                                    parameters={},
                                    positive_effects=[
                                        PredicateCondition(predicate=eff.predicate, args=[])
                                    ],
                                )
                                next_beam.append(mutated)
                                resolved = True
                                break
                        if resolved:
                            break

                next_beam.append(candidate)

            if next_beam:
                beam = next_beam[:beam_width]

        return beam[0]
