"""IR-Native Search Operators, Causal Crossover, Token Cost Tracking, and Epistemic Plan Optimizer (Phase 5)."""

from __future__ import annotations

import copy
import json
import random
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import (
    CausalValidator,
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    ProjectedTruth,
    Provenance,
    SourceType,
    WorldFact,
)
from plan_mode.registry import CapabilityRegistry, CapabilityNotFoundError


class TokenCostTracker(BaseModel):
    """Aggregates LLM token usage, latency, and estimated dollar costs."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    calls_count: int = 0
    calls_by_provider: Dict[str, int] = Field(default_factory=dict)

    MODEL_PRICING: Dict[str, tuple[float, float]] = {
        # model: (prompt_cost_per_M, completion_cost_per_M)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-5-haiku": (0.80, 4.00),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-1.5-pro": (1.25, 5.00),
        "deepseek-chat": (0.14, 0.28),
        "deepseek-reasoner": (0.55, 2.19),
    }

    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Record a single model call invocation telemetry."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency_ms += latency_ms
        self.calls_count += 1
        self.calls_by_provider[provider] = self.calls_by_provider.get(provider, 0) + 1

        if cost_usd is not None:
            self.total_cost_usd += cost_usd
        else:
            pricing = self.MODEL_PRICING.get(model, (1.00, 3.00))
            cost = (prompt_tokens / 1_000_000.0 * pricing[0]) + (completion_tokens / 1_000_000.0 * pricing[1])
            self.total_cost_usd += cost

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregate telemetry summary."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "calls_count": self.calls_count,
            "calls_by_provider": self.calls_by_provider,
        }


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


def mutate_reorder_actions(
    plan_ir: PlanIR,
    index_1: int,
    index_2: int,
) -> PlanIR:
    """Return a new PlanIR with two actions swapped in execution order."""
    if (
        index_1 < 0
        or index_1 >= len(plan_ir.actions)
        or index_2 < 0
        or index_2 >= len(plan_ir.actions)
        or index_1 == index_2
    ):
        return plan_ir.model_copy(deep=True)

    new_plan = plan_ir.model_copy(deep=True)
    actions = list(new_plan.actions)
    actions[index_1], actions[index_2] = actions[index_2], actions[index_1]
    new_plan.actions = actions
    return new_plan


def mutate_delete_action(
    plan_ir: PlanIR,
    action_index: int,
) -> PlanIR:
    """Return a new PlanIR with the specified action deleted."""
    if action_index < 0 or action_index >= len(plan_ir.actions):
        return plan_ir.model_copy(deep=True)

    new_plan = plan_ir.model_copy(deep=True)
    actions = list(new_plan.actions)
    actions.pop(action_index)
    new_plan.actions = actions
    return new_plan


def mutate_insert_action(
    plan_ir: PlanIR,
    target_index: int,
    new_action: ActionIR,
) -> PlanIR:
    """Insert a new action at the specified index."""
    new_plan = plan_ir.model_copy(deep=True)
    idx = max(0, min(target_index, len(new_plan.actions)))
    new_plan.actions.insert(idx, new_action.model_copy(deep=True))
    return new_plan


def mutate_replace_action(
    plan_ir: PlanIR,
    action_index: int,
    new_capability_name: str,
    parameters: Dict[str, Any],
    registry: CapabilityRegistry,
) -> PlanIR:
    """Replace an action with an alternative capability from the closed-world registry."""
    if action_index < 0 or action_index >= len(plan_ir.actions):
        return plan_ir.model_copy(deep=True)

    # Capability Closed-World Verification: must be registered in CapabilityRegistry
    try:
        cap_entry = registry.get(new_capability_name)
    except CapabilityNotFoundError:
        # Unregistered capability is strictly rejected; return unmutated plan
        return plan_ir.model_copy(deep=True)

    new_plan = plan_ir.model_copy(deep=True)
    old_action = new_plan.actions[action_index]

    # Instantiate declared positive and negative effects
    instantiated_pos: List[PredicateCondition] = []
    for eff in cap_entry.positive_effects:
        inst_args = []
        for arg in eff.args:
            if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
                inst_args.append(parameters.get(arg[1:-1], arg))
            elif isinstance(arg, str) and arg.startswith("$"):
                inst_args.append(parameters.get(arg[1:], arg))
            else:
                inst_args.append(arg)
        instantiated_pos.append(
            PredicateCondition(
                predicate=eff.predicate,
                args=inst_args,
                expected_truth=eff.expected_truth,
            )
        )

    instantiated_neg: List[PredicateCondition] = []
    for neg in cap_entry.negative_effects:
        inst_args = []
        for arg in neg.args:
            if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
                inst_args.append(parameters.get(arg[1:-1], arg))
            elif isinstance(arg, str) and arg.startswith("$"):
                inst_args.append(parameters.get(arg[1:], arg))
            else:
                inst_args.append(arg)
        instantiated_neg.append(
            PredicateCondition(
                predicate=neg.predicate,
                args=inst_args,
                expected_truth=neg.expected_truth,
            )
        )

    replaced_action = ActionIR(
        action_id=f"{new_capability_name.replace('.', '_')}_{action_index}",
        capability_name=new_capability_name,
        parameters=parameters,
        preconditions=list(cap_entry.preconditions),
        positive_effects=instantiated_pos,
        negative_effects=instantiated_neg,
        is_idempotent=cap_entry.is_idempotent,
        provenance=Provenance(
            source_type=SourceType.PLANNER_INFERENCE,
            confidence=1.0,
            rationale=f"Replaced {old_action.capability_name} with {new_capability_name}",
        ),
    )
    new_plan.actions[action_index] = replaced_action
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
    """Deterministic and evolutionary search optimizer operating directly on PlanIR space."""

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        validator: Optional[CausalValidator] = None,
        seed: Optional[int] = None,
        cost_tracker: Optional[TokenCostTracker] = None,
    ):
        self.registry = registry or CapabilityRegistry()
        self.validator = validator or CausalValidator()
        self.seed = seed
        self.cost_tracker = cost_tracker or TokenCostTracker()
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

    def search_best_plan(
        self,
        seed_plan: PlanIR,
        max_iterations: int = 10,
        beam_width: int = 5,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ) -> PlanIR:
        """Search the PlanIR space to resolve UNKNOWN and FAIL conditions with deterministic revalidation."""
        beam = [seed_plan.model_copy(deep=True)]
        best_plan = seed_plan.model_copy(deep=True)
        best_score = -100.0

        for iteration in range(max_iterations):
            next_beam: List[PlanIR] = []

            for candidate in beam:
                # Deterministic Revalidation of every candidate
                val_res = self.validator.validate_plan(
                    candidate,
                    registry=self.registry,
                    observed_world_state=observed_world_state,
                )

                score = self._score_validation_result(val_res)
                if score > best_score:
                    best_score = score
                    best_plan = candidate.model_copy(deep=True)

                if val_res.status == ValidationStatus.PASS:
                    return candidate

                # Flaw-directed mutations for UNKNOWN facts
                if val_res.status == ValidationStatus.UNKNOWN and val_res.unknown_facts:
                    target_unknown = val_res.unknown_facts[0]
                    # Look up capability in registry that produces the missing fact
                    for cap_name, cap in sorted(self.registry.capabilities.items()):
                        for eff in cap.positive_effects:
                            if eff.predicate in target_unknown:
                                probe_params: Dict[str, Any] = {}
                                for p_name, p_spec in cap.input_schema.items():
                                    probe_params[p_name] = "default_val"

                                mutated = insert_disambiguation_action(
                                    plan_ir=candidate,
                                    target_action_index=0,
                                    probe_capability_name=cap_name,
                                    parameters=probe_params,
                                    positive_effects=[
                                        PredicateCondition(predicate=eff.predicate, args=[])
                                    ],
                                )
                                next_beam.append(mutated)
                                break

                # Exploratory closed-world parameter & reordering mutations
                if candidate.actions:
                    # Action reordering mutation
                    if len(candidate.actions) >= 2:
                        idx1 = self._rng.randint(0, len(candidate.actions) - 1)
                        idx2 = self._rng.randint(0, len(candidate.actions) - 1)
                        if idx1 != idx2:
                            m_reorder = mutate_reorder_actions(candidate, idx1, idx2)
                            next_beam.append(m_reorder)

                next_beam.append(candidate)

            if next_beam:
                # Deterministically sort beam candidates by validation score
                scored_candidates = []
                for c in next_beam:
                    v_res = self.validator.validate_plan(c, registry=self.registry, observed_world_state=observed_world_state)
                    scored_candidates.append((self._score_validation_result(v_res), c))

                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                beam = [item[1] for item in scored_candidates[:beam_width]]

        return best_plan

    def _score_validation_result(self, val_res) -> float:
        """Deterministic scoring: PASS = 100, UNKNOWN = 50 - 5 * len(unknowns), FAIL = -10."""
        if val_res.status == ValidationStatus.PASS:
            return 100.0 + len(val_res.criteria_satisfied) * 10.0
        elif val_res.status == ValidationStatus.UNKNOWN:
            return 50.0 - (len(val_res.unknown_facts) * 5.0) + (len(val_res.criteria_satisfied) * 2.0)
        else:
            return -10.0 - len(val_res.blocker_reasons) * 2.0
