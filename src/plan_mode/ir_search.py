"""Closed-world IR search with deterministic certification and advisory judges."""

from __future__ import annotations

import asyncio
import concurrent.futures
import random
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import (
    CausalValidator,
    PlanValidationResult,
    ValidationStatus,
)
from plan_mode.ir import (
    ActionIR,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    WorldFact,
)
from plan_mode.judges import JudgeAdapter, JudgeVerdict
from plan_mode.registry import CapabilityNotFoundError, CapabilityRegistry


class TokenCostTracker(BaseModel):
    """Aggregate judge token usage, latency, and cost."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    calls_count: int = 0
    calls_by_provider: Dict[str, int] = Field(default_factory=dict)

    MODEL_PRICING: Dict[str, tuple[float, float]] = {
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
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_completion_tokens += int(completion_tokens)
        self.total_latency_ms += float(latency_ms)
        self.calls_count += 1
        self.calls_by_provider[provider] = self.calls_by_provider.get(provider, 0) + 1

        if cost_usd is not None:
            self.total_cost_usd += float(cost_usd)
        else:
            prompt_price, completion_price = self.MODEL_PRICING.get(model, (1.00, 3.00))
            self.total_cost_usd += (
                prompt_tokens / 1_000_000.0 * prompt_price
                + completion_tokens / 1_000_000.0 * completion_price
            )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "calls_count": self.calls_count,
            "calls_by_provider": dict(self.calls_by_provider),
        }


class SearchResult(BaseModel):
    """Search output with explicit deterministic certification status."""

    plan: PlanIR
    validation_result: PlanValidationResult
    validation_status: ValidationStatus
    is_certified: bool
    iterations_run: int
    cost_summary: Dict[str, Any] = Field(default_factory=dict)
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)


def _instantiate_condition(
    condition: PredicateCondition,
    parameters: Dict[str, Any],
) -> PredicateCondition:
    args: List[Any] = []
    for arg in condition.args:
        if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
            args.append(parameters.get(arg[1:-1], arg))
        elif isinstance(arg, str) and arg.startswith("$"):
            args.append(parameters.get(arg[1:], arg))
        elif isinstance(arg, str) and arg in parameters:
            args.append(parameters[arg])
        else:
            args.append(arg)
    return PredicateCondition(
        predicate=condition.predicate,
        args=args,
        expected_truth=condition.expected_truth,
        active_until_action_id=condition.active_until_action_id,
    )


def _registered_action(
    registry: CapabilityRegistry,
    capability_name: str,
    parameters: Dict[str, Any],
    action_id: str,
    *,
    rationale: str,
) -> Optional[ActionIR]:
    """Instantiate an action solely from a registered capability contract."""
    try:
        capability = registry.get(capability_name)
    except CapabilityNotFoundError:
        return None

    action = ActionIR(
        action_id=action_id,
        capability_name=capability_name,
        parameters=dict(parameters),
        preconditions=[_instantiate_condition(c, parameters) for c in capability.preconditions],
        positive_effects=[_instantiate_condition(c, parameters) for c in capability.positive_effects],
        negative_effects=[_instantiate_condition(c, parameters) for c in capability.negative_effects],
        is_idempotent=capability.is_idempotent,
        provenance=Provenance(
            source_type=SourceType.PLANNER_INFERENCE,
            confidence=1.0,
            rationale=rationale,
        ),
    )
    try:
        registry.validate_action(action)
    except Exception:
        return None
    return action


def mutate_action_parameters(
    plan_ir: PlanIR,
    action_index: int,
    parameter_updates: Dict[str, Any],
    registry: Optional[CapabilityRegistry] = None,
) -> PlanIR:
    """Mutate parameters; with a registry, re-derive the full action contract."""
    if action_index < 0 or action_index >= len(plan_ir.actions):
        return plan_ir.model_copy(deep=True)

    target = plan_ir.actions[action_index]
    parameters = dict(target.parameters)
    parameters.update(parameter_updates)

    if registry is not None:
        rebuilt = _registered_action(
            registry,
            target.capability_name,
            parameters,
            target.action_id,
            rationale="Mutated parameters by registry-grounded IR search",
        )
        if rebuilt is None:
            return plan_ir.model_copy(deep=True)
        new_plan = plan_ir.model_copy(deep=True)
        new_plan.actions[action_index] = rebuilt
        return new_plan

    # Compatibility helper only.  This path is not used as a certification
    # boundary and cannot invent a new capability or effect set.
    new_plan = plan_ir.model_copy(deep=True)
    new_plan.actions[action_index] = target.model_copy(
        update={
            "parameters": parameters,
            "provenance": Provenance(
                source_type=SourceType.PLANNER_INFERENCE,
                rationale="Mutated parameters by IR-native search",
            ),
        }
    )
    return new_plan


def mutate_reorder_actions(plan_ir: PlanIR, index_1: int, index_2: int) -> PlanIR:
    if (
        index_1 < 0
        or index_1 >= len(plan_ir.actions)
        or index_2 < 0
        or index_2 >= len(plan_ir.actions)
        or index_1 == index_2
    ):
        return plan_ir.model_copy(deep=True)
    new_plan = plan_ir.model_copy(deep=True)
    new_plan.actions[index_1], new_plan.actions[index_2] = (
        new_plan.actions[index_2],
        new_plan.actions[index_1],
    )
    return new_plan


def mutate_delete_action(plan_ir: PlanIR, action_index: int) -> PlanIR:
    if action_index < 0 or action_index >= len(plan_ir.actions):
        return plan_ir.model_copy(deep=True)
    new_plan = plan_ir.model_copy(deep=True)
    new_plan.actions.pop(action_index)
    return new_plan


def mutate_insert_action(
    plan_ir: PlanIR,
    target_index: int,
    new_action: ActionIR,
    registry: Optional[CapabilityRegistry] = None,
) -> PlanIR:
    """Insert only a registry-valid action; missing registry fails closed."""
    if registry is None:
        return plan_ir.model_copy(deep=True)
    try:
        registry.get(new_action.capability_name)
        registry.validate_action(new_action)
    except Exception:
        return plan_ir.model_copy(deep=True)
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
    if action_index < 0 or action_index >= len(plan_ir.actions):
        return plan_ir.model_copy(deep=True)
    old_action = plan_ir.actions[action_index]
    replacement = _registered_action(
        registry,
        new_capability_name,
        parameters,
        f"{new_capability_name.replace('.', '_')}_{action_index}",
        rationale=f"Replaced {old_action.capability_name} with {new_capability_name}",
    )
    if replacement is None:
        return plan_ir.model_copy(deep=True)
    new_plan = plan_ir.model_copy(deep=True)
    new_plan.actions[action_index] = replacement
    return new_plan


def insert_disambiguation_action(
    plan_ir: PlanIR,
    target_action_index: int,
    probe_capability_name: str,
    parameters: Dict[str, Any],
    positive_effects: Optional[List[PredicateCondition]] = None,
    registry: Optional[CapabilityRegistry] = None,
) -> PlanIR:
    """Insert a probe derived exactly from the registry; caller effects are ignored."""
    if registry is None:
        return plan_ir.model_copy(deep=True)
    idx = max(0, min(target_action_index, len(plan_ir.actions)))
    probe = _registered_action(
        registry,
        probe_capability_name,
        parameters,
        f"probe_{probe_capability_name.replace('.', '_')}_{idx}",
        rationale="Inserted registry-grounded disambiguation probe",
    )
    if probe is None:
        return plan_ir.model_copy(deep=True)
    new_plan = plan_ir.model_copy(deep=True)
    new_plan.actions.insert(idx, probe)
    return new_plan


def causal_crossover(
    parent_a: PlanIR,
    parent_b: PlanIR,
    split_index_a: int = 1,
    registry: Optional[CapabilityRegistry] = None,
) -> PlanIR:
    """Splice parents only when every resulting action is registry-valid."""
    if registry is None:
        return parent_a.model_copy(deep=True)

    actions_prefix = parent_a.actions[:split_index_a]
    actions_suffix = (
        parent_b.actions[split_index_a:]
        if split_index_a < len(parent_b.actions)
        else parent_b.actions
    )
    combined: List[ActionIR] = []
    seen_ids = set()
    for source_action in actions_prefix + actions_suffix:
        cloned = source_action.model_copy(deep=True)
        if cloned.action_id in seen_ids:
            cloned.action_id = f"{cloned.action_id}_cross"
        seen_ids.add(cloned.action_id)
        try:
            registry.get(cloned.capability_name)
            registry.validate_action(cloned)
        except Exception:
            return parent_a.model_copy(deep=True)
        combined.append(cloned)

    new_plan = parent_a.model_copy(deep=True)
    new_plan.actions = combined
    return new_plan


class EpistemicPlanSearch:
    """Search PlanIR while keeping deterministic validation authoritative.

    An optional judge may rank candidates and propose mutations. Judge output is
    never empirical evidence and can never set ``is_certified``.  Every judge
    mutation is translated through closed-world operators and then revalidated.
    """

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        validator: Optional[CausalValidator] = None,
        seed: Optional[int] = None,
        cost_tracker: Optional[TokenCostTracker] = None,
        judge: Optional[JudgeAdapter] = None,
    ):
        self.registry = registry or CapabilityRegistry()
        self.validator = validator or CausalValidator()
        self.seed = seed
        self.cost_tracker = cost_tracker or TokenCostTracker()
        self.judge = judge
        self._rng = random.Random(seed)
        self._judge_cache: Dict[str, JudgeVerdict] = {}
        self._trajectory: List[Dict[str, Any]] = []

    def _run_judge(
        self,
        plan: PlanIR,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]],
    ) -> Optional[JudgeVerdict]:
        if self.judge is None:
            return None
        plan_hash = plan.compute_hash()
        if plan_hash in self._judge_cache:
            return self._judge_cache[plan_hash]

        async def call() -> JudgeVerdict:
            assert self.judge is not None
            return await self.judge.evaluate(
                plan,
                goal_description=plan.goal_description,
                registry=self.registry,
                observed_world_state=observed_world_state,
            )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    verdict = pool.submit(lambda: asyncio.run(call())).result()
            else:
                verdict = loop.run_until_complete(call())
        except RuntimeError:
            verdict = asyncio.run(call())

        self._judge_cache[plan_hash] = verdict
        usage = verdict.token_usage
        self.cost_tracker.record_usage(
            provider=verdict.provider,
            model=verdict.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=verdict.latency_ms,
            cost_usd=float(usage.get("cost_usd", 0.0) or 0.0),
        )
        return verdict

    def _judge_adjustment(self, verdict: Optional[JudgeVerdict]) -> float:
        if verdict is None:
            return 0.0
        # Advisory only: bounded so it cannot turn FAIL/UNKNOWN into a
        # deterministic certification.
        adjustment = (verdict.feasibility_0_100 - 50.0) / 10.0
        if verdict.verdict == "FAIL":
            adjustment -= 5.0
        elif verdict.verdict == "UNKNOWN":
            adjustment -= 2.5
        return max(-10.0, min(10.0, adjustment))

    def _judge_mutations(self, plan: PlanIR, verdict: Optional[JudgeVerdict]) -> List[PlanIR]:
        if verdict is None:
            return []
        mutations: List[PlanIR] = []
        for suggestion in verdict.suggested_mutations:
            if not isinstance(suggestion, dict):
                continue
            op = str(suggestion.get("op") or suggestion.get("type") or "").lower()
            try:
                if op in {"replace_action", "replace"}:
                    mutations.append(
                        mutate_replace_action(
                            plan,
                            int(suggestion["action_index"]),
                            str(suggestion["capability_name"]),
                            dict(suggestion.get("parameters", {})),
                            self.registry,
                        )
                    )
                elif op in {"insert_action", "insert"}:
                    index = int(suggestion.get("target_index", 0))
                    cap_name = str(suggestion["capability_name"])
                    parameters = dict(suggestion.get("parameters", {}))
                    action = _registered_action(
                        self.registry,
                        cap_name,
                        parameters,
                        f"judge_insert_{cap_name.replace('.', '_')}_{index}",
                        rationale="Judge-suggested action instantiated from registry",
                    )
                    if action is not None:
                        mutations.append(
                            mutate_insert_action(plan, index, action, registry=self.registry)
                        )
                elif op in {"update_parameters", "parameters"}:
                    mutations.append(
                        mutate_action_parameters(
                            plan,
                            int(suggestion["action_index"]),
                            dict(suggestion.get("parameters", {})),
                            registry=self.registry,
                        )
                    )
                elif op in {"reorder_actions", "reorder"}:
                    mutations.append(
                        mutate_reorder_actions(
                            plan,
                            int(suggestion["index_1"]),
                            int(suggestion["index_2"]),
                        )
                    )
                elif op in {"delete_action", "delete"}:
                    mutations.append(mutate_delete_action(plan, int(suggestion["action_index"])))
            except (KeyError, TypeError, ValueError):
                continue
        return [m for m in mutations if m.compute_hash() != plan.compute_hash()]

    def search_best_plan(
        self,
        seed_plan: PlanIR,
        max_iterations: int = 10,
        beam_width: int = 5,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ) -> SearchResult:
        self._trajectory = []
        self._judge_cache = {}
        beam = [seed_plan.model_copy(deep=True)]
        best_plan = seed_plan.model_copy(deep=True)
        best_val_res = self.validator.validate_plan(
            best_plan,
            registry=self.registry,
            observed_world_state=observed_world_state,
        )
        best_judge = self._run_judge(best_plan, observed_world_state)
        best_score = self._score_validation_result(best_val_res) + self._judge_adjustment(best_judge)
        self._record_trajectory(0, best_plan, best_val_res, best_judge)

        if best_val_res.status == ValidationStatus.PASS:
            return self._result(best_plan, best_val_res, 0)

        for iteration in range(max_iterations):
            next_beam: List[PlanIR] = []

            for candidate in beam:
                val_res = self.validator.validate_plan(
                    candidate,
                    registry=self.registry,
                    observed_world_state=observed_world_state,
                )
                judge_verdict = self._run_judge(candidate, observed_world_state)
                self._record_trajectory(iteration + 1, candidate, val_res, judge_verdict)
                score = self._score_validation_result(val_res) + self._judge_adjustment(judge_verdict)
                if score > best_score:
                    best_score = score
                    best_plan = candidate.model_copy(deep=True)
                    best_val_res = val_res

                if val_res.status == ValidationStatus.PASS:
                    return self._result(candidate, val_res, iteration + 1)

                # Judge advice is translated through closed-world operators.
                next_beam.extend(self._judge_mutations(candidate, judge_verdict))

                # Deterministic flaw-directed probe insertion.  The action and
                # effects are instantiated exclusively from the registry.
                if val_res.status == ValidationStatus.UNKNOWN and val_res.unknown_facts:
                    target_unknown = val_res.unknown_facts[0]
                    match = re.match(r"^([\w:-]+)(?:\((.*?)\))?$", target_unknown)
                    target_predicate = match.group(1) if match else target_unknown
                    raw_args = match.group(2) if match else None
                    unknown_args = (
                        [part.strip().strip("'\"") for part in raw_args.split(",")]
                        if raw_args
                        else []
                    )

                    for cap_name, capability in sorted(self.registry.capabilities.items()):
                        matching_effect = next(
                            (
                                effect
                                for effect in capability.positive_effects
                                if effect.predicate == target_predicate
                            ),
                            None,
                        )
                        if matching_effect is None:
                            continue
                        probe_params: Dict[str, Any] = {}
                        for index, param_name in enumerate(capability.input_schema):
                            probe_params[param_name] = (
                                unknown_args[index] if index < len(unknown_args) else "default_val"
                            )
                        mutated = insert_disambiguation_action(
                            candidate,
                            0,
                            cap_name,
                            probe_params,
                            registry=self.registry,
                        )
                        if mutated.compute_hash() != candidate.compute_hash():
                            next_beam.append(mutated)
                        break

                if len(candidate.actions) >= 2:
                    idx1 = self._rng.randint(0, len(candidate.actions) - 1)
                    idx2 = self._rng.randint(0, len(candidate.actions) - 1)
                    if idx1 != idx2:
                        next_beam.append(mutate_reorder_actions(candidate, idx1, idx2))
                next_beam.append(candidate)

            if not next_beam:
                break

            # De-duplicate before expensive validation/judging.
            unique: Dict[str, PlanIR] = {}
            for candidate in next_beam:
                unique[candidate.compute_hash()] = candidate

            scored_candidates = []
            for candidate in unique.values():
                validation = self.validator.validate_plan(
                    candidate,
                    registry=self.registry,
                    observed_world_state=observed_world_state,
                )
                judge_verdict = self._run_judge(candidate, observed_world_state)
                score = self._score_validation_result(validation) + self._judge_adjustment(judge_verdict)
                scored_candidates.append((score, candidate, validation))

            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            beam = [item[1] for item in scored_candidates[:beam_width]]
            if scored_candidates and scored_candidates[0][0] > best_score:
                best_score, best_plan, best_val_res = scored_candidates[0]

        return self._result(best_plan, best_val_res, max_iterations)

    def _record_trajectory(
        self,
        iteration: int,
        plan: PlanIR,
        validation: PlanValidationResult,
        judge: Optional[JudgeVerdict],
    ) -> None:
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "plan_hash": plan.compute_hash(),
            "validation_status": validation.status.value,
        }
        if judge is not None:
            entry.update(
                {
                    "judge_provider": judge.provider,
                    "judge_verdict": judge.verdict,
                    "judge_feasibility_0_100": judge.feasibility_0_100,
                }
            )
        self._trajectory.append(entry)

    def _result(
        self,
        plan: PlanIR,
        validation: PlanValidationResult,
        iterations_run: int,
    ) -> SearchResult:
        return SearchResult(
            plan=plan.model_copy(deep=True),
            validation_result=validation,
            validation_status=validation.status,
            is_certified=validation.status == ValidationStatus.PASS,
            iterations_run=iterations_run,
            cost_summary=self.cost_tracker.get_summary(),
            trajectory=list(self._trajectory),
        )

    def _score_validation_result(self, val_res: PlanValidationResult) -> float:
        if val_res.status == ValidationStatus.PASS:
            return 100.0 + len(val_res.criteria_satisfied) * 10.0
        if val_res.status == ValidationStatus.UNKNOWN:
            return (
                50.0
                - len(val_res.unknown_facts) * 5.0
                + len(val_res.criteria_satisfied) * 2.0
            )
        return -10.0 - len(val_res.blocker_reasons) * 2.0
