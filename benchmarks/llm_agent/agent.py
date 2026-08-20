# Autonomous Planning Agent for Multi-Turn Benchmarking

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from benchmarks.llm_agent.client import BaseLLMClient, LLMResponse, SimulatedLLMClient
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
from plan_mode.ir_search import TokenCostTracker
from plan_mode.registry import CapabilityRegistry


class AgentPlanOutput(BaseModel):
    plan: PlanIR
    raw_response: LLMResponse
    token_cost_usd: float = 0.0
    latency_ms: float = 0.0
    generated_actions_count: int = 0


SYSTEM_PROMPT = """You are an autonomous AI software and DevOps agent.
Given a user instruction and the current world facts, generate a structured sequence of actions to achieve the goal while respecting all safety constraints and invariants.
Output your reasoning and the exact capabilities you wish to execute."""


class AutonomousPlanningAgent:
    """Multi-turn autonomous planning agent that dynamically translates user tasks into executable PlanIR."""

    def __init__(self, client: Optional[BaseLLMClient] = None, cost_tracker: Optional[TokenCostTracker] = None):
        self.client = client or SimulatedLLMClient()
        self.cost_tracker = cost_tracker or TokenCostTracker()

    def generate_plan(
        self,
        task_id: str,
        instruction: str,
        initial_facts: List[WorldFact],
        registry: CapabilityRegistry,
        hard_constraints: Optional[List[HardConstraint]] = None,
        success_criteria: Optional[List[SuccessCriterion]] = None,
    ) -> AgentPlanOutput:
        """Query LLM to dynamically generate PlanIR actions and measure real token usage."""
        start = time.perf_counter()

        user_prompt = f"Task: {task_id}\nInstruction: {instruction}\nInitial State Facts: {[f.predicate for f in initial_facts]}"

        # 1. Dispatch to LLM client
        response = self.client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=[{"name": cap.name, "description": cap.description} for cap in registry._entries.values()] if hasattr(registry, "_entries") else [],
        )

        # 2. Record real token usage and cost
        cost_before = self.cost_tracker.total_cost_usd
        self.cost_tracker.record_usage(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
        cost = self.cost_tracker.total_cost_usd - cost_before

        # 3. Construct dynamic PlanIR actions from the LLM's tool calls
        actions: List[ActionIR] = []
        for i, call in enumerate(response.tool_calls):
            cap_name = call.get("name", "bash_command")
            params = call.get("parameters", {})

            # Lookup capability from registry to retrieve declared effects
            cap_entry = registry.get(cap_name) if hasattr(registry, "get") else None
            pos_effects = cap_entry.positive_effects if cap_entry else []
            neg_effects = cap_entry.negative_effects if cap_entry else []
            preconds = cap_entry.preconditions if cap_entry else []

            act = ActionIR(
                action_id=f"act_{task_id}_{i+1}",
                capability_name=cap_name,
                parameters=params,
                preconditions=preconds,
                positive_effects=pos_effects,
                negative_effects=neg_effects,
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            )
            actions.append(act)

        plan = PlanIR(
            plan_id=f"plan_{task_id}",
            goal_description=instruction,
            initial_state=initial_facts,
            actions=actions,
            hard_constraints=hard_constraints or [],
            success_criteria=success_criteria or [],
        )

        dur = (time.perf_counter() - start) * 1000.0

        return AgentPlanOutput(
            plan=plan,
            raw_response=response,
            token_cost_usd=cost,
            latency_ms=dur,
            generated_actions_count=len(actions),
        )
