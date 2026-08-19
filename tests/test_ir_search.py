"""Tests for IR-Native Search Operators and Epistemic Plan Optimizer."""

import pytest
from plan_mode.ir import (
    FactTruth,
    Provenance,
    SourceType,
    WorldFact,
    PredicateCondition,
    ActionIR,
    PlanIR,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.epistemic_validator import (
    CausalValidator,
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.ir_search import (
    mutate_action_parameters,
    insert_disambiguation_action,
    causal_crossover,
    EpistemicPlanSearch,
)


@pytest.fixture
def test_setup():
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="system.ping",
            description="Pings host",
            input_schema={"host": {"type": "str", "required": True}},
            positive_effects=[PredicateCondition(predicate="host_alive", args=["$host"])],
        )
    )
    reg.register(
        CapabilityEntry(
            name="system.ssh_exec",
            description="Executes command over SSH",
            input_schema={"host": {"type": "str", "required": True}, "cmd": {"type": "str", "required": True}},
            preconditions=[PredicateCondition(predicate="host_alive", args=["$host"])],
            positive_effects=[PredicateCondition(predicate="cmd_executed", args=["$host", "$cmd"])],
        )
    )

    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    plan_a = PlanIR(
        plan_id="plan_a",
        goal_description="Run remote script",
        initial_state=[WorldFact(predicate="host_alive", args=["srv1.internal"], truth=FactTruth.UNKNOWN, provenance=prov)],
        actions=[
            ActionIR(
                action_id="act_01",
                capability_name="system.ssh_exec",
                parameters={"host": "srv1.internal", "cmd": "deploy.sh"},
                preconditions=[PredicateCondition(predicate="host_alive", args=["srv1.internal"])],
                positive_effects=[PredicateCondition(predicate="cmd_executed", args=["srv1.internal", "deploy.sh"])],
                provenance=prov,
            )
        ],
    )
    return reg, plan_a


def test_insert_disambiguation_action(test_setup):
    reg, plan = test_setup

    # Effect-creating search operations are closed-world and require registry grounding.
    disambiguated = insert_disambiguation_action(
        plan_ir=plan,
        target_action_index=0,
        probe_capability_name="system.ping",
        parameters={"host": "srv1.internal"},
        positive_effects=[PredicateCondition(predicate="host_alive", args=["srv1.internal"])],
        registry=reg,
    )

    assert len(disambiguated.actions) == 2
    assert disambiguated.actions[0].capability_name == "system.ping"
    assert disambiguated.actions[1].capability_name == "system.ssh_exec"
    assert disambiguated.actions[0].provenance.source_type == SourceType.PLANNER_INFERENCE
    # The effect comes from the registered contract, not from caller invention.
    assert disambiguated.actions[0].positive_effects[0].predicate == "host_alive"


def test_mutate_action_parameters(test_setup):
    reg, plan = test_setup
    mutated = mutate_action_parameters(plan, action_index=0, parameter_updates={"host": "srv2.internal"})
    assert mutated.actions[0].parameters["host"] == "srv2.internal"
    assert plan.actions[0].parameters["host"] == "srv1.internal"


def test_causal_crossover(test_setup):
    reg, plan_a = test_setup
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)

    plan_b = PlanIR(
        plan_id="plan_b",
        goal_description="Alternative plan",
        initial_state=[],
        actions=[
            ActionIR(
                action_id="b_01",
                capability_name="system.ping",
                parameters={"host": "srv1.internal"},
                positive_effects=[PredicateCondition(predicate="host_alive", args=["srv1.internal"])],
                provenance=prov,
            )
        ],
    )

    child = causal_crossover(parent_a=plan_b, parent_b=plan_a, split_index_a=1, registry=reg)
    assert len(child.actions) == 2
    assert child.actions[0].capability_name == "system.ping"
    assert child.actions[1].capability_name == "system.ssh_exec"


def test_epistemic_plan_search_optimizer(test_setup):
    reg, plan = test_setup
    searcher = EpistemicPlanSearch(registry=reg)

    initial_grounded = [
        WorldFact(predicate="host_ready", args=["srv1.internal"], truth=FactTruth.VERIFIED_TRUE, provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE))
    ]

    optimized = searcher.search_best_plan(
        seed_plan=plan,
        max_iterations=10,
    )
    assert optimized is not None
