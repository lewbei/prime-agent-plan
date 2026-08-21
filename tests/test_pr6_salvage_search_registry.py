"""Search/cache/registry regressions salvaged from superseded PR #6."""
from __future__ import annotations

from plan_mode.ast_search import ASTSearchEngine
from plan_mode.causal_validator import ActionSchema, PlanAST
from plan_mode.ir import ActionIR, FactTruth, PlanIR, Provenance, SourceType, WorldFact
from plan_mode.ir_search import EpistemicPlanSearch
from plan_mode.judges import JudgeAdapter, JudgeVerdict
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, SchemaMismatchError
import pytest


def _prov() -> Provenance:
    return Provenance(source_type=SourceType.USER_REQUIREMENT, rationale="test")


def test_registry_rejects_undeclared_extra_parameters():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(
        name="strict-cap", description="strict",
        input_schema={"x": {"type": "int", "required": True}},
    ))
    action = ActionIR(
        action_id="a", capability_name="strict-cap",
        parameters={"x": 1, "undeclared": "surprise"}, provenance=_prov(),
    )
    with pytest.raises(SchemaMismatchError, match="undeclared"):
        registry.validate_action(action)


def test_ast_transposition_hash_includes_semantic_outputs():
    engine = ASTSearchEngine(objective="x")
    ast_a = PlanAST(goal="x", actions=[ActionSchema(id=1, name="same", outputs=["a.txt"])])
    ast_b = PlanAST(goal="x", actions=[ActionSchema(id=1, name="same", outputs=["b.txt"])])
    assert engine._state_hash(ast_a) != engine._state_hash(ast_b)


class _CountingJudge(JudgeAdapter):
    def __init__(self):
        self.calls = 0

    async def evaluate(
        self, plan_ir, goal_description="", registry=None,
        observed_world_state=None, timeout=30.0,
    ):
        self.calls += 1
        return JudgeVerdict(
            verdict="PASS", feasibility_0_100=90, confidence=1.0,
            provider="counting", model="counting",
        )


def test_ir_search_judge_cache_is_bound_to_world_state():
    judge = _CountingJudge()
    search = EpistemicPlanSearch(judge=judge)
    plan_ir = PlanIR(plan_id="cache", goal_description="cache")
    fact_true = WorldFact(
        predicate="ready", args=[1], truth=FactTruth.VERIFIED_TRUE,
        provenance=_prov(),
    )
    fact_false = fact_true.model_copy(update={"truth": FactTruth.VERIFIED_FALSE})
    search._run_judge(plan_ir, [fact_true])
    search._run_judge(plan_ir, [fact_false])
    assert judge.calls == 2
