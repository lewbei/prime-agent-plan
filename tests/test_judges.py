"""Tests for Dual Divergence Judges (Blind vs Grounded)."""

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
from plan_mode.judges import (
    JudgeVerdict,
    DualJudgeComparison,
    BlindJudge,
    GroundedEpistemicJudge,
    DualJudgeEvaluator,
)


def test_blind_vs_grounded_divergence_optimism():
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    
    # Missing/unknown precondition in ground truth
    plan = PlanIR(
        plan_id="plan_diverge_001",
        goal_description="Migrate customer records",
        initial_state=[
            WorldFact(predicate="dest_partition_mounted", args=[], truth=FactTruth.UNKNOWN, provenance=prov)
        ],
        actions=[
            ActionIR(
                action_id="step_1",
                capability_name="fs.migrate_data",
                parameters={"src": "/mnt/old", "dst": "/mnt/new"},
                preconditions=[PredicateCondition(predicate="dest_partition_mounted", args=[])],
                provenance=prov,
            )
        ],
    )

    evaluator = DualJudgeEvaluator()
    comparison = evaluator.evaluate_plan(plan)

    # Blind judge sees plausible text and predicts PASS
    assert comparison.blind_verdict.verdict == "PASS"
    # Grounded judge detects UNKNOWN precondition
    assert comparison.grounded_verdict.verdict == "UNKNOWN"
    assert comparison.blind_optimism_detected is True
    assert comparison.verdict_concordance is False
    assert comparison.confidence_divergence > 0.0


def test_concordance_when_plan_is_grounded_pass():
    prov = Provenance(source_type=SourceType.OBSERVED_WORLD_STATE)
    plan = PlanIR(
        plan_id="plan_concord_001",
        goal_description="Restart web service",
        initial_state=[
            WorldFact(predicate="service_installed", args=["nginx"], truth=FactTruth.VERIFIED_TRUE, provenance=prov)
        ],
        actions=[
            ActionIR(
                action_id="step_1",
                capability_name="system.restart_service",
                parameters={"name": "nginx"},
                preconditions=[PredicateCondition(predicate="service_installed", args=["nginx"])],
                positive_effects=[PredicateCondition(predicate="service_running", args=["nginx"])],
                provenance=prov,
            )
        ],
    )

    evaluator = DualJudgeEvaluator()
    comparison = evaluator.evaluate_plan(plan)

    assert comparison.blind_verdict.verdict == "PASS"
    assert comparison.grounded_verdict.verdict == "PASS"
    assert comparison.verdict_concordance is True
    assert comparison.blind_optimism_detected is False
