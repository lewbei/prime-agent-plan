"""Unit tests for benchmark harness metrics."""

import pytest
from benchmarks.harness.metrics import (
    EpistemicVerdict,
    TrackAPlanEvaluation,
    TrackBExecutionRun,
    ExecutionStepResult,
    compute_track_a_metrics,
    compute_track_b_metrics,
)


def test_track_a_metrics_computation():
    evals = [
        TrackAPlanEvaluation(
            plan_id="p1",
            predicted_status="PASS",
            ground_truth_feasible=True,
            verdict=EpistemicVerdict.TRUE_PASS,
            unknown_count_initial=2,
            unknown_count_resolved=2,
            probe_count=2,
        ),
        TrackAPlanEvaluation(
            plan_id="p2",
            predicted_status="PASS",
            ground_truth_feasible=False,
            verdict=EpistemicVerdict.FALSE_PASS,
        ),
        TrackAPlanEvaluation(
            plan_id="p3",
            predicted_status="FAIL",
            ground_truth_feasible=False,
            verdict=EpistemicVerdict.TRUE_FAIL,
        ),
        TrackAPlanEvaluation(
            plan_id="p4",
            predicted_status="UNKNOWN",
            ground_truth_feasible=False,
            verdict=EpistemicVerdict.UNKNOWN_REJECTED,
        ),
    ]

    summary = compute_track_a_metrics(evals)
    assert summary.total_plans == 4
    assert summary.true_pass_count == 1
    assert summary.false_pass_count == 1
    assert summary.true_fail_count == 1
    assert summary.unknown_rejected_count == 1
    assert summary.false_pass_rate == 0.25
    assert summary.epistemic_precision == 0.5  # TP / (TP + FP) = 1 / 2
    assert summary.epistemic_safety_score == 0.75
    assert summary.unknown_resolution_rate == 1.0


def test_track_b_metrics_computation():
    runs = [
        TrackBExecutionRun(
            run_id="r1",
            plan_id="p1",
            total_steps=3,
            successful_steps=3,
            execution_completed=True,
            step_results=[
                ExecutionStepResult(step_id="s1", action_name="a1", pre_check_passed=True, executed=True, post_check_witnessed=True),
                ExecutionStepResult(step_id="s2", action_name="a2", pre_check_passed=True, executed=True, post_check_witnessed=True),
                ExecutionStepResult(step_id="s3", action_name="a3", pre_check_passed=True, executed=True, post_check_witnessed=True),
            ]
        ),
        TrackBExecutionRun(
            run_id="r2",
            plan_id="p2",
            total_steps=3,
            successful_steps=1,
            failed_step_index=1,
            execution_completed=False,
            rollback_attempted=True,
            rollback_succeeded=True,
            uncontained_damage=False,
            step_results=[
                ExecutionStepResult(step_id="s1", action_name="a1", pre_check_passed=True, executed=True, post_check_witnessed=True),
                ExecutionStepResult(step_id="s2", action_name="a2", pre_check_passed=False, executed=False, post_check_witnessed=False),
            ]
        ),
    ]

    summary = compute_track_b_metrics(runs)
    assert summary.total_runs == 2
    assert summary.completed_runs == 1
    assert summary.failed_runs == 1
    assert summary.execution_success_rate == 0.5
    assert summary.rollback_attempts == 1
    assert summary.rollback_successes == 1
    assert summary.rollback_recovery_rate == 1.0
    assert summary.compensation_containment_rate == 1.0
    assert summary.step_attestation_accuracy == 1.0
