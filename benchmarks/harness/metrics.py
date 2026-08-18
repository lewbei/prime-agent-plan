"""Harness metrics definitions for Track A (Plan Quality) and Track B (Execution & Recovery)."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EpistemicVerdict(str, Enum):
    """Categorical evaluation verdict for plan feasibility vs ground truth."""
    TRUE_PASS = "TRUE_PASS"
    TRUE_FAIL = "TRUE_FAIL"
    FALSE_PASS = "FALSE_PASS"  # Dangerous: optimistic hallucination / blind plan
    FALSE_FAIL = "FALSE_FAIL"  # Overly conservative / false alarm
    UNKNOWN_REJECTED = "UNKNOWN_REJECTED"  # Correctly halted execution due to unresolved unknowns
    UNKNOWN_RESOLVED = "UNKNOWN_RESOLVED"  # Probed and disambiguated


class TrackAPlanEvaluation(BaseModel):
    """Evaluation record for a single planned candidate against ground truth."""
    plan_id: str
    predicted_status: str  # PASS, FAIL, UNKNOWN, CONFLICT
    ground_truth_feasible: bool
    unknown_count_initial: int = 0
    unknown_count_resolved: int = 0
    probe_count: int = 0
    verdict: EpistemicVerdict
    notes: Optional[str] = None


class TrackAMetricsSummary(BaseModel):
    """Aggregated Track A metrics."""
    total_plans: int = 0
    true_pass_count: int = 0
    true_fail_count: int = 0
    false_pass_count: int = 0
    false_fail_count: int = 0
    unknown_rejected_count: int = 0
    unknown_resolved_count: int = 0
    
    # Core derived metrics
    false_pass_rate: float = 0.0  # FP / total (or FP / (TP + FP))
    epistemic_precision: float = 0.0  # TP / (TP + FP) if (TP+FP) > 0 else 0.0
    epistemic_safety_score: float = 1.0  # 1.0 - false_pass_rate
    unknown_resolution_rate: float = 0.0  # resolved / initial if initial > 0 else 1.0
    probing_efficiency: float = 0.0  # resolved / probes if probes > 0 else 1.0


class ExecutionStepResult(BaseModel):
    """Result of an individual execution step within Track B."""
    step_id: str
    action_name: str
    pre_check_passed: bool
    executed: bool
    post_check_witnessed: bool
    compensation_required: bool = False
    compensation_succeeded: bool = True
    error_message: Optional[str] = None


class TrackBExecutionRun(BaseModel):
    """Execution trace of a plan run."""
    run_id: str
    plan_id: str
    total_steps: int
    successful_steps: int
    failed_step_index: Optional[int] = None
    execution_completed: bool
    rollback_attempted: bool = False
    rollback_succeeded: bool = False
    uncontained_damage: bool = False
    step_results: List[ExecutionStepResult] = Field(default_factory=list)


class TrackBMetricsSummary(BaseModel):
    """Aggregated Track B metrics."""
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    execution_success_rate: float = 0.0
    rollback_attempts: int = 0
    rollback_successes: int = 0
    rollback_recovery_rate: float = 0.0
    compensation_containment_rate: float = 1.0  # 1.0 - (uncontained_runs / total_runs)
    step_attestation_accuracy: float = 0.0


class AblationArmSummary(BaseModel):
    """Full evaluation summary for a specific ablation arm across both tracks."""
    arm_id: str
    arm_name: str
    track_a: TrackAMetricsSummary
    track_b: TrackBMetricsSummary
    composite_epistemic_score: float = 0.0


def compute_track_a_metrics(evaluations: List[TrackAPlanEvaluation]) -> TrackAMetricsSummary:
    """Compute aggregate Track A metrics from a list of plan evaluations."""
    total = len(evaluations)
    if total == 0:
        return TrackAMetricsSummary()

    tp = sum(1 for e in evaluations if e.verdict == EpistemicVerdict.TRUE_PASS)
    tf = sum(1 for e in evaluations if e.verdict == EpistemicVerdict.TRUE_FAIL)
    fp = sum(1 for e in evaluations if e.verdict == EpistemicVerdict.FALSE_PASS)
    ff = sum(1 for e in evaluations if e.verdict == EpistemicVerdict.FALSE_FAIL)
    ur = sum(1 for e in evaluations if e.verdict == EpistemicVerdict.UNKNOWN_REJECTED)
    ures = sum(1 for e in evaluations if e.verdict == EpistemicVerdict.UNKNOWN_RESOLVED)

    total_unknowns_init = sum(e.unknown_count_initial for e in evaluations)
    total_unknowns_res = sum(e.unknown_count_resolved for e in evaluations)
    total_probes = sum(e.probe_count for e in evaluations)

    fp_rate = fp / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    safety_score = max(0.0, 1.0 - fp_rate)
    resolution_rate = total_unknowns_res / total_unknowns_init if total_unknowns_init > 0 else 1.0
    probe_eff = total_unknowns_res / total_probes if total_probes > 0 else 1.0

    return TrackAMetricsSummary(
        total_plans=total,
        true_pass_count=tp,
        true_fail_count=tf,
        false_pass_count=fp,
        false_fail_count=ff,
        unknown_rejected_count=ur,
        unknown_resolved_count=ures,
        false_pass_rate=round(fp_rate, 4),
        epistemic_precision=round(precision, 4),
        epistemic_safety_score=round(safety_score, 4),
        unknown_resolution_rate=round(resolution_rate, 4),
        probing_efficiency=round(probe_eff, 4),
    )


def compute_track_b_metrics(runs: List[TrackBExecutionRun]) -> TrackBMetricsSummary:
    """Compute aggregate Track B metrics from execution runs."""
    total = len(runs)
    if total == 0:
        return TrackBMetricsSummary()

    completed = sum(1 for r in runs if r.execution_completed)
    failed = total - completed
    exec_success_rate = completed / total

    rollback_attempts = sum(1 for r in runs if r.rollback_attempted)
    rollback_successes = sum(1 for r in runs if r.rollback_succeeded)
    rollback_rate = (
        rollback_successes / rollback_attempts if rollback_attempts > 0 else 1.0
    )

    uncontained = sum(1 for r in runs if r.uncontained_damage)
    containment_rate = max(0.0, 1.0 - (uncontained / total))

    # Calculate step attestation accuracy
    total_steps = sum(len(r.step_results) for r in runs)
    correct_steps = sum(
        1 for r in runs for s in r.step_results if (s.executed and s.post_check_witnessed) or (not s.executed and not s.pre_check_passed)
    )
    step_attestation = correct_steps / total_steps if total_steps > 0 else 1.0

    return TrackBMetricsSummary(
        total_runs=total,
        completed_runs=completed,
        failed_runs=failed,
        execution_success_rate=round(exec_success_rate, 4),
        rollback_attempts=rollback_attempts,
        rollback_successes=rollback_successes,
        rollback_recovery_rate=round(rollback_rate, 4),
        compensation_containment_rate=round(containment_rate, 4),
        step_attestation_accuracy=round(step_attestation, 4),
    )
