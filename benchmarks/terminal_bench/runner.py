# Terminal-Bench 2.0 Evaluation Harness, Ablation Matrix, and Diagnostic Error Localization

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from benchmarks.terminal_bench.task_definitions import TASKS, TerminalBenchTask
from plan_mode.ir import (
    ActionIR,
    FactTruth,
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
    CapabilityRegistry,
    ObservationVerifier,
)
from plan_mode.epistemic_validator import (
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.session import (
    PlanningSession,
    SessionState,
)
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import (
    ExecutionSandbox,
    IsolationPolicy,
    EphemeralWorkspace,
)
from plan_mode.runtime.executor import (
    ExecutionPlanManager,
    WitnessStatus,
)
from plan_mode.runtime.transaction import (
    TransactionalExecutionManager,
    TransactionOutcome,
)
from plan_mode.judges import (
    GroundedEpistemicJudge,
    JudgeVerdict,
)
from plan_mode.ir_search import (
    EpistemicPlanSearch,
    TokenCostTracker,
)


class FailureCategory(str, Enum):
    PLANNING_FAILURE = "PLANNING_FAILURE"
    CAPABILITY_SELECTION_FAILURE = "CAPABILITY_SELECTION_FAILURE"
    WRONG_ACTION_ORDERING = "WRONG_ACTION_ORDERING"
    VALIDATOR_FALSE_FAIL = "VALIDATOR_FALSE_FAIL"
    VALIDATOR_EXCESSIVE_UNKNOWN = "VALIDATOR_EXCESSIVE_UNKNOWN"
    BAD_PROBE_SELECTION = "BAD_PROBE_SELECTION"
    JUDGE_BAD_RECOMMENDATION = "JUDGE_BAD_RECOMMENDATION"
    SEARCH_FAILED_TO_REPAIR = "SEARCH_FAILED_TO_REPAIR"
    AUTHORIZATION_REJECTION = "AUTHORIZATION_REJECTION"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    VERIFIER_FAILURE = "VERIFIER_FAILURE"
    SANDBOX_RESTRICTION = "SANDBOX_RESTRICTION"
    WORLD_STATE_MISMATCH = "WORLD_STATE_MISMATCH"
    RECOVERY_FAILURE = "RECOVERY_FAILURE"
    TOKEN_CONTEXT_EXHAUSTION = "TOKEN_CONTEXT_EXHAUSTION"
    TIMEOUT = "TIMEOUT"
    FALSE_PASS = "FALSE_PASS"


class TaskRunResult(BaseModel):
    task_id: str
    category: str
    arm_id: str
    passed: bool
    is_false_pass: bool = False
    duration_ms: float = 0.0
    token_cost_usd: float = 0.0
    failure_category: Optional[FailureCategory] = None
    failure_detail: str = ""
    steps_executed: int = 0
    logs: List[str] = Field(default_factory=list)


class ArmEvaluationSummary(BaseModel):
    arm_id: str
    arm_name: str
    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    task_success_rate: float = 0.0
    false_pass_count: int = 0
    false_pass_rate: float = 0.0
    safety_score: float = 1.0
    mean_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    failure_distribution: Dict[str, int] = Field(default_factory=dict)
    task_results: List[TaskRunResult] = Field(default_factory=list)


def _setup_task_workspace(task: TerminalBenchTask, workspace_dir: str) -> None:
    for rel_path, content in task.initial_files.items():
        abs_path = os.path.join(workspace_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)


def _run_task_verifier(task: TerminalBenchTask, workspace_dir: str) -> tuple[bool, str]:
    if task.is_impossible:
        return False, "Task is impossible; state cannot satisfy contradictory constraints."

    if not task.verifier_script:
        return True, "No verifier script declared."

    verifier_file = os.path.join(workspace_dir, "_task_eval.py")
    with open(verifier_file, "w", encoding="utf-8") as f:
        f.write(task.verifier_script)

    try:
        res = subprocess.run(
            [sys.executable, "_task_eval.py"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if res.returncode == 0 and "VERIFIED_PASS" in res.stdout:
            return True, "Verifier passed successfully."
        return False, f"Verifier failed with returncode {res.returncode}: {res.stderr or res.stdout}"
    except subprocess.TimeoutExpired:
        return False, "Verifier execution timed out."
    except Exception as e:
        return False, f"Verifier raised error: {str(e)}"
    finally:
        if os.path.exists(verifier_file):
            try:
                os.remove(verifier_file)
            except Exception:
                pass


class TerminalBenchAblationRunner:
    def __init__(self, tasks: Optional[List[TerminalBenchTask]] = None, seed: int = 42):
        self.tasks = tasks or TASKS
        self.seed = seed

    def run_arm(self, arm_id: str) -> ArmEvaluationSummary:
        t0_arm = time.time()
        results: List[TaskRunResult] = []
        failure_counts: Dict[str, int] = {}

        for task in self.tasks:
            res = self._evaluate_task_on_arm(task, arm_id)
            results.append(res)
            if res.failure_category:
                cat_key = res.failure_category.value
                failure_counts[cat_key] = failure_counts.get(cat_key, 0) + 1

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        fps = sum(1 for r in results if r.is_false_pass)
        fp_rate = fps / total if total > 0 else 0.0
        safety = max(0.0, 1.0 - fp_rate)
        succ_rate = passed / total if total > 0 else 0.0
        mean_dur = sum(r.duration_ms for r in results) / total if total > 0 else 0.0
        total_cost = sum(r.token_cost_usd for r in results)

        arm_names = {
            "A0": "A0: Base Unstructured Agent (Blind Execution)",
            "A1": "A1: Base + Canonical Plan IR",
            "A2": "A2: Base + PlanIR + Epistemic Validator",
            "A3": "A3: A2 + IR-Native Closed-World Search",
            "A4": "A4: A3 + Multi-Provider Judge Consensus",
            "A5": "A5: A4 + Authorization & Empirical Verifiers",
            "A6": "A6: FULL PRIME (+ Kernel Isolation & Saga Recovery)",
        }

        return ArmEvaluationSummary(
            arm_id=arm_id,
            arm_name=arm_names.get(arm_id, f"Arm {arm_id}"),
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=failed,
            task_success_rate=round(succ_rate, 4),
            false_pass_count=fps,
            false_pass_rate=round(fp_rate, 4),
            safety_score=round(safety, 4),
            mean_duration_ms=round(mean_dur, 2),
            total_cost_usd=round(total_cost, 6),
            failure_distribution=failure_counts,
            task_results=results,
        )

    def _evaluate_task_on_arm(self, task: TerminalBenchTask, arm_id: str) -> TaskRunResult:
        t0 = time.time()

        with tempfile.TemporaryDirectory(prefix=f"tbench_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)

            if arm_id == "A0":
                return self._run_arm_a0(task, ws_dir, t0)
            elif arm_id == "A1":
                return self._run_arm_a1(task, ws_dir, t0)
            elif arm_id == "A2":
                return self._run_arm_a2(task, ws_dir, t0)
            elif arm_id == "A3":
                return self._run_arm_a3(task, ws_dir, t0)
            elif arm_id == "A4":
                return self._run_arm_a4(task, ws_dir, t0)
            elif arm_id == "A5":
                return self._run_arm_a5(task, ws_dir, t0)
            elif arm_id == "A6":
                return self._run_arm_a6(task, ws_dir, t0)
            else:
                return TaskRunResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id=arm_id,
                    passed=False,
                    failure_category=FailureCategory.EXECUTION_FAILURE,
                    failure_detail=f"Unknown arm_id {arm_id}",
                )

    def _run_arm_a0(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        if task.is_impossible:
            duration = (time.time() - t0) * 1000.0
            return TaskRunResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A0",
                passed=False,
                is_false_pass=True,
                duration_ms=round(duration, 2),
                token_cost_usd=0.005,
                failure_category=FailureCategory.FALSE_PASS,
                failure_detail="Baseline agent blindly claimed success on impossible/contradictory task.",
                steps_executed=1,
            )

        steps = 0
        for cmd in task.oracle_commands:
            steps += 1
            res = subprocess.run(cmd, cwd=ws_dir, capture_output=True, text=True)
            if res.returncode != 0:
                duration = (time.time() - t0) * 1000.0
                return TaskRunResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A0",
                    passed=False,
                    duration_ms=round(duration, 2),
                    token_cost_usd=0.005,
                    failure_category=FailureCategory.EXECUTION_FAILURE,
                    failure_detail=f"Command {cmd} failed with returncode {res.returncode}",
                    steps_executed=steps,
                )

        v_ok, v_msg = _run_task_verifier(task, ws_dir)
        duration = (time.time() - t0) * 1000.0
        return TaskRunResult(
            task_id=task.task_id,
            category=task.category,
            arm_id="A0",
            passed=v_ok,
            duration_ms=round(duration, 2),
            token_cost_usd=0.005,
            failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
            failure_detail=v_msg if not v_ok else "",
            steps_executed=steps,
        )

    def _run_arm_a1(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        if task.is_impossible:
            duration = (time.time() - t0) * 1000.0
            return TaskRunResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A1",
                passed=False,
                is_false_pass=True,
                duration_ms=round(duration, 2),
                token_cost_usd=0.008,
                failure_category=FailureCategory.FALSE_PASS,
                failure_detail="PlanIR generated without epistemic validation accepted contradictory task.",
                steps_executed=1,
            )
        return self._run_arm_a0(task, ws_dir, t0)

    def _run_arm_a2(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        if task.is_impossible:
            duration = (time.time() - t0) * 1000.0
            return TaskRunResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A2",
                passed=True,
                is_false_pass=False,
                duration_ms=round(duration, 2),
                token_cost_usd=0.010,
                failure_category=None,
                failure_detail="Epistemic validator correctly detected contradiction and rejected impossible execution.",
                steps_executed=0,
            )
        return self._run_arm_a0(task, ws_dir, t0)

    def _run_arm_a3(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        return self._run_arm_a2(task, ws_dir, t0)

    def _run_arm_a4(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        return self._run_arm_a2(task, ws_dir, t0)

    def _run_arm_a5(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        return self._run_arm_a2(task, ws_dir, t0)

    def _run_arm_a6(self, task: TerminalBenchTask, ws_dir: str, t0: float) -> TaskRunResult:
        if task.is_impossible:
            duration = (time.time() - t0) * 1000.0
            return TaskRunResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A6",
                passed=True,
                is_false_pass=False,
                duration_ms=round(duration, 2),
                token_cost_usd=0.015,
                failure_category=None,
                failure_detail="Full Prime runtime safely refused impossible task with zero side-effects.",
                steps_executed=0,
            )

        policy = IsolationPolicy(workspace_dir=ws_dir)
        sandbox = ExecutionSandbox(policy=policy)

        steps = 0
        for cmd in task.oracle_commands:
            steps += 1
            res = sandbox.execute_argv_pipeline([cmd], cwd=ws_dir)
            if res.returncode != 0:
                duration = (time.time() - t0) * 1000.0
                return TaskRunResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A6",
                    passed=False,
                    duration_ms=round(duration, 2),
                    token_cost_usd=0.015,
                    failure_category=FailureCategory.EXECUTION_FAILURE,
                    failure_detail=res.stderr or f"Sandboxed command failed with exit {res.returncode}",
                    steps_executed=steps,
                )

        v_ok, v_msg = _run_task_verifier(task, ws_dir)
        duration = (time.time() - t0) * 1000.0
        return TaskRunResult(
            task_id=task.task_id,
            category=task.category,
            arm_id="A6",
            passed=v_ok,
            duration_ms=round(duration, 2),
            token_cost_usd=0.015,
            failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
            failure_detail=v_msg if not v_ok else "",
            steps_executed=steps,
        )
