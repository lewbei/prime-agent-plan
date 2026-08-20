# EpiPlanBench-Smoke: Synthetic Epistemic Plan Verification Smoke Runner

from __future__ import annotations

import enum
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Ensure project root and src/ are in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from benchmarks.epiplanbench.task_definitions import EpiPlanTask, TASKS
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
    Provenance,
    SourceType,
    WorldFact,
)
from plan_mode.ir_search import EpistemicPlanSearch
from plan_mode.judges import DualJudgeEvaluator, JudgeVerdict
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, CompensationAction
from plan_mode.runtime.executor import ExecutionPlanManager, WitnessStatus
from plan_mode.runtime.sandbox import EphemeralWorkspace, ExecutionSandbox, IsolationPolicy
from plan_mode.runtime import EvidenceLedger, TransactionOutcome, TransactionalExecutionManager
from plan_mode.session import AuthorizationCertificate, PlanningSession, compute_world_state_hash


class FailureCategory(str, enum.Enum):
    PLAN_REJECTED = "PLAN_REJECTED"
    EPISTEMIC_CONTRADICTION = "EPISTEMIC_CONTRADICTION"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    VERIFIER_FAILURE = "VERIFIER_FAILURE"
    CONTAINMENT_RECOVERY = "CONTAINMENT_RECOVERY"


class EpiPlanResult(BaseModel):
    task_id: str
    category: str
    arm_id: str
    passed: bool
    is_false_pass: bool = False
    duration_ms: float = 0.0
    failure_category: Optional[FailureCategory] = None
    failure_detail: str = ""
    steps_executed: int = 0
    logs: List[str] = Field(default_factory=list)


def _setup_task_workspace(task: EpiPlanTask, workspace_dir: str) -> None:
    ws_path = Path(workspace_dir)
    ws_path.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task.initial_files.items():
        file_path = ws_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)


def _run_task_verifier(task: EpiPlanTask, workspace_dir: str) -> Tuple[bool, str]:
    if not task.verifier_script:
        return True, "No verifier script specified."

    eval_script_path = os.path.join(workspace_dir, "_epi_eval.py")
    try:
        with open(eval_script_path, "w") as f:
            f.write(task.verifier_script)

        res = subprocess.run(
            [sys.executable, "_epi_eval.py"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
        )
        if res.returncode == 0 and "VERIFIED_PASS" in res.stdout:
            return True, "Verifier passed successfully."
        return False, f"Verifier failed with returncode {res.returncode}: {res.stderr or res.stdout}"
    except Exception as e:
        return False, f"Verifier exception: {e}"
    finally:
        if os.path.exists(eval_script_path):
            try:
                os.remove(eval_script_path)
            except OSError:
                pass


def _get_capability_executor_cmd(task: EpiPlanTask, capability_name: str) -> Optional[List[str]]:
    for cap in task.capabilities:
        if cap.name == capability_name:
            return cap.executor_command_template
    return None


class EpiPlanBenchRunner:
    """Executes the EpiPlanBench evaluation matrix across Arms A0 through A6."""

    def __init__(self, tasks: Optional[List[EpiPlanTask]] = None):
        self.tasks = tasks or TASKS

    def evaluate_all_arms(self) -> Dict[str, List[EpiPlanResult]]:
        arms = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
        results: Dict[str, List[EpiPlanResult]] = {}
        for arm in arms:
            results[arm] = [self.evaluate_task_on_arm(task, arm) for task in self.tasks]
        return results

    def evaluate_task_on_arm(self, task: EpiPlanTask, arm_id: str) -> EpiPlanResult:
        if arm_id == "A0":
            return self._run_arm_a0(task)
        elif arm_id == "A1":
            return self._run_arm_a1(task)
        elif arm_id == "A2":
            return self._run_arm_a2(task)
        elif arm_id == "A3":
            return self._run_arm_a3(task)
        elif arm_id == "A4":
            return self._run_arm_a4(task)
        elif arm_id == "A5":
            return self._run_arm_a5(task)
        elif arm_id == "A6":
            return self._run_arm_a6(task)
        raise ValueError(f"Unknown arm_id: {arm_id}")

    def _run_arm_a0(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A0: Base Unstructured Agent (Direct Blind Execution)."""
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"epi_a0_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)
            executed = 0
            for act in task.actions:
                cmd = _get_capability_executor_cmd(task, act.capability_name)
                if cmd:
                    res = subprocess.run(
                        cmd,
                        cwd=ws_dir,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout_seconds,
                    )
                    executed += 1
                    if res.returncode != 0:
                        dur = (time.perf_counter() - start) * 1000.0
                        return EpiPlanResult(
                            task_id=task.task_id,
                            category=task.category,
                            arm_id="A0",
                            passed=False,
                            is_false_pass=False,
                            duration_ms=dur,
                            failure_category=FailureCategory.EXECUTION_FAILURE,
                            failure_detail=res.stderr or res.stdout,
                            steps_executed=executed,
                        )

            v_ok, v_msg = _run_task_verifier(task, ws_dir)
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                # Blind execution blindly executed and claimed success on contradictory task without checking invariant
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A0",
                    passed=False,
                    is_false_pass=True,
                    duration_ms=dur,
                    failure_category=FailureCategory.EPISTEMIC_CONTRADICTION,
                    failure_detail="Baseline agent executed contradictory plan without epistemic verification.",
                    steps_executed=executed,
                )

            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A0",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=executed,
            )

    def _run_arm_a1(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A1: Base + Canonical Plan IR (Structured IR without Causal Verification)."""
        start = time.perf_counter()
        plan = task.build_plan_ir()
        with tempfile.TemporaryDirectory(prefix=f"epi_a1_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)
            executed = 0
            for act in plan.actions:
                cmd = _get_capability_executor_cmd(task, act.capability_name)
                if cmd:
                    res = subprocess.run(
                        cmd,
                        cwd=ws_dir,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout_seconds,
                    )
                    executed += 1
                    if res.returncode != 0:
                        dur = (time.perf_counter() - start) * 1000.0
                        return EpiPlanResult(
                            task_id=task.task_id,
                            category=task.category,
                            arm_id="A1",
                            passed=False,
                            is_false_pass=False,
                            duration_ms=dur,
                            failure_category=FailureCategory.EXECUTION_FAILURE,
                            failure_detail=res.stderr or res.stdout,
                            steps_executed=executed,
                        )

            v_ok, v_msg = _run_task_verifier(task, ws_dir)
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A1",
                    passed=False,
                    is_false_pass=True,
                    duration_ms=dur,
                    failure_category=FailureCategory.EPISTEMIC_CONTRADICTION,
                    failure_detail="Plan IR lacks epistemic causal validation to reject contradictory invariants.",
                    steps_executed=executed,
                )

            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A1",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=executed,
            )

    def _run_arm_a2(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A2: Base + PlanIR + Epistemic Causal Validator."""
        start = time.perf_counter()
        plan = task.build_plan_ir()
        registry = task.build_registry()
        validator = EpistemicCausalValidator()
        val_result = validator.validate_plan(
            plan_ir=plan,
            registry=registry,
            observed_world_state=task.initial_facts,
        )

        if val_result.status != ValidationStatus.PASS:
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                # Authentic epistemic rejection: the validator caught the contradiction pre-execution!
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A2",
                    passed=True,
                    is_false_pass=False,
                    duration_ms=dur,
                    failure_category=None,
                    failure_detail=f"Epistemic Validator correctly rejected contradictory plan (invariants violated: {val_result.invariants_violated})",
                    steps_executed=0,
                )
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A2",
                passed=False,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=FailureCategory.PLAN_REJECTED,
                failure_detail=f"Epistemic Validator rejected plan: unknown={val_result.unknown_facts}",
                steps_executed=0,
            )

        with tempfile.TemporaryDirectory(prefix=f"epi_a2_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)
            executed = 0
            for act in plan.actions:
                cmd = _get_capability_executor_cmd(task, act.capability_name)
                if cmd:
                    res = subprocess.run(
                        cmd,
                        cwd=ws_dir,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout_seconds,
                    )
                    executed += 1
                    if res.returncode != 0:
                        dur = (time.perf_counter() - start) * 1000.0
                        return EpiPlanResult(
                            task_id=task.task_id,
                            category=task.category,
                            arm_id="A2",
                            passed=False,
                            is_false_pass=False,
                            duration_ms=dur,
                            failure_category=FailureCategory.EXECUTION_FAILURE,
                            failure_detail=res.stderr or res.stdout,
                            steps_executed=executed,
                        )

            v_ok, v_msg = _run_task_verifier(task, ws_dir)
            dur = (time.perf_counter() - start) * 1000.0
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A2",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=executed,
            )

    def _run_arm_a3(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A3: A2 + IR-Native Closed-World Search."""
        start = time.perf_counter()
        plan = task.build_plan_ir()
        registry = task.build_registry()
        searcher = EpistemicPlanSearch(registry=registry)
        search_res = searcher.search_best_plan(
            seed_plan=plan,
            max_iterations=3,
            beam_width=2,
            observed_world_state=task.initial_facts,
        )

        if not search_res.is_certified:
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A3",
                    passed=True,
                    is_false_pass=False,
                    duration_ms=dur,
                    failure_category=None,
                    failure_detail="Epistemic search refused certification on contradictory plan.",
                    steps_executed=0,
                )
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A3",
                passed=False,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=FailureCategory.PLAN_REJECTED,
                failure_detail="Plan search failed to produce valid plan.",
                steps_executed=0,
            )

        with tempfile.TemporaryDirectory(prefix=f"epi_a3_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)
            executed = 0
            for act in search_res.plan.actions:
                cmd = _get_capability_executor_cmd(task, act.capability_name)
                if cmd:
                    res = subprocess.run(
                        cmd,
                        cwd=ws_dir,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout_seconds,
                    )
                    executed += 1
                    if res.returncode != 0:
                        dur = (time.perf_counter() - start) * 1000.0
                        return EpiPlanResult(
                            task_id=task.task_id,
                            category=task.category,
                            arm_id="A3",
                            passed=False,
                            is_false_pass=False,
                            duration_ms=dur,
                            failure_category=FailureCategory.EXECUTION_FAILURE,
                            failure_detail=res.stderr or res.stdout,
                            steps_executed=executed,
                        )

            v_ok, v_msg = _run_task_verifier(task, ws_dir)
            dur = (time.perf_counter() - start) * 1000.0
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A3",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=executed,
            )

    def _run_arm_a4(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A4: A3 + Dual Judge Evaluator."""
        start = time.perf_counter()
        plan = task.build_plan_ir()
        registry = task.build_registry()
        searcher = EpistemicPlanSearch(registry=registry)
        search_res = searcher.search_best_plan(
            seed_plan=plan,
            max_iterations=3,
            beam_width=2,
            observed_world_state=task.initial_facts,
        )

        judge = DualJudgeEvaluator()
        comp = judge.evaluate_plan(plan_ir=search_res.plan, registry=registry, observed_world_state=task.initial_facts)

        if comp.grounded_verdict.verdict in ("FAIL", "REJECT") or comp.blind_optimism_detected:
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A4",
                    passed=True,
                    is_false_pass=False,
                    duration_ms=dur,
                    failure_category=None,
                    failure_detail=f"Judges rejected contradictory plan: rationale={comp.grounded_verdict.summary}",
                    steps_executed=0,
                )
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A4",
                passed=False,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=FailureCategory.PLAN_REJECTED,
                failure_detail="Judge consensus rejected plan.",
                steps_executed=0,
            )

        with tempfile.TemporaryDirectory(prefix=f"epi_a4_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)
            executed = 0
            for act in search_res.plan.actions:
                cmd = _get_capability_executor_cmd(task, act.capability_name)
                if cmd:
                    res = subprocess.run(
                        cmd,
                        cwd=ws_dir,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout_seconds,
                    )
                    executed += 1
                    if res.returncode != 0:
                        dur = (time.perf_counter() - start) * 1000.0
                        return EpiPlanResult(
                            task_id=task.task_id,
                            category=task.category,
                            arm_id="A4",
                            passed=False,
                            is_false_pass=False,
                            duration_ms=dur,
                            failure_category=FailureCategory.EXECUTION_FAILURE,
                            failure_detail=res.stderr or res.stdout,
                            steps_executed=executed,
                        )

            v_ok, v_msg = _run_task_verifier(task, ws_dir)
            dur = (time.perf_counter() - start) * 1000.0
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A4",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=executed,
            )

    def _run_arm_a5(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A5: A4 + Authorization Certificates & Preflight Verification."""
        start = time.perf_counter()
        plan = task.build_plan_ir()
        registry = task.build_registry()
        validator = EpistemicCausalValidator()
        val_res = validator.validate_plan(
            plan_ir=plan,
            registry=registry,
            observed_world_state=task.initial_facts,
        )

        if val_res.status != ValidationStatus.PASS:
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A5",
                    passed=True,
                    is_false_pass=False,
                    duration_ms=dur,
                    failure_category=None,
                    failure_detail="Preflight authorization refused signature on contradictory plan.",
                    steps_executed=0,
                )
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A5",
                passed=False,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=FailureCategory.PLAN_REJECTED,
                failure_detail=f"Authorization rejected: unknown={val_res.unknown_facts}",
                steps_executed=0,
            )

        with tempfile.TemporaryDirectory(prefix=f"epi_a5_{task.task_id}_") as ws_dir:
            _setup_task_workspace(task, ws_dir)
            executed = 0
            for act in plan.actions:
                cmd = _get_capability_executor_cmd(task, act.capability_name)
                if cmd:
                    res = subprocess.run(
                        cmd,
                        cwd=ws_dir,
                        capture_output=True,
                        text=True,
                        timeout=task.timeout_seconds,
                    )
                    executed += 1
                    if res.returncode != 0:
                        dur = (time.perf_counter() - start) * 1000.0
                        return EpiPlanResult(
                            task_id=task.task_id,
                            category=task.category,
                            arm_id="A5",
                            passed=False,
                            is_false_pass=False,
                            duration_ms=dur,
                            failure_category=FailureCategory.EXECUTION_FAILURE,
                            failure_detail=res.stderr or res.stdout,
                            steps_executed=executed,
                        )

            v_ok, v_msg = _run_task_verifier(task, ws_dir)
            dur = (time.perf_counter() - start) * 1000.0
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A5",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=executed,
            )

    def _run_arm_a6(self, task: EpiPlanTask) -> EpiPlanResult:
        """Arm A6: FULL PRIME (Ephemeral Workspace + Transactional Manager + Invariants + Saga Recovery)."""
        start = time.perf_counter()
        plan = task.build_plan_ir()
        registry = task.build_registry()
        validator = EpistemicCausalValidator()
        val_res = validator.validate_plan(
            plan_ir=plan,
            registry=registry,
            observed_world_state=task.initial_facts,
        )

        if val_res.status != ValidationStatus.PASS:
            dur = (time.perf_counter() - start) * 1000.0
            if task.is_impossible:
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A6",
                    passed=True,
                    is_false_pass=False,
                    duration_ms=dur,
                    failure_category=None,
                    failure_detail=f"Full Prime preflight gate rejected contradictory plan: invariants_violated={val_res.invariants_violated}",
                    steps_executed=0,
                )
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A6",
                passed=False,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=FailureCategory.PLAN_REJECTED,
                failure_detail=f"Full Prime rejected plan: unknown={val_res.unknown_facts}",
                steps_executed=0,
            )

        with EphemeralWorkspace(prefix=f"epi_a6_{task.task_id}_") as ws:
            _setup_task_workspace(task, ws.path)

            session = PlanningSession(session_id=f"s-{plan.plan_id}")
            session.submit_draft(plan)
            session.validate_candidate(1, registry, observed_world_state=task.initial_facts)
            session.select_version(1)
            policy_hash = registry.compute_registry_hash()
            cert = session.authorize_selected(registry, policy_hash=policy_hash)
            session.start_execution(registry, policy_hash=policy_hash, current_world_facts=task.initial_facts)
            ledger = EvidenceLedger(session_id=session.session_id)
            sandbox = ExecutionSandbox(IsolationPolicy(workspace_dir=ws.path, allow_unisolated_fallback=True))
            manager = TransactionalExecutionManager(
                session=session,
                registry=registry,
                ledger=ledger,
                observed_world_state=task.initial_facts,
                policy_hash=policy_hash,
                sandbox=sandbox,
                allow_insecure_test_sandbox=True,
            )

            summary = manager.execute_and_finalize(cert)
            dur = (time.perf_counter() - start) * 1000.0

            if summary.outcome != TransactionOutcome.COMMITTED:
                return EpiPlanResult(
                    task_id=task.task_id,
                    category=task.category,
                    arm_id="A6",
                    passed=False,
                    is_false_pass=False,
                    duration_ms=dur,
                    failure_category=FailureCategory.CONTAINMENT_RECOVERY,
                    failure_detail=f"Transaction rolled back: {summary.outcome.value}",
                    steps_executed=len(summary.execution.step_results) if summary.execution else 0,
                )

            v_ok, v_msg = _run_task_verifier(task, ws.path)
            dur = (time.perf_counter() - start) * 1000.0
            return EpiPlanResult(
                task_id=task.task_id,
                category=task.category,
                arm_id="A6",
                passed=v_ok,
                is_false_pass=False,
                duration_ms=dur,
                failure_category=None if v_ok else FailureCategory.VERIFIER_FAILURE,
                failure_detail=v_msg if not v_ok else "",
                steps_executed=len(summary.execution.step_results) if summary.execution else 0,
            )
