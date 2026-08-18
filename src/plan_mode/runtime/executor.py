"""Execution Plan Manager with Precondition Checking, Process Execution, and Postcondition Witnessing."""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import ActionIR, FactTruth, PlanIR, PredicateCondition, WorldFact
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier
from plan_mode.session import AuthorizationCertificate, PlanningSession, SessionState
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult


class WitnessStatus(str, Enum):
    """Result of empirical postcondition witnessing."""
    WITNESSED_TRUE = "WITNESSED_TRUE"
    WITNESSED_FALSE = "WITNESSED_FALSE"
    WITNESS_CONFLICT = "WITNESS_CONFLICT"


class PreconditionFailedError(Exception):
    """Raised when an action's live precondition is false prior to execution."""
    pass


class StepExecutionResult(BaseModel):
    """Outcome of an individual action step execution and observation."""
    step_id: str
    capability_name: str
    exit_code: int
    witness_status: WitnessStatus
    duration_ms: float
    error_message: Optional[str] = None


class ExecutionSummary(BaseModel):
    """Summary of entire plan execution run."""
    plan_id: str
    plan_version: int
    success: bool
    step_results: List[StepExecutionResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    failed_step_id: Optional[str] = None


class ExecutionPlanManager:
    """Coordinates safe plan execution, ledger recording, and empirical verification."""

    def __init__(
        self,
        session: PlanningSession,
        registry: CapabilityRegistry,
        ledger: EvidenceLedger,
        sandbox: Optional[ExecutionSandbox] = None,
    ):
        self.session = session
        self.registry = registry
        self.ledger = ledger
        self.sandbox = sandbox or ExecutionSandbox()

    def execute_authorized_plan(
        self,
        certificate: AuthorizationCertificate,
        custom_action_handler: Optional[Callable[[ActionIR, Dict[str, Any]], SandboxExecutionResult]] = None,
    ) -> ExecutionSummary:
        start_time = time.time()
        
        # Verify certificate
        if not certificate.verify_signature(self.session.secret_key):
            raise ValueError("Authorization certificate signature invalid.")
        if certificate.is_expired():
            raise ValueError("Authorization certificate has expired.")

        version_obj = self.session.versions[certificate.plan_version]
        plan_ir = version_obj.plan_ir

        step_results: List[StepExecutionResult] = []
        current_state: Dict[str, WorldFact] = {f.fact_key: f for f in plan_ir.initial_state}

        for action in plan_ir.actions:
            step_start = time.time()
            step_id = action.action_id
            cap = self.registry.get(action.capability_name)

            # 1. Precondition re-check
            pre_passed = True
            for pre in action.preconditions:
                key = pre.fact_key
                fact = current_state.get(key)
                if fact is None or fact.truth != pre.expected_truth:
                    pre_passed = False
                    break

            self.ledger.append_record(
                LedgerEventType.PRECHECK_EVALUATED,
                {
                    "step_id": step_id,
                    "passed": pre_passed,
                },
            )

            if not pre_passed:
                duration = (time.time() - step_start) * 1000.0
                res = StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=1,
                    witness_status=WitnessStatus.WITNESSED_FALSE,
                    duration_ms=round(duration, 2),
                    error_message=f"Live precondition check failed for step '{step_id}'.",
                )
                step_results.append(res)
                self.ledger.append_record(
                    LedgerEventType.PLAN_ABORTED,
                    {"failed_step_id": step_id, "reason": res.error_message},
                )
                total_duration = (time.time() - start_time) * 1000.0
                return ExecutionSummary(
                    plan_id=plan_ir.plan_id,
                    plan_version=plan_ir.version,
                    success=False,
                    step_results=step_results,
                    total_duration_ms=round(total_duration, 2),
                    failed_step_id=step_id,
                )

            # 2. Execute Action
            if custom_action_handler:
                exec_res = custom_action_handler(action, action.parameters)
            else:
                # Default mock execution using echo/test or capability verifier template
                exec_res = self._execute_default_capability(action, cap)

            self.ledger.append_record(
                LedgerEventType.ACTION_EXECUTED,
                {
                    "step_id": step_id,
                    "capability": action.capability_name,
                    "exit_code": exec_res.returncode,
                    "duration_ms": exec_res.duration_ms,
                },
            )

            if exec_res.returncode != 0:
                duration = (time.time() - step_start) * 1000.0
                res = StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=exec_res.returncode,
                    witness_status=WitnessStatus.WITNESSED_FALSE,
                    duration_ms=round(duration, 2),
                    error_message=exec_res.stderr or "Process returned non-zero exit code.",
                )
                step_results.append(res)
                self.ledger.append_record(
                    LedgerEventType.PLAN_ABORTED,
                    {"failed_step_id": step_id, "reason": res.error_message},
                )
                total_duration = (time.time() - start_time) * 1000.0
                return ExecutionSummary(
                    plan_id=plan_ir.plan_id,
                    plan_version=plan_ir.version,
                    success=False,
                    step_results=step_results,
                    total_duration_ms=round(total_duration, 2),
                    failed_step_id=step_id,
                )

            # 3. Postcondition Witnessing
            witness_status = self._witness_postconditions(action, cap)
            self.ledger.append_record(
                LedgerEventType.POSTCHECK_WITNESSED,
                {
                    "step_id": step_id,
                    "witness_status": witness_status.value,
                },
            )

            step_duration = (time.time() - step_start) * 1000.0
            step_results.append(
                StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=0,
                    witness_status=witness_status,
                    duration_ms=round(step_duration, 2),
                )
            )

            if witness_status != WitnessStatus.WITNESSED_TRUE:
                self.ledger.append_record(
                    LedgerEventType.PLAN_ABORTED,
                    {"failed_step_id": step_id, "reason": "Postcondition witnessing failed."},
                )
                total_duration = (time.time() - start_time) * 1000.0
                return ExecutionSummary(
                    plan_id=plan_ir.plan_id,
                    plan_version=plan_ir.version,
                    success=False,
                    step_results=step_results,
                    total_duration_ms=round(total_duration, 2),
                    failed_step_id=step_id,
                )

        # Plan completed successfully
        self.ledger.append_record(
            LedgerEventType.PLAN_COMMITTED,
            {"plan_id": plan_ir.plan_id, "version": plan_ir.version},
        )
        if self.session.current_state == SessionState.EXECUTING:
            self.session.commit_execution()

        total_duration = (time.time() - start_time) * 1000.0
        return ExecutionSummary(
            plan_id=plan_ir.plan_id,
            plan_version=plan_ir.version,
            success=True,
            step_results=step_results,
            total_duration_ms=round(total_duration, 2),
        )

    def _execute_default_capability(self, action: ActionIR, cap: CapabilityEntry) -> SandboxExecutionResult:
        """Run default safe capability execution in sandbox."""
        pipeline = [["true"]]
        return self.sandbox.execute_argv_pipeline(pipeline, timeout_seconds=action.timeout_seconds)

    def _witness_postconditions(self, action: ActionIR, cap: CapabilityEntry) -> WitnessStatus:
        """Execute verifiers declared in capability contract."""
        if not cap.verifiers:
            return WitnessStatus.WITNESSED_TRUE

        for v in cap.verifiers:
            cmd = []
            for token in v.command_template:
                if token.startswith("$"):
                    param_key = token[1:]
                    val = str(action.parameters.get(param_key, ""))
                    cmd.append(val)
                else:
                    cmd.append(token)

            res = self.sandbox.execute_argv_pipeline([cmd], timeout_seconds=v.timeout_seconds)
            if res.returncode != 0:
                return WitnessStatus.WITNESSED_FALSE

            if v.expected_output_pattern:
                if not re.search(v.expected_output_pattern, res.stdout):
                    return WitnessStatus.WITNESSED_FALSE

        return WitnessStatus.WITNESSED_TRUE
