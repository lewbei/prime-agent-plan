"""Execution Plan Manager with Precondition Checking, Process Execution, and Postcondition Witnessing."""

from __future__ import annotations

import copy
import re
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    ProjectedTruth,
    Provenance,
    SourceType,
    WitnessabilityStatus,
    WorldFact,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier, typed_args_equal
from plan_mode.epistemic_validator import normalize_trusted_snapshot
from plan_mode.session import (
    AuthorizationCertificate,
    PlanningSession,
    SessionState,
    StateDriftError,
    SignatureVerificationError,
)
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult


class WitnessStatus(str, Enum):
    """Result of empirical postcondition witnessing."""
    WITNESSED_TRUE = "WITNESSED_TRUE"
    WITNESSED_FALSE = "WITNESSED_FALSE"
    WITNESS_CONFLICT = "WITNESS_CONFLICT"
    UNWITNESSED = "UNWITNESSED"


class PreconditionFailedError(Exception):
    """Raised when an action's live precondition is false prior to execution."""
    pass


class ExecutionContractMissingError(Exception):
    """Raised when a capability lacks a concrete executor contract for runtime execution."""
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
    live_world_state: Dict[str, WorldFact] = Field(default_factory=dict)


def _resolve_template_tokens(tokens: List[str], params: Dict[str, Any]) -> List[str]:
    """Replace {param} and $param template placeholders in command tokens."""
    resolved: List[str] = []
    for token in tokens:
        item = token
        for k, v in params.items():
            item = item.replace(f"{{{k}}}", str(v))
            item = item.replace(f"${k}", str(v))
        resolved.append(item)
    return resolved


class ExecutionPlanManager:
    """Coordinates safe plan execution, ledger recording, and empirical verification."""

    def __init__(
        self,
        session: PlanningSession,
        registry: CapabilityRegistry,
        ledger: EvidenceLedger,
        sandbox: Optional[ExecutionSandbox] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ):
        self.session = session
        self.registry = registry
        self.ledger = ledger
        self.sandbox = sandbox or ExecutionSandbox()
        self.live_world_state: Dict[str, WorldFact] = normalize_trusted_snapshot(observed_world_state)

    def execute_authorized_plan(
        self,
        certificate: AuthorizationCertificate,
        custom_action_handler: Optional[Callable[[ActionIR, Dict[str, Any]], SandboxExecutionResult]] = None,
    ) -> ExecutionSummary:
        start_time = time.time()

        # 1. Verify authorization certificate signature
        if not certificate.verify_signature(self.session.secret_key):
            raise SignatureVerificationError("Authorization certificate signature invalid or tampered.")
        if certificate.is_expired():
            raise ValueError("Authorization certificate has expired.")

        # 2. Check plan version existence and hash identity
        if certificate.plan_version not in self.session.versions:
            raise ValueError(f"Plan version {certificate.plan_version} not found in session.")

        version_obj = self.session.versions[certificate.plan_version]
        plan_ir = version_obj.plan_ir

        if certificate.plan_hash != plan_ir.compute_hash():
            raise StateDriftError(
                f"Plan identity drift detected: certificate hash '{certificate.plan_hash}' does not match PlanIR canonical hash '{plan_ir.compute_hash()}'."
            )

        # 3. Initialize live world state if not pre-populated
        if not self.live_world_state and plan_ir.initial_state:
            self.live_world_state = normalize_trusted_snapshot(plan_ir.initial_state)

        step_results: List[StepExecutionResult] = []

        for action in plan_ir.actions:
            step_start = time.time()
            step_id = action.action_id
            cap = self.registry.get(action.capability_name)

            # 4. Strict Live Empirical Precondition Check
            pre_passed = True
            pre_failure_reason = ""
            for pre in action.preconditions:
                key = pre.fact_key
                fact = self.live_world_state.get(key)
                if fact is None or fact.truth != pre.expected_truth:
                    pre_passed = False
                    current_truth = fact.truth.value if fact else "MISSING"
                    pre_failure_reason = f"Precondition failed on '{step_id}': expected '{key}' == {pre.expected_truth.value}, empirical truth is {current_truth}"
                    break

            self.ledger.append_record(
                LedgerEventType.PRECHECK_EVALUATED,
                {
                    "step_id": step_id,
                    "passed": pre_passed,
                    "reason": pre_failure_reason if not pre_passed else "OK",
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
                    error_message=pre_failure_reason,
                )
                step_results.append(res)
                self.ledger.append_record(
                    LedgerEventType.PLAN_ABORTED,
                    {"failed_step_id": step_id, "reason": pre_failure_reason},
                )
                total_duration = (time.time() - start_time) * 1000.0
                return ExecutionSummary(
                    plan_id=plan_ir.plan_id,
                    plan_version=plan_ir.version,
                    success=False,
                    step_results=step_results,
                    total_duration_ms=round(total_duration, 2),
                    failed_step_id=step_id,
                    live_world_state=copy.deepcopy(self.live_world_state),
                )

            # 5. Execute Capability Action
            if custom_action_handler is not None:
                exec_res = custom_action_handler(action, action.parameters)
            else:
                if not cap.executor_command_template:
                    duration = (time.time() - step_start) * 1000.0
                    err_msg = f"Action '{step_id}' cannot execute: capability '{cap.name}' has no concrete executor contract."
                    res = StepExecutionResult(
                        step_id=step_id,
                        capability_name=action.capability_name,
                        exit_code=127,
                        witness_status=WitnessStatus.UNWITNESSED,
                        duration_ms=round(duration, 2),
                        error_message=err_msg,
                    )
                    step_results.append(res)
                    self.ledger.append_record(
                        LedgerEventType.PLAN_ABORTED,
                        {"failed_step_id": step_id, "reason": err_msg},
                    )
                    total_duration = (time.time() - start_time) * 1000.0
                    return ExecutionSummary(
                        plan_id=plan_ir.plan_id,
                        plan_version=plan_ir.version,
                        success=False,
                        step_results=step_results,
                        total_duration_ms=round(total_duration, 2),
                        failed_step_id=step_id,
                        live_world_state=copy.deepcopy(self.live_world_state),
                    )

                cmd = _resolve_template_tokens(cap.executor_command_template, action.parameters)
                exec_res = self.sandbox.execute_argv_pipeline([cmd], timeout_seconds=action.timeout_seconds)

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
                err_msg = exec_res.stderr or f"Process execution failed with exit code {exec_res.returncode}"
                res = StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=exec_res.returncode,
                    witness_status=WitnessStatus.WITNESSED_FALSE,
                    duration_ms=round(duration, 2),
                    error_message=err_msg,
                )
                step_results.append(res)
                self.ledger.append_record(
                    LedgerEventType.PLAN_ABORTED,
                    {"failed_step_id": step_id, "reason": err_msg},
                )
                total_duration = (time.time() - start_time) * 1000.0
                return ExecutionSummary(
                    plan_id=plan_ir.plan_id,
                    plan_version=plan_ir.version,
                    success=False,
                    step_results=step_results,
                    total_duration_ms=round(total_duration, 2),
                    failed_step_id=step_id,
                    live_world_state=copy.deepcopy(self.live_world_state),
                )

            # 6. Independent Empirical Postcondition Witnessing
            witness_status = self._witness_postconditions(action, cap)
            self.ledger.append_record(
                LedgerEventType.POSTCHECK_WITNESSED,
                {
                    "step_id": step_id,
                    "witness_status": witness_status.value,
                },
            )

            step_duration = (time.time() - step_start) * 1000.0

            if witness_status != WitnessStatus.WITNESSED_TRUE:
                err_msg = f"Observation verifier failed or missing for action '{step_id}'"
                res = StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=0,
                    witness_status=witness_status,
                    duration_ms=round(step_duration, 2),
                    error_message=err_msg,
                )
                step_results.append(res)
                self.ledger.append_record(
                    LedgerEventType.PLAN_ABORTED,
                    {"failed_step_id": step_id, "reason": err_msg},
                )
                total_duration = (time.time() - start_time) * 1000.0
                return ExecutionSummary(
                    plan_id=plan_ir.plan_id,
                    plan_version=plan_ir.version,
                    success=False,
                    step_results=step_results,
                    total_duration_ms=round(total_duration, 2),
                    failed_step_id=step_id,
                    live_world_state=copy.deepcopy(self.live_world_state),
                )

            # 7. Verifier Succeeded: Update Live World State to VERIFIED_TRUE / VERIFIED_FALSE
            now_ts = time.time()
            for pos in action.positive_effects:
                pos_key = pos.fact_key
                self.live_world_state[pos_key] = WorldFact(
                    predicate=pos.predicate,
                    args=pos.args,
                    truth=FactTruth.VERIFIED_TRUE,
                    projected_truth=ProjectedTruth.SUPPORTED_TRUE,
                    witnessability=WitnessabilityStatus.WITNESSABLE,
                    created_at=now_ts,
                    updated_at=now_ts,
                    provenance=Provenance(
                        source_type=SourceType.OBSERVED_WORLD_STATE,
                        source_id=f"verifier_{step_id}",
                        confidence=1.0,
                        rationale=f"Witnessed live post-execution of {step_id}",
                    ),
                    metadata={"observed_at": now_ts},
                )

            for neg in action.negative_effects:
                neg_key = neg.fact_key
                self.live_world_state[neg_key] = WorldFact(
                    predicate=neg.predicate,
                    args=neg.args,
                    truth=FactTruth.VERIFIED_FALSE,
                    projected_truth=ProjectedTruth.SUPPORTED_FALSE,
                    witnessability=WitnessabilityStatus.WITNESSABLE,
                    created_at=now_ts,
                    updated_at=now_ts,
                    provenance=Provenance(
                        source_type=SourceType.OBSERVED_WORLD_STATE,
                        source_id=f"verifier_{step_id}",
                        confidence=1.0,
                        rationale=f"Witnessed live post-execution deletion of {step_id}",
                    ),
                    metadata={"observed_at": now_ts},
                )

            step_results.append(
                StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=0,
                    witness_status=witness_status,
                    duration_ms=round(step_duration, 2),
                )
            )

        # 8. Commit Execution
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
            live_world_state=copy.deepcopy(self.live_world_state),
        )

    def _witness_postconditions(self, action: ActionIR, cap: CapabilityEntry) -> WitnessStatus:
        """Execute independent observation verifiers declared in capability contract."""
        has_effects = bool(action.positive_effects or action.negative_effects)
        if has_effects and not cap.verifiers:
            # Capability with effects but NO verifiers cannot witness true!
            return WitnessStatus.WITNESSED_FALSE

        if not cap.verifiers:
            return WitnessStatus.WITNESSED_TRUE

        for v in cap.verifiers:
            if not v.command_template:
                return WitnessStatus.WITNESSED_FALSE

            cmd = _resolve_template_tokens(v.command_template, action.parameters)
            res = self.sandbox.execute_argv_pipeline([cmd], timeout_seconds=v.timeout_seconds)
            if res.returncode != 0:
                return WitnessStatus.WITNESSED_FALSE

            if v.expected_output_pattern:
                if not re.search(v.expected_output_pattern, res.stdout):
                    return WitnessStatus.WITNESSED_FALSE

        return WitnessStatus.WITNESSED_TRUE
