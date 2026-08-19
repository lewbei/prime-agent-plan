"""Execution Plan Manager with Precondition Checking, Process Execution, and Postcondition Witnessing."""

from __future__ import annotations

import copy
import json
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
    InvalidStateTransitionError,
    compute_world_state_hash,
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


def _matches_verifier_runtime(
    cond: PredicateCondition,
    v: ObservationVerifier,
    params: Dict[str, Any],
) -> bool:
    """Check if an ObservationVerifier strictly matches the target predicate and bound arguments."""
    if v.predicate != cond.predicate:
        return False

    if cond.args:
        if not v.target_args_mapping:
            return False
        resolved_args: List[Any] = []
        for arg in v.target_args_mapping:
            if isinstance(arg, str):
                if arg.startswith("{") and arg.endswith("}"):
                    var_name = arg[1:-1]
                    resolved_args.append(params.get(var_name, arg))
                elif arg.startswith("$"):
                    var_name = arg[1:]
                    resolved_args.append(params.get(var_name, arg))
                elif arg in params:
                    resolved_args.append(params[arg])
                else:
                    resolved_args.append(arg)
            else:
                resolved_args.append(arg)
        return typed_args_equal(resolved_args, cond.args)
    else:
        if not v.target_args_mapping:
            return True
        resolved_args = []
        for arg in v.target_args_mapping:
            if isinstance(arg, str):
                if arg.startswith("{") and arg.endswith("}"):
                    var_name = arg[1:-1]
                    resolved_args.append(params.get(var_name, arg))
                elif arg.startswith("$"):
                    var_name = arg[1:]
                    resolved_args.append(params.get(var_name, arg))
                elif arg in params:
                    resolved_args.append(params[arg])
                else:
                    resolved_args.append(arg)
            else:
                resolved_args.append(arg)
        return typed_args_equal(resolved_args, [])


class ExecutionPlanManager:
    """Coordinates safe plan execution, ledger recording, and empirical verification."""

    def __init__(
        self,
        session: PlanningSession,
        registry: CapabilityRegistry,
        ledger: EvidenceLedger,
        sandbox: Optional[ExecutionSandbox] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        policy_hash: Optional[str] = None,
    ):
        self.session = session
        self.registry = registry
        self.ledger = ledger
        self.sandbox = sandbox or ExecutionSandbox()
        self.policy_hash = policy_hash
        self._has_trusted_snapshot = (observed_world_state is not None)
        self.live_world_state: Dict[str, WorldFact] = normalize_trusted_snapshot(observed_world_state) if self._has_trusted_snapshot else {}

    def execute_authorized_plan(
        self,
        certificate: AuthorizationCertificate,
        custom_action_handler: Optional[Callable[[ActionIR, Dict[str, Any]], SandboxExecutionResult]] = None,
    ) -> ExecutionSummary:
        start_time = time.time()

        # -------------------------------------------------------------------
        # Central Runtime Preflight Verification (Fail-Closed)
        # -------------------------------------------------------------------
        # A. Session State Preflight
        if self.session.current_state != SessionState.EXECUTING:
            raise InvalidStateTransitionError(
                f"Cannot execute plan in state '{self.session.current_state.value}'. Expected EXECUTING (session.start_execution() must be called first)."
            )

        # B. Certificate Signature & Expiration
        if not certificate.verify_signature(self.session.secret_key):
            raise SignatureVerificationError("Authorization certificate signature invalid or tampered.")
        if certificate.is_expired():
            raise ValueError("Authorization certificate has expired.")

        # C. Certificate must match session's active authorized certificate
        if (
            self.session.authorization_certificate is None
            or certificate.certificate_id != self.session.authorization_certificate.certificate_id
            or certificate.signature_hmac != self.session.authorization_certificate.signature_hmac
        ):
            raise SignatureVerificationError("Authorization certificate does not match active session authorized certificate.")

        # D. Plan Version Existence and Hash Identity
        if certificate.plan_version not in self.session.versions:
            raise ValueError(f"Plan version {certificate.plan_version} not found in session.")

        version_obj = self.session.versions[certificate.plan_version]
        plan_ir = version_obj.plan_ir

        if certificate.plan_hash != plan_ir.compute_hash():
            raise StateDriftError(
                f"Plan identity drift detected: certificate hash '{certificate.plan_hash}' != PlanIR canonical hash '{plan_ir.compute_hash()}'."
            )

        # E. Registry Hash Preflight Check
        current_reg_hash = self.registry.compute_registry_hash()
        if certificate.registry_hash and certificate.registry_hash != current_reg_hash:
            raise StateDriftError(
                f"Registry drift detected: certificate registry hash '{certificate.registry_hash}' != current registry hash '{current_reg_hash}'."
            )

        # F. Policy Hash Preflight Check
        if self.policy_hash is None:
            raise StateDriftError("Runtime requires explicit current policy_hash identity before executing.")

        if certificate.policy_hash != self.policy_hash:
            raise StateDriftError(
                f"Policy drift detected: certificate policy hash '{certificate.policy_hash}' != runtime policy '{self.policy_hash}'."
            )

        if self.session.authorized_policy_hash and self.session.authorized_policy_hash != self.policy_hash:
            raise StateDriftError(
                f"Policy drift detected: session policy hash '{self.session.authorized_policy_hash}' != runtime policy '{self.policy_hash}'."
            )

        # G. Live World State Hash Check (Distinguish None from empty list)
        if not self._has_trusted_snapshot:
            raise StateDriftError(
                "No trusted live world state snapshot was supplied to runtime (observed_world_state is None)."
            )

        current_live_ws_hash = compute_world_state_hash(list(self.live_world_state.values()))
        if current_live_ws_hash != certificate.world_state_hash:
            raise StateDriftError(
                f"Live world state drift detected before execution: authorized hash '{certificate.world_state_hash}' != live hash '{current_live_ws_hash}'."
            )

        step_results: List[StepExecutionResult] = []

        # -------------------------------------------------------------------
        # Step-by-Step Execution Loop
        # -------------------------------------------------------------------
        for action in plan_ir.actions:
            step_start = time.time()
            step_id = action.action_id

            # 1. Re-validate action against current registry
            try:
                self.registry.validate_action(action)
                cap = self.registry.get(action.capability_name)
            except Exception as e:
                err_msg = f"Action '{step_id}' failed registry contract validation: {str(e)}"
                duration = (time.time() - step_start) * 1000.0
                res = StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=1,
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

            # 2. Strict Live Empirical Precondition Check
            pre_passed = True
            pre_failure_reason = ""
            for pre in action.preconditions:
                key = pre.fact_key
                fact = self.live_world_state.get(key)
                if fact is None or fact.truth != pre.expected_truth:
                    pre_passed = False
                    current_truth = fact.truth.value if fact else "MISSING"
                    pre_failure_reason = f"Precondition failed on '{step_id}': expected '{key}' == {pre.expected_truth.value}, empirical live truth is {current_truth}"
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
                    witness_status=WitnessStatus.UNWITNESSED,
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

            # 3. Check Executor Contract
            if not cap.executor_command_template:
                err_msg = f"Action '{step_id}' cannot execute: capability '{cap.name}' has no concrete executor contract."
                duration = (time.time() - step_start) * 1000.0
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

            # 4. Execute Capability Action
            cmd = _resolve_template_tokens(cap.executor_command_template, action.parameters)
            if custom_action_handler is not None:
                exec_res = custom_action_handler(cmd, timeout_seconds=action.timeout_seconds)
            else:
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

            # 5. Independent Per-Effect Postcondition Witnessing
            has_effects = bool(action.positive_effects or action.negative_effects)
            step_duration = (time.time() - step_start) * 1000.0
            now_ts = time.time()

            if not has_effects:
                step_results.append(
                    StepExecutionResult(
                        step_id=step_id,
                        capability_name=action.capability_name,
                        exit_code=0,
                        witness_status=WitnessStatus.WITNESSED_TRUE,
                        duration_ms=round(step_duration, 2),
                    )
                )
                continue

            # Verify and witness each positive effect individually
            all_effects_witnessed = True
            witness_failure_reason = ""
            any_verifier_ran = False

            for pos in action.positive_effects:
                pos_key = pos.fact_key
                matching_verifiers = [
                    v for v in cap.verifiers
                    if _matches_verifier_runtime(pos, v, action.parameters)
                ]

                if not matching_verifiers:
                    all_effects_witnessed = False
                    witness_failure_reason = f"No matching observation verifier bound for positive effect '{pos_key}'"
                    break

                # Run each matching verifier
                pos_effect_verified = False
                for v in matching_verifiers:
                    any_verifier_ran = True
                    v_res, v_err = self._run_verifier(v, action.parameters)
                    if v_res:
                        pos_effect_verified = True
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
                                source_id=v.verifier_id,
                                confidence=1.0,
                                rationale=f"Witnessed live post-execution of {step_id}",
                            ),
                            metadata={"observed_at": now_ts},
                        )
                        break
                    else:
                        witness_failure_reason = v_err

                if not pos_effect_verified:
                    all_effects_witnessed = False
                    break

            if all_effects_witnessed:
                # Verify and witness each negative effect individually
                for neg in action.negative_effects:
                    neg_key = neg.fact_key
                    matching_verifiers = [
                        v for v in cap.verifiers
                        if _matches_verifier_runtime(neg, v, action.parameters)
                    ]

                    if not matching_verifiers:
                        all_effects_witnessed = False
                        witness_failure_reason = f"No matching observation verifier bound for negative effect '{neg_key}'"
                        break

                    neg_effect_verified = False
                    for v in matching_verifiers:
                        any_verifier_ran = True
                        v_res, v_err = self._run_verifier(v, action.parameters)
                        if v_res:
                            neg_effect_verified = True
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
                                    source_id=v.verifier_id,
                                    confidence=1.0,
                                    rationale=f"Witnessed live post-execution deletion of {step_id}",
                                ),
                                metadata={"observed_at": now_ts},
                            )
                            break
                        else:
                            witness_failure_reason = v_err

                    if not neg_effect_verified:
                        all_effects_witnessed = False
                        break

            if all_effects_witnessed:
                witness_status = WitnessStatus.WITNESSED_TRUE
            elif any_verifier_ran:
                witness_status = WitnessStatus.WITNESSED_FALSE
            else:
                witness_status = WitnessStatus.UNWITNESSED

            self.ledger.append_record(
                LedgerEventType.POSTCHECK_WITNESSED,
                {
                    "step_id": step_id,
                    "witness_status": witness_status.value,
                    "reason": "All effects witnessed" if all_effects_witnessed else witness_failure_reason,
                },
            )

            if not all_effects_witnessed:
                err_msg = witness_failure_reason or f"Observation verifier failed for action '{step_id}'"
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

            step_results.append(
                StepExecutionResult(
                    step_id=step_id,
                    capability_name=action.capability_name,
                    exit_code=0,
                    witness_status=WitnessStatus.WITNESSED_TRUE,
                    duration_ms=round(step_duration, 2),
                )
            )

        # 6. Commit Execution
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

    def _run_verifier(self, v: ObservationVerifier, params: Dict[str, Any]) -> tuple[bool, str]:
        """Execute a single observation verifier with command execution, pattern match, and JSON path check."""
        if not v.command_template:
            return False, f"Verifier '{v.verifier_id}' missing command template"

        cmd = _resolve_template_tokens(v.command_template, params)
        res = self.sandbox.execute_argv_pipeline([cmd], timeout_seconds=v.timeout_seconds)
        if res.returncode != 0:
            return False, f"Verifier command '{cmd}' exited with code {res.returncode}"

        if v.expected_output_pattern:
            if not re.search(v.expected_output_pattern, res.stdout):
                return False, f"Verifier output did not match pattern '{v.expected_output_pattern}'"

        if v.json_path:
            try:
                parsed_json = json.loads(res.stdout)
                actual_val = parsed_json
                for key in v.json_path.split("."):
                    if isinstance(actual_val, dict) and key in actual_val:
                        actual_val = actual_val[key]
                    else:
                        return False, f"JSON path '{v.json_path}' key '{key}' not found in output"

                if v.expected_value is not None:
                    if str(actual_val) != str(v.expected_value) and actual_val != v.expected_value:
                        return False, f"JSON path '{v.json_path}' value '{actual_val}' != expected '{v.expected_value}'"
            except Exception as e:
                return False, f"JSON parsing failed for verifier '{v.verifier_id}': {str(e)}"

        return True, ""
    def _witness_postconditions(self, action: ActionIR, cap: CapabilityEntry) -> WitnessStatus:
        """Evaluate postcondition verifiers and return WitnessStatus."""
        has_effects = bool(action.positive_effects or action.negative_effects)
        if not has_effects:
            return WitnessStatus.WITNESSED_TRUE

        if not cap.verifiers:
            return WitnessStatus.UNWITNESSED

        any_ran = False
        all_passed = True

        for pos in action.positive_effects:
            matching = [v for v in cap.verifiers if _matches_verifier_runtime(pos, v, action.parameters)]
            if not matching:
                return WitnessStatus.UNWITNESSED
            pos_ok = False
            for v in matching:
                any_ran = True
                ok, _ = self._run_verifier(v, action.parameters)
                if ok:
                    pos_ok = True
                    break
            if not pos_ok:
                all_passed = False
                break

        for neg in action.negative_effects:
            matching = [v for v in cap.verifiers if _matches_verifier_runtime(neg, v, action.parameters)]
            if not matching:
                return WitnessStatus.UNWITNESSED
            neg_ok = False
            for v in matching:
                any_ran = True
                ok, _ = self._run_verifier(v, action.parameters)
                if ok:
                    neg_ok = True
                    break
            if not neg_ok:
                all_passed = False
                break

        if all_passed:
            return WitnessStatus.WITNESSED_TRUE
        elif any_ran:
            return WitnessStatus.WITNESSED_FALSE
        else:
            return WitnessStatus.UNWITNESSED

    def witness_postconditions(self, action: ActionIR, cap: CapabilityEntry) -> WitnessStatus:
        """Public interface to evaluate postcondition verifiers."""
        return self._witness_postconditions(action, cap)
