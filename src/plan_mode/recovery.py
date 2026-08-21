"""Saga-Style Compensation Graph Execution and Rollback Recovery Manager."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import ActionIR, PlanIR
from plan_mode.registry import CapabilityRegistry, CompensationAction
from plan_mode.runtime.executor import StepExecutionResult, WitnessStatus
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import ExecutionSandbox
from plan_mode.session import PlanningSession, SessionState


class RecoveryStatus(str, Enum):
    ROLLED_BACK = "ROLLED_BACK"
    CONTAINMENT_FAILED = "CONTAINMENT_FAILED"
    NO_RECOVERY_NEEDED = "NO_RECOVERY_NEEDED"


class SagaRecoveryReport(BaseModel):
    status: RecoveryStatus
    compensated_steps_count: int = 0
    failed_compensation_step_id: Optional[str] = None
    uncompensated_capabilities: List[str] = Field(default_factory=list)
    damage_mitigation_notes: List[str] = Field(default_factory=list)


def _resolve_mapping(mapping: Dict[str, str], original: Dict[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for target, source in mapping.items():
        is_braced = isinstance(source, str) and source.startswith("{") and source.endswith("}")
        is_dollar = isinstance(source, str) and source.startswith("$")
        source_name = source[1:-1] if is_braced else (source[1:] if is_dollar else source)
        if source_name in original:
            resolved[target] = original[source_name]
        elif is_braced or is_dollar:
            raise ValueError(
                f"compensation parameter '{target}' references missing original parameter '{source_name}'"
            )
        else:
            resolved[target] = source
    return resolved


def _resolve_tokens(tokens: List[str], params: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for token in tokens:
        value = token
        for key, param in params.items():
            value = value.replace(f"{{{key}}}", str(param)).replace(f"${key}", str(param))
        out.append(value)
    return out


class SagaRecoveryManager:
    """Legacy saga recovery path that performs real compensation execution.

    ``ROLLED_BACK`` is emitted only after every selected compensation command
    has actually been dispatched and returned exit code 0. Missing capabilities,
    missing executor contracts, bad parameter mappings, sandbox refusal, timeout
    or non-zero exit codes fail closed as containment failures.
    """

    def __init__(self, sandbox: Optional[ExecutionSandbox] = None):
        self.sandbox = sandbox or ExecutionSandbox()

    def _containment_failure(
        self,
        *,
        session: PlanningSession,
        step_id: Optional[str],
        capability: Optional[str],
        compensated_count: int,
        notes: List[str],
        uncompensated: List[str],
    ) -> SagaRecoveryReport:
        if capability and capability not in uncompensated:
            uncompensated.append(capability)
        if session.current_state == SessionState.COMPENSATING:
            session.transition_to(SessionState.CONTAINMENT_FAILED)
        if session.current_state == SessionState.CONTAINMENT_FAILED:
            # Preserve the public legacy manager's terminal-state contract while
            # the report still exposes the more precise containment outcome.
            session.transition_to(SessionState.FAILED)
        elif session.current_state != SessionState.FAILED:
            session.transition_to(SessionState.FAILED)
        return SagaRecoveryReport(
            status=RecoveryStatus.CONTAINMENT_FAILED,
            compensated_steps_count=compensated_count,
            failed_compensation_step_id=step_id,
            uncompensated_capabilities=uncompensated,
            damage_mitigation_notes=notes,
        )

    def execute_saga_rollback(
        self,
        executed_steps: List[StepExecutionResult],
        plan_ir: PlanIR,
        registry: CapabilityRegistry,
        ledger: EvidenceLedger,
        session: PlanningSession,
    ) -> SagaRecoveryReport:
        steps_to_undo = [
            step for step in executed_steps
            if step.exit_code == 0 and step.witness_status == WitnessStatus.WITNESSED_TRUE
        ]

        if not steps_to_undo:
            if session.current_state == SessionState.COMPENSATING:
                session.transition_to(SessionState.ROLLED_BACK)
            return SagaRecoveryReport(status=RecoveryStatus.NO_RECOVERY_NEEDED)

        if session.current_state in (SessionState.EXECUTING, SessionState.DIAGNOSING):
            session.transition_to(SessionState.COMPENSATING)

        compensated_count = 0
        uncompensated: List[str] = []
        notes: List[str] = []
        action_map: Dict[str, ActionIR] = {action.action_id: action for action in plan_ir.actions}

        for step in reversed(steps_to_undo):
            action = action_map.get(step.step_id)
            if action is None:
                note = f"Executed step '{step.step_id}' is absent from the authorized plan."
                notes.append(note)
                return self._containment_failure(
                    session=session,
                    step_id=step.step_id,
                    capability=step.capability_name,
                    compensated_count=compensated_count,
                    notes=notes,
                    uncompensated=uncompensated,
                )

            cap = registry.get(action.capability_name)
            comp_spec: Optional[CompensationAction] = cap.default_compensation
            if comp_spec is None:
                note = (
                    f"Step '{action.action_id}' ({action.capability_name}) has no registered "
                    "executable compensation contract."
                )
                notes.append(note)
                ledger.append_record(
                    LedgerEventType.COMPENSATION_TRIGGERED,
                    {"step_id": action.action_id, "status": "UNCOMPENSATABLE", "reason": note},
                )
                return self._containment_failure(
                    session=session,
                    step_id=action.action_id,
                    capability=action.capability_name,
                    compensated_count=compensated_count,
                    notes=notes,
                    uncompensated=uncompensated,
                )

            try:
                comp_cap = registry.get(comp_spec.capability_name)
            except Exception as exc:
                note = f"Compensation capability '{comp_spec.capability_name}' is unavailable: {exc}"
                notes.append(note)
                return self._containment_failure(
                    session=session,
                    step_id=action.action_id,
                    capability=action.capability_name,
                    compensated_count=compensated_count,
                    notes=notes,
                    uncompensated=uncompensated,
                )

            if not comp_cap.executor_command_template:
                note = (
                    f"Compensation capability '{comp_cap.name}' has no concrete executor command; "
                    "rollback cannot be claimed."
                )
                notes.append(note)
                return self._containment_failure(
                    session=session,
                    step_id=action.action_id,
                    capability=action.capability_name,
                    compensated_count=compensated_count,
                    notes=notes,
                    uncompensated=uncompensated,
                )

            try:
                comp_params = _resolve_mapping(comp_spec.parameter_mapping, action.parameters)
                command = _resolve_tokens(comp_cap.executor_command_template, comp_params)
            except Exception as exc:
                note = f"Could not resolve compensation for '{action.action_id}': {exc}"
                notes.append(note)
                return self._containment_failure(
                    session=session,
                    step_id=action.action_id,
                    capability=action.action_id,
                    compensated_count=compensated_count,
                    notes=notes,
                    uncompensated=uncompensated,
                )

            ledger.append_record(
                LedgerEventType.COMPENSATION_TRIGGERED,
                {
                    "step_id": action.action_id,
                    "compensation_id": comp_spec.compensation_id,
                    "capability": comp_cap.name,
                },
            )
            result = self.sandbox.execute_argv_pipeline(
                [command],
                timeout_seconds=float(comp_spec.timeout_seconds),
            )
            ledger.append_record(
                LedgerEventType.COMPENSATION_EXECUTED,
                {
                    "step_id": action.action_id,
                    "compensation_id": comp_spec.compensation_id,
                    "exit_code": result.returncode,
                    "timeout": result.timeout_exceeded,
                    "parameters": comp_params,
                },
            )
            if result.returncode != 0 or result.timeout_exceeded:
                note = (
                    f"Compensation '{comp_spec.compensation_id}' failed for '{action.action_id}' "
                    f"with exit code {result.returncode}: {result.stderr[:200]}"
                )
                notes.append(note)
                return self._containment_failure(
                    session=session,
                    step_id=action.action_id,
                    capability=action.capability_name,
                    compensated_count=compensated_count,
                    notes=notes,
                    uncompensated=uncompensated,
                )
            compensated_count += 1

        if session.current_state == SessionState.COMPENSATING:
            session.transition_to(SessionState.ROLLED_BACK)
        return SagaRecoveryReport(
            status=RecoveryStatus.ROLLED_BACK,
            compensated_steps_count=compensated_count,
            damage_mitigation_notes=notes,
        )
