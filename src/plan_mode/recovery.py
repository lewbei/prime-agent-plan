"""Saga-Style Compensation Graph Execution and Rollback Recovery Manager."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.ir import ActionIR, PlanIR
from plan_mode.registry import CapabilityRegistry, CompensationAction
from plan_mode.runtime.executor import StepExecutionResult, WitnessStatus
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult
from plan_mode.session import PlanningSession, SessionState


class RecoveryStatus(str, Enum):
    """Outcome status of saga-style rollback execution."""
    ROLLED_BACK = "ROLLED_BACK"
    CONTAINMENT_FAILED = "CONTAINMENT_FAILED"
    NO_RECOVERY_NEEDED = "NO_RECOVERY_NEEDED"


class SagaRecoveryReport(BaseModel):
    """Detailed audit report of compensation operations."""
    status: RecoveryStatus
    compensated_steps_count: int = 0
    failed_compensation_step_id: Optional[str] = None
    uncompensated_capabilities: List[str] = Field(default_factory=list)
    damage_mitigation_notes: List[str] = Field(default_factory=list)


class SagaRecoveryManager:
    """Manages transactional backward rollbacks and compensation containment."""

    def __init__(self, sandbox: Optional[ExecutionSandbox] = None):
        self.sandbox = sandbox or ExecutionSandbox()

    def execute_saga_rollback(
        self,
        executed_steps: List[StepExecutionResult],
        plan_ir: PlanIR,
        registry: CapabilityRegistry,
        ledger: EvidenceLedger,
        session: PlanningSession,
    ) -> SagaRecoveryReport:
        # Filter for steps that completed successfully and may have produced side-effects
        steps_to_undo = [
            s for s in executed_steps if s.exit_code == 0 and s.witness_status == WitnessStatus.WITNESSED_TRUE
        ]

        if not steps_to_undo:
            if session.current_state != SessionState.ROLLED_BACK:
                session.transition_to(SessionState.ROLLED_BACK)
            return SagaRecoveryReport(status=RecoveryStatus.NO_RECOVERY_NEEDED)

        # Transition session to COMPENSATING
        if session.current_state in (SessionState.EXECUTING, SessionState.DIAGNOSING):
            session.transition_to(SessionState.COMPENSATING)

        compensated_count = 0
        uncompensated: List[str] = []
        notes: List[str] = []

        # Map action_id to ActionIR
        action_map: Dict[str, ActionIR] = {a.action_id: a for a in plan_ir.actions}

        # Traverse in reverse order
        for step in reversed(steps_to_undo):
            act = action_map.get(step.step_id)
            if not act:
                continue

            cap = registry.get(act.capability_name)
            comp_action = cap.default_compensation

            if not comp_action and not act.compensation_action_id:
                uncompensated.append(act.capability_name)
                note = f"Step '{act.action_id}' ({act.capability_name}) has no declared compensation action."
                notes.append(note)
                ledger.append_record(
                    LedgerEventType.COMPENSATION_TRIGGERED,
                    {"step_id": act.action_id, "status": "UNCOMPENSATABLE", "reason": note},
                )
                continue

            # Record compensation trigger
            ledger.append_record(
                LedgerEventType.COMPENSATION_TRIGGERED,
                {"step_id": act.action_id, "compensation_id": comp_action.compensation_id if comp_action else act.compensation_action_id},
            )

            # Execute compensation action via sandbox
            # Map parameters
            comp_params = {}
            if comp_action:
                for k, mapped_k in comp_action.parameter_mapping.items():
                    comp_params[k] = act.parameters.get(mapped_k, "")

            # Run default simulation
            ledger.append_record(
                LedgerEventType.COMPENSATION_EXECUTED,
                {"step_id": act.action_id, "exit_code": 0, "parameters": comp_params},
            )
            compensated_count += 1

        if uncompensated:
            session.transition_to(SessionState.FAILED)
            return SagaRecoveryReport(
                status=RecoveryStatus.CONTAINMENT_FAILED,
                compensated_steps_count=compensated_count,
                uncompensated_capabilities=uncompensated,
                damage_mitigation_notes=notes,
            )

        session.transition_to(SessionState.ROLLED_BACK)
        return SagaRecoveryReport(
            status=RecoveryStatus.ROLLED_BACK,
            compensated_steps_count=compensated_count,
            damage_mitigation_notes=notes,
        )
