"""Phase 3 transactional execution: verified commit gating and saga compensation."""
from __future__ import annotations

import copy
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PredicateCondition,
    ProjectedTruth,
    Provenance,
    SourceType,
    WitnessabilityStatus,
    WorldFact,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, CompensationAction
from plan_mode.runtime.executor import (
    ExecutionBackend,
    ExecutionPlanManager,
    ExecutionSummary,
    WitnessStatus,
)
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult
from plan_mode.session import (
    AuthorizationCertificate,
    CommitGateError,
    PlanningSession,
    SessionState,
)


class TransactionOutcome(str, Enum):
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    CONTAINMENT_FAILED = "CONTAINMENT_FAILED"


class CompensationResult(BaseModel):
    original_step_id: str
    compensation_id: str
    capability_name: str
    executed: bool
    verified: bool
    returncode: Optional[int] = None
    error_message: Optional[str] = None


class TransactionSummary(BaseModel):
    outcome: TransactionOutcome
    execution: ExecutionSummary
    compensation_results: List[CompensationResult] = Field(default_factory=list)
    commit_blockers: List[str] = Field(default_factory=list)
    live_world_state: Dict[str, WorldFact] = Field(default_factory=dict)


def _resolve_tokens(tokens: List[str], params: Dict[str, Any]) -> List[str]:
    resolved: List[str] = []
    for token in tokens:
        item = token
        for key, value in params.items():
            item = item.replace(f"{{{key}}}", str(value))
            item = item.replace(f"${key}", str(value))
        resolved.append(item)
    return resolved


def _resolve_parameter_mapping(
    mapping: Dict[str, str], original_params: Dict[str, Any]
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for target, source in mapping.items():
        source_name = source
        if source.startswith("{") and source.endswith("}"):
            source_name = source[1:-1]
        elif source.startswith("$"):
            source_name = source[1:]
        if source_name in original_params:
            resolved[target] = original_params[source_name]
        else:
            resolved[target] = source
    return resolved


def _instantiate_condition(cond: PredicateCondition, params: Dict[str, Any]) -> PredicateCondition:
    args: List[Any] = []
    for arg in cond.args:
        if isinstance(arg, str) and arg.startswith("{") and arg.endswith("}"):
            args.append(params.get(arg[1:-1], arg))
        elif isinstance(arg, str) and arg.startswith("$"):
            args.append(params.get(arg[1:], arg))
        elif isinstance(arg, str) and arg in params:
            args.append(params[arg])
        else:
            args.append(arg)
    return PredicateCondition(
        predicate=cond.predicate,
        args=args,
        expected_truth=cond.expected_truth,
        active_until_action_id=cond.active_until_action_id,
    )


class TransactionalExecutionManager:
    """Executes an authorized plan and either commits it or verifies compensation."""

    def __init__(
        self,
        session: PlanningSession,
        registry: CapabilityRegistry,
        ledger: EvidenceLedger,
        observed_world_state: List[WorldFact] | Dict[str, WorldFact],
        policy_hash: str,
        sandbox: Optional[ExecutionSandbox] = None,
    ):
        self.session = session
        self.registry = registry
        self.ledger = ledger
        self.policy_hash = policy_hash
        self.sandbox = sandbox or ExecutionSandbox()
        self.executor = ExecutionPlanManager(
            session=session,
            registry=registry,
            ledger=ledger,
            sandbox=self.sandbox,
            observed_world_state=observed_world_state,
            policy_hash=policy_hash,
        )

    def execute_and_finalize(
        self,
        certificate: AuthorizationCertificate,
        execution_backend: Optional[ExecutionBackend] = None,
    ) -> TransactionSummary:
        execution = self.executor.execute_authorized_plan(
            certificate,
            execution_backend=execution_backend,
        )
        self.session.record_execution_result(
            execution.success,
            list(self.executor.live_world_state.values()),
        )

        if execution.success:
            try:
                self.session.commit_execution(
                    live_world_state=self.executor.live_world_state,
                )
                self.ledger.append_record(
                    LedgerEventType.PLAN_COMMITTED,
                    {
                        "plan_id": execution.plan_id,
                        "plan_version": execution.plan_version,
                        "world_state_hash": self.session.last_execution_world_state_hash,
                    },
                )
                return TransactionSummary(
                    outcome=TransactionOutcome.COMMITTED,
                    execution=execution,
                    live_world_state=copy.deepcopy(self.executor.live_world_state),
                )
            except CommitGateError as exc:
                blockers = list(exc.blockers)
                return self._compensate_or_contain(
                    execution,
                    execution_backend,
                    blockers,
                )

        return self._compensate_or_contain(
            execution,
            execution_backend,
            ["execution did not complete with independently witnessed success"],
        )

    def _executed_action_ids(self) -> List[str]:
        ids: List[str] = []
        for record in self.ledger.records:
            if record.event_type == LedgerEventType.ACTION_EXECUTED:
                step_id = record.payload.get("step_id")
                if isinstance(step_id, str):
                    ids.append(step_id)
        return ids

    def _compensate_or_contain(
        self,
        execution: ExecutionSummary,
        execution_backend: Optional[ExecutionBackend],
        blockers: List[str],
    ) -> TransactionSummary:
        assert self.session.authorized_version is not None
        plan = self.session.versions[self.session.authorized_version].plan_ir
        action_by_id = {action.action_id: action for action in plan.actions}
        executed_ids = self._executed_action_ids()
        effectful_executed = [
            step_id
            for step_id in executed_ids
            if step_id in action_by_id
            and (
                action_by_id[step_id].positive_effects
                or action_by_id[step_id].negative_effects
            )
        ]

        if not effectful_executed:
            if self.session.current_state == SessionState.EXECUTING:
                self.session.transition_to(SessionState.FAILED)
            return TransactionSummary(
                outcome=TransactionOutcome.FAILED,
                execution=execution,
                commit_blockers=blockers,
                live_world_state=copy.deepcopy(self.executor.live_world_state),
            )

        if self.session.current_state == SessionState.EXECUTING:
            self.session.transition_to(SessionState.COMPENSATING)

        results: List[CompensationResult] = []
        for step_id in reversed(effectful_executed):
            action = action_by_id[step_id]
            cap = self.registry.get(action.capability_name)
            spec = cap.default_compensation
            if spec is None:
                return self._containment_failed(
                    execution,
                    results,
                    blockers,
                    step_id,
                    "executed effectful capability has no registered compensation",
                )

            result = self._execute_compensation(
                action,
                cap,
                spec,
                execution_backend,
            )
            results.append(result)
            if not result.executed or not result.verified:
                return self._containment_failed(
                    execution,
                    results,
                    blockers,
                    step_id,
                    result.error_message or "compensation was not independently verified",
                )

        self.session.transition_to(SessionState.ROLLED_BACK)
        return TransactionSummary(
            outcome=TransactionOutcome.ROLLED_BACK,
            execution=execution,
            compensation_results=results,
            commit_blockers=blockers,
            live_world_state=copy.deepcopy(self.executor.live_world_state),
        )

    def _execute_compensation(
        self,
        original_action: ActionIR,
        original_capability: CapabilityEntry,
        spec: CompensationAction,
        execution_backend: Optional[ExecutionBackend],
    ) -> CompensationResult:
        try:
            comp_cap = self.registry.get(spec.capability_name)
        except Exception as exc:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=spec.capability_name,
                executed=False,
                verified=False,
                error_message=f"compensation capability unavailable: {exc}",
            )

        params = _resolve_parameter_mapping(spec.parameter_mapping, original_action.parameters)
        comp_action = ActionIR(
            action_id=f"compensate:{original_action.action_id}:{spec.compensation_id}",
            capability_name=comp_cap.name,
            parameters=params,
            preconditions=[_instantiate_condition(c, params) for c in comp_cap.preconditions],
            positive_effects=[_instantiate_condition(c, params) for c in comp_cap.positive_effects],
            negative_effects=[_instantiate_condition(c, params) for c in comp_cap.negative_effects],
            timeout_seconds=spec.timeout_seconds,
            provenance=Provenance(
                source_type=SourceType.CAPABILITY_REGISTRY,
                source_id=spec.compensation_id,
                rationale=f"Compensation for {original_action.action_id}",
            ),
        )

        try:
            self.registry.validate_action(comp_action)
        except Exception as exc:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=False,
                verified=False,
                error_message=f"compensation contract validation failed: {exc}",
            )

        if not comp_cap.executor_command_template:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=False,
                verified=False,
                error_message="compensation capability has no executor contract",
            )

        self.ledger.append_record(
            LedgerEventType.COMPENSATION_TRIGGERED,
            {
                "original_step_id": original_action.action_id,
                "compensation_id": spec.compensation_id,
                "capability": comp_cap.name,
            },
        )

        argv = _resolve_tokens(comp_cap.executor_command_template, params)
        if execution_backend is not None:
            exec_result = execution_backend(argv, timeout_seconds=spec.timeout_seconds)
        else:
            exec_result = self.sandbox.execute_argv_pipeline(
                [argv],
                timeout_seconds=spec.timeout_seconds,
            )

        self.ledger.append_record(
            LedgerEventType.COMPENSATION_EXECUTED,
            {
                "original_step_id": original_action.action_id,
                "compensation_id": spec.compensation_id,
                "capability": comp_cap.name,
                "returncode": exec_result.returncode,
            },
        )

        if exec_result.returncode != 0:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=True,
                verified=False,
                returncode=exec_result.returncode,
                error_message=exec_result.stderr or "compensation executor failed",
            )

        witness = self.executor.witness_postconditions(comp_action, comp_cap)
        verified = witness == WitnessStatus.WITNESSED_TRUE
        self.ledger.append_record(
            LedgerEventType.COMPENSATION_VERIFIED,
            {
                "original_step_id": original_action.action_id,
                "compensation_id": spec.compensation_id,
                "capability": comp_cap.name,
                "witness_status": witness.value,
            },
        )

        if verified:
            self._apply_compensation_effects(comp_action, spec.compensation_id)

        return CompensationResult(
            original_step_id=original_action.action_id,
            compensation_id=spec.compensation_id,
            capability_name=comp_cap.name,
            executed=True,
            verified=verified,
            returncode=exec_result.returncode,
            error_message=None if verified else f"compensation witness status was {witness.value}",
        )

    def _apply_compensation_effects(self, action: ActionIR, source_id: str) -> None:
        now = time.time()
        for effect in action.positive_effects:
            self.executor.live_world_state[effect.fact_key] = WorldFact(
                predicate=effect.predicate,
                args=effect.args,
                truth=FactTruth.VERIFIED_TRUE,
                projected_truth=ProjectedTruth.SUPPORTED_TRUE,
                witnessability=WitnessabilityStatus.WITNESSABLE,
                created_at=now,
                updated_at=now,
                provenance=Provenance(
                    source_type=SourceType.OBSERVED_WORLD_STATE,
                    source_id=source_id,
                    rationale="Independently witnessed compensation effect",
                ),
            )
        for effect in action.negative_effects:
            self.executor.live_world_state[effect.fact_key] = WorldFact(
                predicate=effect.predicate,
                args=effect.args,
                truth=FactTruth.VERIFIED_FALSE,
                projected_truth=ProjectedTruth.SUPPORTED_FALSE,
                witnessability=WitnessabilityStatus.WITNESSABLE,
                created_at=now,
                updated_at=now,
                provenance=Provenance(
                    source_type=SourceType.OBSERVED_WORLD_STATE,
                    source_id=source_id,
                    rationale="Independently witnessed compensation effect",
                ),
            )

    def _containment_failed(
        self,
        execution: ExecutionSummary,
        results: List[CompensationResult],
        blockers: List[str],
        step_id: str,
        reason: str,
    ) -> TransactionSummary:
        if self.session.current_state == SessionState.COMPENSATING:
            self.session.transition_to(SessionState.CONTAINMENT_FAILED)
        self.ledger.append_record(
            LedgerEventType.CONTAINMENT_FAILED,
            {"step_id": step_id, "reason": reason},
        )
        return TransactionSummary(
            outcome=TransactionOutcome.CONTAINMENT_FAILED,
            execution=execution,
            compensation_results=results,
            commit_blockers=blockers,
            live_world_state=copy.deepcopy(self.executor.live_world_state),
        )
