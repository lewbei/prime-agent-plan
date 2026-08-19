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
from plan_mode.registry import CapabilityRegistry, CompensationAction
from plan_mode.runtime.executor import (
    ExecutionBackend,
    ExecutionPlanManager,
    ExecutionSummary,
    WitnessStatus,
)
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType
from plan_mode.runtime.sandbox import ExecutionSandbox
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


def _resolve_parameter_mapping(mapping: Dict[str, str], original_params: Dict[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for target, source in mapping.items():
        is_braced = source.startswith("{") and source.endswith("}")
        is_dollar = source.startswith("$")
        source_name = source[1:-1] if is_braced else (source[1:] if is_dollar else source)
        if source_name in original_params:
            resolved[target] = original_params[source_name]
        elif is_braced or is_dollar:
            raise ValueError(
                f"compensation parameter '{target}' references missing original parameter '{source_name}'"
            )
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
        plan = self._plan_for_certificate(certificate)
        self._assert_unique_action_ids(plan.actions)
        dispatch_index = 0

        def tracking_backend(argv: List[str], *, timeout_seconds: float):
            nonlocal dispatch_index
            if dispatch_index >= len(plan.actions):
                raise RuntimeError("execution backend invoked more times than authorized plan actions")
            action = plan.actions[dispatch_index]
            dispatch_index += 1
            self.ledger.append_record(
                LedgerEventType.ACTION_DISPATCHED,
                {
                    "step_id": action.action_id,
                    "capability": action.capability_name,
                },
            )
            if execution_backend is not None:
                return execution_backend(argv, timeout_seconds=timeout_seconds)
            return self.sandbox.execute_argv_pipeline([argv], timeout_seconds=timeout_seconds)

        try:
            execution = self.executor.execute_authorized_plan(
                certificate,
                execution_backend=tracking_backend,
            )
        except Exception as exc:
            dispatched = self._dispatched_action_ids()
            execution = ExecutionSummary(
                plan_id=plan.plan_id,
                plan_version=plan.version,
                success=False,
                failed_step_id=dispatched[-1] if dispatched else None,
                live_world_state=copy.deepcopy(self.executor.live_world_state),
            )
            if self.session.current_state == SessionState.EXECUTING:
                self.session.record_execution_result(False, list(self.executor.live_world_state.values()))
            return self._compensate_or_contain(
                certificate,
                execution,
                execution_backend,
                [f"execution raised after dispatch: {type(exc).__name__}: {exc}"],
            )

        try:
            self.session.record_execution_result(
                execution.success,
                list(self.executor.live_world_state.values()),
            )
        except Exception as exc:
            return self._compensate_or_contain(
                certificate,
                execution,
                execution_backend,
                [f"could not bind execution attestation: {type(exc).__name__}: {exc}"],
            )

        if execution.success:
            try:
                self.session.commit_execution(live_world_state=self.executor.live_world_state)
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
            except Exception as exc:
                blockers = [f"commit finalization error: {type(exc).__name__}: {exc}"]
            return self._compensate_or_contain(certificate, execution, execution_backend, blockers)

        return self._compensate_or_contain(
            certificate,
            execution,
            execution_backend,
            ["execution did not complete with independently witnessed success"],
        )

    def _plan_for_certificate(self, certificate: AuthorizationCertificate):
        if certificate.plan_version not in self.session.versions:
            raise ValueError(f"Plan version {certificate.plan_version} not found in session")
        return self.session.versions[certificate.plan_version].plan_ir

    @staticmethod
    def _assert_unique_action_ids(actions: List[ActionIR]) -> None:
        seen = set()
        duplicates = set()
        for action in actions:
            if action.action_id in seen:
                duplicates.add(action.action_id)
            seen.add(action.action_id)
        if duplicates:
            rendered = ", ".join(sorted(duplicates))
            raise ValueError(f"transaction requires unique action_id values; duplicates: {rendered}")

    def _dispatched_action_ids(self) -> List[str]:
        ids: List[str] = []
        for record in self.ledger.records:
            if record.event_type == LedgerEventType.ACTION_DISPATCHED:
                step_id = record.payload.get("step_id")
                if isinstance(step_id, str):
                    ids.append(step_id)
        return ids

    def _compensate_or_contain(
        self,
        certificate: AuthorizationCertificate,
        execution: ExecutionSummary,
        execution_backend: Optional[ExecutionBackend],
        blockers: List[str],
    ) -> TransactionSummary:
        assert self.session.authorized_version is not None
        plan = self.session.versions[self.session.authorized_version].plan_ir
        action_by_id = {action.action_id: action for action in plan.actions}
        effectful_dispatched = [
            step_id
            for step_id in self._dispatched_action_ids()
            if step_id in action_by_id
            and (action_by_id[step_id].positive_effects or action_by_id[step_id].negative_effects)
        ]

        if not effectful_dispatched:
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

        if self.registry.compute_registry_hash() != certificate.registry_hash:
            return self._containment_failed(
                execution, [], blockers, effectful_dispatched[-1], "capability registry drifted before compensation"
            )
        if self.policy_hash != certificate.policy_hash:
            return self._containment_failed(
                execution, [], blockers, effectful_dispatched[-1], "runtime policy drifted before compensation"
            )

        results: List[CompensationResult] = []
        for step_id in reversed(effectful_dispatched):
            action = action_by_id[step_id]
            cap = self.registry.get(action.capability_name)
            spec = cap.default_compensation
            if spec is None:
                return self._containment_failed(
                    execution,
                    results,
                    blockers,
                    step_id,
                    "dispatched effectful capability has no registered compensation",
                )
            result = self._execute_compensation(action, spec, execution_backend)
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

        try:
            params = _resolve_parameter_mapping(spec.parameter_mapping, original_action.parameters)
        except ValueError as exc:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=False,
                verified=False,
                error_message=str(exc),
            )

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
        if not (comp_action.positive_effects or comp_action.negative_effects):
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=False,
                verified=False,
                error_message="compensation capability declares no observable postcondition effects",
            )

        for pre in comp_action.preconditions:
            fact = self.executor.live_world_state.get(pre.fact_key)
            if fact is None or fact.truth != pre.expected_truth:
                observed = fact.truth.value if fact is not None else "MISSING"
                return CompensationResult(
                    original_step_id=original_action.action_id,
                    compensation_id=spec.compensation_id,
                    capability_name=comp_cap.name,
                    executed=False,
                    verified=False,
                    error_message=(
                        f"compensation precondition '{pre.fact_key}' expected {pre.expected_truth.value}, observed {observed}"
                    ),
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
        try:
            if execution_backend is not None:
                exec_result = execution_backend(argv, timeout_seconds=spec.timeout_seconds)
            else:
                exec_result = self.sandbox.execute_argv_pipeline([argv], timeout_seconds=spec.timeout_seconds)
        except Exception as exc:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=True,
                verified=False,
                error_message=f"compensation backend raised: {type(exc).__name__}: {exc}",
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

        try:
            witness = self.executor.witness_postconditions(comp_action, comp_cap)
        except Exception as exc:
            return CompensationResult(
                original_step_id=original_action.action_id,
                compensation_id=spec.compensation_id,
                capability_name=comp_cap.name,
                executed=True,
                verified=False,
                returncode=exec_result.returncode,
                error_message=f"compensation verifier raised: {type(exc).__name__}: {exc}",
            )

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
