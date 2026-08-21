"""Transaction-bound execution, workspace retention, and ledger scoping."""
from __future__ import annotations

import copy
import os
import secrets
from pathlib import Path
from typing import Any

from .runtime_closure_context import (
    ACTIVE_TRANSACTION_ID,
    ACTIVE_WORKSPACE_ID,
    ACTIVE_WORKSPACE_PATH,
    CERTIFICATE_WORKSPACES,
    CERTIFICATE_WORKSPACES_LOCK,
    workspace_identity,
)


def install_execution_closure() -> None:
    from .runtime import executor as executor_mod
    from .runtime import ledger as ledger_mod
    from .runtime import sandbox as sandbox_mod
    from .runtime import transaction as transaction_mod
    from .session import CommitGateError, PlanningSession, StateDriftError

    unmanaged_test_env = "PLAN_ALLOW_UNMANAGED_TEST_EXECUTION"

    EvidenceLedger = ledger_mod.EvidenceLedger
    TransactionalExecutionManager = transaction_mod.TransactionalExecutionManager
    ExecutionPlanManager = executor_mod.ExecutionPlanManager

    raw_append = EvidenceLedger.append_record
    if not getattr(raw_append, "_runtime_closure", False):
        def append_record(self: Any, event_type: Any, payload: dict[str, Any], timestamp: float | None = None):
            enriched = copy.deepcopy(payload)
            transaction_id = ACTIVE_TRANSACTION_ID.get()
            workspace_id = ACTIVE_WORKSPACE_ID.get()
            if transaction_id:
                enriched.setdefault("transaction_id", transaction_id)
            if workspace_id:
                enriched.setdefault("workspace_identity", workspace_id)
            return raw_append(self, event_type, enriched, timestamp=timestamp)

        append_record._runtime_closure = True  # type: ignore[attr-defined]
        EvidenceLedger.append_record = append_record  # type: ignore[assignment]

    raw_sandbox_execute = sandbox_mod.ExecutionSandbox.execute_argv_pipeline
    if not getattr(raw_sandbox_execute, "_runtime_closure", False):
        def sandbox_execute(self: Any, pipeline: Any, cwd: Any = None, env: Any = None,
                            timeout_seconds: float = 10.0, input_data: str | None = None):
            expected = ACTIVE_WORKSPACE_ID.get()
            expected_path = ACTIVE_WORKSPACE_PATH.get()
            if expected and expected_path:
                current_policy_path = self.policy.workspace_dir
                if current_policy_path and workspace_identity(current_policy_path) != expected:
                    return sandbox_mod.SandboxExecutionResult(
                        stderr="Security violation: transaction workspace identity changed after binding.",
                        returncode=126,
                    )
                effective = str(Path(cwd or current_policy_path or os.getcwd()).resolve(strict=False))
                if effective != str(Path(expected_path).resolve(strict=False)):
                    return sandbox_mod.SandboxExecutionResult(
                        stderr="Security violation: command cwd differs from transaction workspace.",
                        returncode=126,
                    )
            return raw_sandbox_execute(
                self,
                pipeline,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
                input_data=input_data,
            )

        sandbox_execute._runtime_closure = True  # type: ignore[attr-defined]
        sandbox_mod.ExecutionSandbox.execute_argv_pipeline = sandbox_execute  # type: ignore[assignment]

    raw_execute_authorized = ExecutionPlanManager.execute_authorized_plan
    if not getattr(raw_execute_authorized, "_runtime_closure", False):
        def execute_authorized_plan(
            self: Any,
            certificate: Any,
            execution_backend: Any = None,
            custom_action_handler: Any = None,
        ):
            unmanaged_test_opt_in = os.environ.get(unmanaged_test_env, "").strip().lower() in {
                "1", "true", "yes",
            }
            if ACTIVE_TRANSACTION_ID.get() is None and not unmanaged_test_opt_in:
                raise sandbox_mod.SandboxSecurityViolationError(
                    "Authorized execution must run inside TransactionalExecutionManager; "
                    f"the explicit {unmanaged_test_env}=1 escape hatch is test-only."
                )
            return raw_execute_authorized(
                self,
                certificate,
                execution_backend=execution_backend,
                custom_action_handler=custom_action_handler,
            )

        execute_authorized_plan._runtime_closure = True  # type: ignore[attr-defined]
        ExecutionPlanManager.execute_authorized_plan = execute_authorized_plan  # type: ignore[assignment]

    raw_commit = PlanningSession.commit_execution
    if not getattr(raw_commit, "_runtime_closure", False):
        def commit_execution(self: Any, live_world_state: Any = None) -> None:
            transaction_id = ACTIVE_TRANSACTION_ID.get()
            if not transaction_id:
                raise CommitGateError([
                    "commit must be finalized inside TransactionalExecutionManager"
                ])
            expected_workspace = ACTIVE_WORKSPACE_ID.get()
            workspace_path = ACTIVE_WORKSPACE_PATH.get()
            if expected_workspace and workspace_path:
                if workspace_identity(workspace_path) != expected_workspace:
                    raise CommitGateError([
                        "transaction workspace identity changed before commit"
                    ])
            raw_commit(self, live_world_state=live_world_state)

        commit_execution._runtime_closure = True  # type: ignore[attr-defined]
        PlanningSession.commit_execution = commit_execution  # type: ignore[assignment]

    def dispatched_action_ids(self: Any) -> list[str]:
        if not self.ledger.verify_integrity():
            raise ledger_mod.LedgerTamperError(
                "evidence ledger integrity failed before recovery decision"
            )
        transaction_id = getattr(self, "_active_transaction_id", None)
        ids: list[str] = []
        for record in self.ledger.records:
            if record.event_type != ledger_mod.LedgerEventType.ACTION_DISPATCHED:
                continue
            if transaction_id and record.payload.get("transaction_id") != transaction_id:
                continue
            step_id = record.payload.get("step_id")
            if isinstance(step_id, str):
                ids.append(step_id)
        return ids

    TransactionalExecutionManager._dispatched_action_ids = dispatched_action_ids  # type: ignore[assignment]

    raw_execute_and_finalize = TransactionalExecutionManager.execute_and_finalize
    if getattr(raw_execute_and_finalize, "_runtime_closure", False):
        return

    def execute_and_finalize(self: Any, certificate: Any, execution_backend: Any = None):
        if getattr(self, "_closed", False):
            raise RuntimeError("transaction manager is already closed")
        if not self.ledger.verify_integrity():
            raise ledger_mod.LedgerTamperError(
                "evidence ledger integrity failed before transaction start"
            )

        workspace_path = self.sandbox.policy.workspace_dir
        workspace_id = workspace_identity(workspace_path)
        with CERTIFICATE_WORKSPACES_LOCK:
            existing = CERTIFICATE_WORKSPACES.get(certificate.certificate_id)
            if existing is not None and existing != workspace_id:
                raise StateDriftError(
                    "authorization certificate is already bound to another workspace"
                )
            CERTIFICATE_WORKSPACES[certificate.certificate_id] = workspace_id

        transaction_id = f"tx_{secrets.token_hex(12)}"
        self._active_transaction_id = transaction_id
        self._bound_workspace_identity = workspace_id
        self._transaction_start_index = len(self.ledger.records)
        token_tx = ACTIVE_TRANSACTION_ID.set(transaction_id)
        token_ws = ACTIVE_WORKSPACE_ID.set(workspace_id)
        token_path = ACTIVE_WORKSPACE_PATH.set(workspace_path)
        try:
            summary = self._execute_and_finalize(certificate, execution_backend)
        except BaseException:
            dispatched = any(
                record.event_type == ledger_mod.LedgerEventType.ACTION_DISPATCHED
                and record.payload.get("transaction_id") == transaction_id
                for record in self.ledger.records
            )
            if dispatched and getattr(self, "_owned_workspace", None) is not None:
                self._retained_workspace_reason = "execution-exception"
            else:
                self.close()
            raise
        finally:
            ACTIVE_WORKSPACE_PATH.reset(token_path)
            ACTIVE_WORKSPACE_ID.reset(token_ws)
            ACTIVE_TRANSACTION_ID.reset(token_tx)

        if not self.ledger.verify_integrity():
            if getattr(self, "_owned_workspace", None) is not None:
                self._retained_workspace_reason = "ledger-integrity-failure"
            raise ledger_mod.LedgerTamperError(
                "evidence ledger integrity failed after transaction finalization"
            )

        if summary.outcome in {
            transaction_mod.TransactionOutcome.COMMITTED,
            transaction_mod.TransactionOutcome.CONTAINMENT_FAILED,
        } and getattr(self, "_owned_workspace", None) is not None:
            self._retained_workspace_reason = summary.outcome.value
        else:
            self.close()
        return summary

    execute_and_finalize._runtime_closure = True  # type: ignore[attr-defined]
    TransactionalExecutionManager.execute_and_finalize = execute_and_finalize  # type: ignore[assignment]
