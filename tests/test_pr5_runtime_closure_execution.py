"""Adversarial closure tests for the final PR5 runtime audit."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import plan_mode
from plan_mode.causal_validator import ActionSchema, CausalValidator, PlanAST, Proposition
from plan_mode.execution_trace import verify_execution_trace
from plan_mode.ir import ActionIR, FactTruth, PlanIR, Provenance, SourceType, WorldFact
from plan_mode.memory_distiller import ContextBudgeter
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.runtime import (
    EvidenceLedger,
    ExecutionPlanManager,
    LedgerEventType,
    TransactionOutcome,
    TransactionalExecutionManager,
)
from plan_mode.runtime.ledger import LedgerTamperError
from plan_mode.runtime.sandbox import SandboxExecutionResult
from plan_mode.search_engine import _backprop, _fresh_tree, _hash, _new_node, _prune, _select
from plan_mode.session import CommitGateError, PlanningSession, StateDriftError, compute_world_state_hash


def _prov(source: SourceType = SourceType.PLANNER_INFERENCE) -> Provenance:
    return Provenance(source_type=source, confidence=1.0)


def _rollout(value: float = 1.0) -> dict:
    return {
        "score": value * 100.0,
        "value": value,
        "verify_ok": True,
        "sim_ok": True,
        "critiques": [],
    }


def _prepare_noop_session(session_id: str = "runtime-closure"):
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="noop",
            description="No-op",
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id=f"plan-{session_id}",
        goal_description="Execute through the transaction boundary",
        actions=[
            ActionIR(
                action_id="a1",
                capability_name="noop",
                provenance=_prov(),
            )
        ],
    )
    session = PlanningSession(session_id=session_id)
    session.submit_draft(plan)
    result = session.validate_candidate(1, registry, observed_world_state=[])
    assert result.status.value == "PASS"
    session.select_version(1)
    policy = registry.compute_registry_hash()
    certificate = session.authorize_selected(registry, policy_hash=policy)
    session.start_execution(registry, policy_hash=policy, current_world_facts=[])
    return session, registry, certificate, policy


def test_direct_custom_backend_cannot_execute_outside_transaction():
    session, registry, certificate, policy = _prepare_noop_session("direct-backend")
    calls = []

    def backend(argv, *, timeout_seconds):
        calls.append(argv)
        return SandboxExecutionResult(returncode=0)

    manager = ExecutionPlanManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy,
    )
    summary = manager.execute_authorized_plan(certificate, execution_backend=backend)
    assert summary.success is False
    assert calls == []
    assert "TransactionalExecutionManager" in (summary.step_results[-1].error_message or "")


def test_commit_cannot_be_called_outside_transaction_manager():
    session, _, _, _ = _prepare_noop_session("manual-commit")
    session.record_execution_result(True, [])
    with pytest.raises(CommitGateError, match="TransactionalExecutionManager"):
        session.commit_execution(live_world_state=[])


def test_owned_committed_workspace_survives_until_explicit_close():
    session, registry, certificate, policy = _prepare_noop_session("workspace-retain")
    ledger = EvidenceLedger(session_id=session.session_id)
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy,
        allow_insecure_test_sandbox=True,
    )
    workspace = Path(manager.workspace_dir)

    def backend(argv, *, timeout_seconds):
        (workspace / "result.txt").write_text("committed", encoding="utf-8")
        return SandboxExecutionResult(returncode=0)

    summary = manager.execute_and_finalize(certificate, execution_backend=backend)
    assert summary.outcome == TransactionOutcome.COMMITTED
    assert workspace.exists()
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "committed"
    assert manager._retained_workspace_reason == "COMMITTED"
    manager.close()
    assert not workspace.exists()


def test_certificate_cannot_be_reused_for_another_workspace():
    session, registry, certificate, policy = _prepare_noop_session("workspace-binding")
    first = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy,
        allow_insecure_test_sandbox=True,
    )
    first_path = Path(first.workspace_dir)
    summary = first.execute_and_finalize(
        certificate,
        execution_backend=lambda argv, timeout_seconds: SandboxExecutionResult(returncode=0),
    )
    assert summary.outcome == TransactionOutcome.COMMITTED

    second = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy,
        allow_insecure_test_sandbox=True,
    )
    assert Path(second.workspace_dir) != first_path
    with pytest.raises(StateDriftError, match="another workspace"):
        second.execute_and_finalize(
            certificate,
            execution_backend=lambda argv, timeout_seconds: SandboxExecutionResult(returncode=0),
        )
    second.close()
    first.close()


def test_transaction_records_are_scoped_and_ledger_tampering_fails_closed():
    session, registry, certificate, policy = _prepare_noop_session("ledger-scope")
    ledger = EvidenceLedger(session_id=session.session_id)
    ledger.append_record(LedgerEventType.ACTION_DISPATCHED, {"step_id": "old"})
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=ledger,
        observed_world_state=[],
        policy_hash=policy,
        allow_insecure_test_sandbox=True,
    )
    summary = manager.execute_and_finalize(
        certificate,
        execution_backend=lambda argv, timeout_seconds: SandboxExecutionResult(returncode=0),
    )
    assert summary.outcome == TransactionOutcome.COMMITTED
    assert manager._dispatched_action_ids() == ["a1"]
    current = [
        record for record in ledger.records
        if record.payload.get("transaction_id") == manager._active_transaction_id
    ]
    assert current
    assert all(record.payload.get("workspace_identity") for record in current)
    manager.close()

    bad_session, bad_registry, bad_certificate, bad_policy = _prepare_noop_session("ledger-tamper")
    bad_ledger = EvidenceLedger(session_id=bad_session.session_id)
    record = bad_ledger.append_record(LedgerEventType.SESSION_INIT, {"ok": True})
    record.payload["ok"] = False
    bad_manager = TransactionalExecutionManager(
        session=bad_session,
        registry=bad_registry,
        ledger=bad_ledger,
        observed_world_state=[],
        policy_hash=bad_policy,
        allow_insecure_test_sandbox=True,
    )
    with pytest.raises(LedgerTamperError):
        bad_manager.execute_and_finalize(
            bad_certificate,
            execution_backend=lambda argv, timeout_seconds: SandboxExecutionResult(returncode=0),
        )
    bad_manager.close()


def test_world_state_identity_binds_provenance():
    observed = WorldFact(
        predicate="ready",
        args=[1],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=_prov(SourceType.OBSERVED_WORLD_STATE),
    )
    asserted = observed.model_copy(
        deep=True,
        update={"provenance": _prov(SourceType.PLANNER_INFERENCE)},
    )
    assert compute_world_state_hash([observed]) != compute_world_state_hash([asserted])


def test_execution_trace_rejects_duplicate_ids_and_implicit_success():
    plan_text = "1. Build. Output: out.txt.\n"
    evidence = {
        "agent_id": "executor",
        "verifier_agent_id": "verifier",
        "tasks": [
            {"task_id": 1, "status": "done", "files_created": ["out.txt"]},
            {"task_id": 1, "files_created": ["out.txt"]},
        ],
    }
    result = verify_execution_trace(plan_text, evidence)
    assert result["ok"] is False
    assert any("duplicate task_id" in error for error in result["errors"])
    assert any("explicit status" in error for error in result["errors"])


def test_execution_trace_can_require_plan_session_and_workspace_binding():
    plan_text = "1. Build. Output: out.txt.\n"
    plan_hash = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
    evidence = {
        "agent_id": "executor",
        "verifier_agent_id": "verifier",
        "plan_hash": plan_hash,
        "session_id": "session-a",
        "workspace_identity": "workspace-a",
        "tasks": [{"task_id": 1, "status": "done", "files_created": ["out.txt"]}],
    }
    result = verify_execution_trace(
        plan_text,
        evidence,
        require_bound_identity=True,
        expected_session_id="session-b",
        expected_workspace_identity="workspace-a",
    )
    assert result["ok"] is False
    assert any("another session" in error for error in result["errors"])


def test_committed_round_is_never_folded_by_context_budgeter():
    rounds = [
        {
            "version": index,
            "plan_text": f"version {index} " + ("x" * 400),
            "score": float(index),
            "delta": 1.0,
            "critiques": [],
        }
        for index in range(1, 6)
    ]
    session = {
        "rounds": rounds,
        "best_version": 5,
        "committed_version": 1,
    }
    original = rounds[0]["plan_text"]
    ContextBudgeter.compress_history(session, max_context_tokens=50, keep_last=1)
    assert rounds[0]["plan_text"] == original
    assert rounds[0].get("folded") is not True
    assert session["context_budget_exceeded"] is True


def test_causal_validator_enforces_lower_bounds():
    ast = PlanAST(
        goal="minimum",
        actions=[ActionSchema(id=1, name="only")],
        constraints={"tasks": [{"type": "at least", "value": 2.0}]},
    )
    result = CausalValidator.validate(ast)
    assert result["ok"] is False
    assert any(flaw["type"] == "resource_minimum_not_met" for flaw in result["flaws"])


def test_negated_initial_fact_does_not_seed_positive_fact():
    ast = PlanAST(
        goal="negative initial state",
        actions=[
            ActionSchema(
                id=1,
                name="requires ready",
                preconditions=[Proposition.parse("ready")],
            )
        ],
        initial_state={"not ready"},
    )
    result = CausalValidator.validate(ast)
    assert result["ok"] is False
    assert any(flaw["type"] == "unsatisfied_precondition" for flaw in result["flaws"])
