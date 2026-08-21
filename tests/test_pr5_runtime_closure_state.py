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


def test_session_ids_cannot_escape_plans_directory(tmp_path):
    with pytest.raises(ValueError, match="session_id"):
        plan_mode._session_path(tmp_path, "../escape")
    with pytest.raises(ValueError, match="session_id"):
        plan_mode.session_lock(tmp_path, "a/b").__enter__()


def test_rewind_deletes_commit_fields_created_after_checkpoint(tmp_path):
    state = {
        "session_id": "rewind-safe",
        "plans_dir": str(tmp_path),
        "rounds": [{"version": 1, "plan_text": "1. A. Output: a.txt."}],
        "best_version": 1,
        "best_score": 90.0,
        "status": "converged",
    }
    plan_mode._save_session(tmp_path, state)
    cp = plan_mode.checkpoint(state, plans_dir=tmp_path, note="before release")
    state.update({
        "committed_version": 2,
        "committed_score": 99.0,
        "committed_at": "later",
        "committed_plan_hash": "later-hash",
    })
    plan_mode._save_session(tmp_path, state)

    restored = plan_mode.rewind(
        state,
        cp["checkpoint_id"],
        plans_dir=tmp_path,
    )
    assert "committed_version" not in restored
    assert "committed_score" not in restored
    assert "committed_at" not in restored
    assert "committed_plan_hash" not in restored


def test_tampered_checkpoint_is_rejected(tmp_path):
    state = {
        "session_id": "rewind-tamper",
        "plans_dir": str(tmp_path),
        "rounds": [{"version": 1, "plan_text": "original"}],
        "best_version": 1,
        "best_score": 90.0,
        "status": "converged",
    }
    plan_mode._save_session(tmp_path, state)
    cp = plan_mode.checkpoint(state, plans_dir=tmp_path)
    state["checkpoints"][-1]["snapshot"]["rounds"][0]["plan_text"] = "tampered"
    with pytest.raises(ValueError, match="integrity"):
        plan_mode.rewind(state, cp["checkpoint_id"], plans_dir=tmp_path)


def test_transposition_reuse_never_creates_self_or_ancestor_cycle():
    tree = _fresh_tree()
    root = _new_node(tree, "root", None, 0, "root", _rollout())
    tree["root"] = root
    assert _new_node(tree, "root", root, 1, "same", _rollout()) == root
    assert root not in tree["nodes"][root]["children"]

    child = _new_node(tree, "child", root, 1, "child", _rollout(0.5))
    assert _new_node(tree, "root", child, 2, "ancestor", _rollout()) == root
    assert root not in tree["nodes"][child]["children"]


def test_corrupt_search_cycles_fail_closed():
    tree = _fresh_tree()
    root = _new_node(tree, "root", None, 0, "root", _rollout())
    child = _new_node(tree, "child", root, 1, "child", _rollout(0.5))
    tree["root"] = root
    tree["nodes"][child]["children"].append(root)
    with pytest.raises(RuntimeError, match="cycle"):
        _select(tree, 1.4, 0.0)

    tree["nodes"][root]["parent"] = child
    with pytest.raises(RuntimeError, match="cycle"):
        _backprop(tree, child, 1.0)


def test_pruning_cleans_transpositions_and_never_reuses_node_ids():
    tree = _fresh_tree()
    root = _new_node(tree, "root", None, 0, "root", _rollout())
    tree["root"] = root
    child = _new_node(tree, "child", root, 1, "child", _rollout(0.1))
    tree["nodes"][child]["visits"] = 2
    tree["nodes"][child]["q"] = 0.0
    tree["nodes"][root]["q"] = 1.0
    child_hash = _hash("child")
    _prune(tree, 0.2)
    assert child not in tree["nodes"]
    assert child_hash not in tree["transposition"]
    next_id = _new_node(tree, "replacement", root, 1, "replacement", _rollout(0.5))
    assert next_id == "n3"


