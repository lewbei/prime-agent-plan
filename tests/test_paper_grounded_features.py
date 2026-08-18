"""Paper-grounded feature tests.

Sources:
- ACID-Agent 2608.13900 Section 2.2.3 (isolation)
- Drift recovery graph 2608.14109 (five-node routing)
- SymPlanner 2505.01479 (typed predicates)
- FlowScout 2608.10039 (execution-feedback MCTS)
"""
from __future__ import annotations

import asyncio

import pytest

import plan_mode
from plan_mode import (
    AgentIsolation,
    DriftEvidence,
    IsolationManager,
    OperationIsolation,
    PredicateSignature,
    RecoveryGraph,
    detect_conflicts,
    feedback_penalty,
    validate_typed_atom,
)
from plan_mode.causal_validator import CausalValidator, PlanParser, Proposition


def test_isolation_detects_write_write_conflict():
    mgr = IsolationManager()
    mgr.acquire_artifact("runner.py", "agent-a", operation="write", workspace="main")
    mgr.acquire_artifact("runner.py", "agent-b", operation="write", workspace="main")
    report = mgr.detect_conflicts(isolation=OperationIsolation.SERIALIZABLE)
    assert report.ok is False
    assert any(c["path"] == "runner.py" for c in report.conflicts)

    disjoint = IsolationManager()
    disjoint.acquire_artifact("runner.py", "agent-a", operation="write", workspace="branch-a")
    disjoint.acquire_artifact("runner.py", "agent-b", operation="write", workspace="branch-b")
    assert disjoint.detect_conflicts().ok is True


def test_recovery_graph_routes_non_aligned_drift_to_terminal_node():
    evidence = DriftEvidence(
        task="task-2",
        step=3,
        suspected_onset=1,
        is_aligned=False,
        why="state drift detected",
        write_operations=["runner.py"],
        write_not_reversible=["external_api_call"],
        risk=0.8,
        checkpoint_available=True,
    )
    decision = RecoveryGraph(K=2).decide(evidence)
    assert decision.node_path[0] == "n1"
    assert decision.node_path[-1] == "n5"
    assert decision.action == "rewind"


def test_recovery_graph_backward_walk_when_no_writes():
    evidence = DriftEvidence(
        task="task-2",
        step=4,
        suspected_onset=1,
        is_aligned=False,
        why="off-task read",
        write_operations=[],
        risk=0.1,
    )
    decision = RecoveryGraph(K=2).decide(evidence)
    assert "n2" in decision.node_path
    assert decision.action == "retry"


def test_typed_predicate_validation_rejects_arity_mismatch():
    sig = PredicateSignature(name="exists", arity=1, arg_types=("path",))
    assert validate_typed_atom(Proposition("exists", ("runner.py",)), sig) == []
    errors = validate_typed_atom(Proposition("exists", ()), sig)
    assert any("arity" in e for e in errors)


def test_typed_predicate_validation_inside_validator():
    plan = (
        "# Goal\nGoal: x.\n\n"
        "## Predicate Signature: exists(path)\n"
        "## Tasks\n"
        "1. Bad. Effects: exists(). Output: a.md.\n"
    )
    ast = PlanParser.parse_plan(plan)
    result = CausalValidator.validate(ast)
    assert result["ok"] is False
    assert any(f["type"] == "type_mismatch" for f in result["flaws"])


def test_feedback_penalty_penalizes_only_mismatched_task():
    plan = (
        "# Goal\nGoal: x.\n\n## Tasks\n"
        "1. A. Output: a.md.\n"
        "2. B. Depends on 1. Inputs: a.md. Output: b.md.\n"
    )
    feedback = [{"task_id": 2, "missing_outputs": ["b.md"], "detail": "expected 40, got 3"}]
    assert feedback_penalty(plan, feedback) > 0
    no_feedback = feedback_penalty(plan, None)
    assert no_feedback == 0.0


@pytest.mark.asyncio
async def test_beam_search_accepts_execution_feedback(tmp_path):
    plan = (
        "# Goal\nGoal: x. In scope: x. Out of scope: y.\n\n"
        "## Success criteria\n- S1: 1 test passes. Pass/fail. Deadline: within 1 day.\n\n"
        "## Tasks\n1. A. Output: a.md.\n2. B. Depends on 1. Inputs: a.md. Output: b.md (verifies S1).\n"
    )
    session = plan_mode.start("feedback search", plans_dir=tmp_path)
    plan_mode.assess(session, plan, plans_dir=tmp_path)
    result = await plan_mode.search(
        session,
        iterations=1,
        width=1,
        mode="beam",
        expansion="rules",
        execution_feedback=[{"task_id": 2, "detail": "expected 40, got 3"}],
        plans_dir=tmp_path,
    )
    assert result["nodes"] >= 1
