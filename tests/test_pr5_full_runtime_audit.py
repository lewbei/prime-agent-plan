"""Adversarial regressions from the full post-PR4 runtime audit.

These tests state public invariants rather than implementation details. They are
intentionally committed RED before the PR #5 production fixes.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
import time

import pytest

import plan
import plan_mode
from plan_mode.ast_search import ASTSearchEngine
from plan_mode.causal_validator import ActionSchema, PlanAST
from plan_mode.execution_contract import ExecutionContract, artifact_audit
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WorldFact,
)
from plan_mode.ir_search import EpistemicPlanSearch
from plan_mode.isolation import IsolationManager
from plan_mode.judges import EnsembleJudge, JudgeAdapter, JudgeVerdict
from plan_mode.probing import DiagnosticProbe, VOIProbingEngine
from plan_mode.recovery import RecoveryStatus, SagaRecoveryManager
from plan_mode.recovery_graph import DriftEvidence, recovery_decision
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, CompensationAction, SchemaMismatchError
from plan_mode.runtime.executor import StepExecutionResult, WitnessStatus
from plan_mode.runtime.ledger import EvidenceLedger
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult, SecurityProfile
from plan_mode.session import PlanningSession, SessionState


def _prov() -> Provenance:
    return Provenance(source_type=SourceType.USER_REQUIREMENT, rationale="test")


def _plan_ir(*, timeout: float = 5.0, mandatory: bool = True) -> PlanIR:
    return PlanIR(
        plan_id="p",
        goal_description="test",
        actions=[ActionIR(action_id="a1", capability_name="noop", timeout_seconds=timeout, provenance=_prov())],
        success_criteria=[SuccessCriterion(
            criterion_id="s1",
            description="done",
            condition=PredicateCondition(predicate="done", args=[1]),
            is_mandatory=mandatory,
        )],
    )


def _legacy_plan_with_input() -> str:
    return (
        "# Goal\nGoal: atomic release. In scope: local. Out of scope: network.\n\n"
        "## Success criteria\n- S1: 1 test passes. Pass/fail. Deadline: within 1 day.\n\n"
        "## Tasks\n"
        "1. Read input. Inputs: input.txt. Output: a.txt.\n"
        "2. Finish. Depends on 1. Inputs: a.txt. Output: out.txt.\n"
    )


def test_failed_execution_cwd_release_never_commits_memory_or_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("ok\n")
    execution_cwd = tmp_path / "execution"
    execution_cwd.mkdir()
    plans_dir = tmp_path / "plans"
    session = plan.start("atomic-release-regression", plans_dir=plans_dir, max_rounds=1)
    assessed = plan.assess(session, _legacy_plan_with_input(), plans_dir=plans_dir)
    assert assessed["status"] == "converged"
    assert session.get("committed_version") is None
    gate = plan.release(session, min_score=0, require_judge=False, execution_cwd=execution_cwd, plans_dir=plans_dir)
    assert gate["ok"] is False
    assert session.get("committed_version") is None
    persisted = json.loads((plans_dir / f"{session['session_id']}.json").read_text())
    assert persisted.get("committed_version") is None
    assert persisted.get("committed_plan_hash") in (None, "")


def test_direct_plan_mode_search_mutations_are_cross_process_deterministic(tmp_path):
    code = (
        "import json, plan_mode.search_engine as se; "
        "p='# Goal\\nGoal: stable.\\n## Tasks\\n1. A. Output: a.txt.'; "
        "print(json.dumps([x['note'] for x in se._mutations(p, 4)]))"
    )
    outputs = []
    for seed in ("1", "2", "17", "987654"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env,
                                   capture_output=True, text=True, timeout=10)
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout.strip()))
    assert all(item == outputs[0] for item in outputs[1:])


def test_pipeline_uses_pipefail_semantics(tmp_path):
    sandbox = ExecutionSandbox(policy=SecurityProfile.get_profile("PERMISSIVE_DEV").model_copy(
        update={"workspace_dir": str(tmp_path)}))
    result = sandbox.execute_argv_pipeline([
        [sys.executable, "-c", "import sys; print('payload'); sys.exit(7)"],
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
    ], cwd=str(tmp_path), timeout_seconds=2.0)
    assert result.returncode != 0


def test_pipeline_drains_intermediate_stderr_without_timeout(tmp_path):
    sandbox = ExecutionSandbox(policy=SecurityProfile.get_profile("PERMISSIVE_DEV").model_copy(
        update={"workspace_dir": str(tmp_path)}))
    result = sandbox.execute_argv_pipeline([
        [sys.executable, "-c", "import sys; sys.stderr.write('x'*200000); print('ok')"],
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
    ], cwd=str(tmp_path), timeout_seconds=2.0)
    assert result.timeout_exceeded is False
    assert result.returncode == 0
    assert "ok" in result.stdout


def test_unwhitelisted_environment_cannot_be_injected(tmp_path):
    sandbox = ExecutionSandbox(policy=SecurityProfile.get_profile("PERMISSIVE_DEV").model_copy(
        update={"workspace_dir": str(tmp_path)}))
    result = sandbox.execute_argv_pipeline(
        [[sys.executable, "-c", "import os; print(os.getenv('EVIL_SECRET',''))"]],
        cwd=str(tmp_path), env={"EVIL_SECRET": "leak-me"})
    assert "leak-me" not in result.stdout


def test_strict_profile_masks_prime_credential_store():
    policy = SecurityProfile.get_profile("STRICT")
    assert os.path.expanduser("~/.prime") in policy.blocked_paths


def test_spoofed_external_judge_metadata_cannot_satisfy_strict_release(tmp_path):
    session = plan_mode.start("external-attestation-regression", plans_dir=tmp_path, max_rounds=1)
    plan_mode.assess(session, "1. Setup. Output: a.txt.\n", plans_dir=tmp_path)
    session["status"] = "converged"
    session["best_score"] = 95.0
    plan_mode.record_judge(session, {
        "ok": True, "verdict": "go", "falsifiable_criteria": True,
        "source": "external_llm", "external": True,
    }, plans_dir=tmp_path)
    gate = plan_mode.release(session, min_score=0, require_judge=True,
                             require_external_judge=True, plans_dir=tmp_path)
    assert gate["ok"] is False


@pytest.mark.asyncio
async def test_judge_ensemble_does_not_reuse_prior_plan_version_votes(tmp_path, monkeypatch):
    session = plan_mode.start("stale-judge-regression", plans_dir=tmp_path, max_rounds=4)
    plan_mode.assess(session, "1. First. Output: a.txt.\n", plans_dir=tmp_path)
    plan_mode.record_judge(session, {
        "ok": True, "verdict": "go", "feasibility_0_100": 99,
        "falsifiable_criteria": True, "source": "external_llm", "external": True,
    }, round_version=1, plans_dir=tmp_path)
    plan_mode.assess(session, "1. Second changed plan. Output: b.txt.\n", plans_dir=tmp_path)
    session["best_version"] = 2

    async def current_judge(*args, **kwargs):
        return {"ok": True, "verdict": "rework", "feasibility_0_100": 20,
                "falsifiable_criteria": True, "source": "external_llm", "external": True}

    monkeypatch.setattr(plan_mode, "judge", current_judge)
    entry = await plan_mode.judge_ensemble(session, "1. Second changed plan. Output: b.txt.\n",
                                            "stale-judge-regression", n=3, plans_dir=tmp_path)
    assert not any(v.get("round_version") == 1 for v in entry.get("votes", []))


def test_plan_hash_binds_execution_semantics():
    base = _plan_ir(timeout=5.0, mandatory=True)
    timeout_changed = base.model_copy(deep=True)
    timeout_changed.actions[0].timeout_seconds = 99.0
    assert timeout_changed.compute_hash() != base.compute_hash()
    comp_changed = base.model_copy(deep=True)
    comp_changed.actions[0].compensation_action_id = "undo-a1"
    assert comp_changed.compute_hash() != base.compute_hash()
    mandatory_changed = base.model_copy(deep=True)
    mandatory_changed.success_criteria[0].is_mandatory = False
    assert mandatory_changed.compute_hash() != base.compute_hash()
    lifetime_changed = base.model_copy(deep=True)
    lifetime_changed.success_criteria[0].condition.active_until_action_id = "a1"
    assert lifetime_changed.compute_hash() != base.compute_hash()


def test_submit_draft_after_committed_state_enters_ir_valid():
    session = PlanningSession(session_id="post-commit", current_state=SessionState.COMMITTED)
    version = session.submit_draft(_plan_ir())
    assert version.version_number == 1
    assert session.current_state == SessionState.IR_VALID


def test_probe_fact_key_preserves_argument_type_identity():
    int_probe = DiagnosticProbe(probe_id="int", target_predicate="item", target_args=[123], argv_pipeline=[["echo", "x"]])
    str_probe = DiagnosticProbe(probe_id="str", target_predicate="item", target_args=["123"], argv_pipeline=[["echo", "x"]])
    assert int_probe.fact_key != str_probe.fact_key


def test_malformed_probe_output_is_unknown_not_empirical_false():
    engine = VOIProbingEngine()
    integer_probe = DiagnosticProbe(probe_id="integer", target_predicate="count",
                                    argv_pipeline=[["echo", "x"]], expected_output_parser="integer")
    json_probe = DiagnosticProbe(probe_id="json", target_predicate="json",
                                 argv_pipeline=[["echo", "x"]], expected_output_parser="json")
    assert engine.parse_probe_output(integer_probe, "not-an-integer", 0) == FactTruth.UNKNOWN
    assert engine.parse_probe_output(json_probe, "{broken", 0) == FactTruth.UNKNOWN


def test_saga_recovery_executes_compensation_and_observes_failure():
    class FailingSandbox:
        def __init__(self):
            self.calls = []
        def execute_argv_pipeline(self, pipeline, **kwargs):
            self.calls.append((pipeline, kwargs))
            return SandboxExecutionResult(returncode=7, stderr="undo failed")

    sandbox = FailingSandbox()
    manager = SagaRecoveryManager(sandbox=sandbox)  # type: ignore[arg-type]
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="do", description="do", default_compensation=CompensationAction(
        compensation_id="undo-1", capability_name="undo")))
    registry.register(CapabilityEntry(name="undo", description="undo",
                                      executor_command_template=[sys.executable, "-c", "print('undo')"]))
    action = ActionIR(action_id="a1", capability_name="do", provenance=_prov())
    step = StepExecutionResult(step_id="a1", capability_name="do", exit_code=0,
                               witness_status=WitnessStatus.WITNESSED_TRUE, duration_ms=1.0)
    session = PlanningSession(session_id="saga", current_state=SessionState.EXECUTING)
    report = manager.execute_saga_rollback(
        [step], PlanIR(plan_id="p", goal_description="g", actions=[action]), registry,
        EvidenceLedger(session_id="saga"), session)
    assert sandbox.calls, "rollback must execute the registered compensation command"
    assert report.status == RecoveryStatus.CONTAINMENT_FAILED


def test_artifact_audit_rejects_paths_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret")
    contract = ExecutionContract(expected_artifacts={str(outside): {"min_bytes": 1}})
    result = artifact_audit(contract, cwd=workspace)
    assert result["ok"] is False
    assert any("outside" in e.lower() or "workspace" in e.lower() for e in result["errors"])


def test_registry_rejects_undeclared_extra_parameters():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="strict-cap", description="strict",
                                      input_schema={"x": {"type": "int", "required": True}}))
    action = ActionIR(action_id="a", capability_name="strict-cap",
                      parameters={"x": 1, "undeclared": "surprise"}, provenance=_prov())
    with pytest.raises(SchemaMismatchError, match="undeclared"):
        registry.validate_action(action)


def test_ast_transposition_hash_includes_semantic_outputs():
    engine = ASTSearchEngine(objective="x")
    ast_a = PlanAST(goal="x", actions=[ActionSchema(id=1, name="same", outputs=["a.txt"])])
    ast_b = PlanAST(goal="x", actions=[ActionSchema(id=1, name="same", outputs=["b.txt"])])
    assert engine._state_hash(ast_a) != engine._state_hash(ast_b)


def test_recovery_decision_does_not_mutate_input_evidence():
    evidence = DriftEvidence(task="t", step=4, suspected_onset=2, why="x")
    before = copy.deepcopy(evidence)
    recovery_decision(evidence, K=2)
    assert evidence == before


def test_isolation_release_ends_prior_ownership_conflict():
    manager = IsolationManager()
    manager.acquire_artifact("x.txt", "agent-a", operation="write")
    manager.release_artifact("x.txt", "agent-a")
    manager.acquire_artifact("x.txt", "agent-b", operation="write")
    report = manager.detect_conflicts()
    assert report.ok is True, report.conflicts


class _CountingJudge(JudgeAdapter):
    def __init__(self):
        self.calls = 0
    async def evaluate(self, plan_ir, goal_description="", registry=None, observed_world_state=None, timeout=30.0):
        self.calls += 1
        return JudgeVerdict(verdict="PASS", feasibility_0_100=90, confidence=1.0,
                            provider="counting", model="counting")


def test_ir_search_judge_cache_is_bound_to_world_state():
    judge = _CountingJudge()
    search = EpistemicPlanSearch(judge=judge)
    plan_ir = PlanIR(plan_id="cache", goal_description="cache")
    fact_true = WorldFact(predicate="ready", args=[1], truth=FactTruth.VERIFIED_TRUE, provenance=_prov())
    fact_false = fact_true.model_copy(update={"truth": FactTruth.VERIFIED_FALSE})
    search._run_judge(plan_ir, [fact_true])
    search._run_judge(plan_ir, [fact_false])
    assert judge.calls == 2


@pytest.mark.asyncio
async def test_ensemble_enforces_outer_timeout_for_misbehaving_judge():
    class SlowJudge(JudgeAdapter):
        async def evaluate(self, plan_ir, goal_description="", registry=None, observed_world_state=None, timeout=30.0):
            await asyncio.sleep(0.5)
            return JudgeVerdict(verdict="PASS", feasibility_0_100=100, confidence=1.0)

    ensemble = EnsembleJudge([SlowJudge()])
    started = time.monotonic()
    verdict = await ensemble.evaluate(PlanIR(plan_id="slow", goal_description="slow"), timeout=0.02)
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert verdict.verdict == "UNKNOWN"


def test_judge_verdict_does_not_default_falsifiable_to_true():
    verdict = JudgeVerdict()
    assert verdict.falsifiable_criteria is False
