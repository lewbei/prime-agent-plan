"""Gate-level adversarial tests for Phase 4 isolation and Phase 5 judge/search integration."""
from __future__ import annotations

import json
import os

import httpx
import pytest

from plan_mode.epistemic_validator import ValidationStatus
from plan_mode.ir import ActionIR, PlanIR, PredicateCondition, Provenance, SourceType
from plan_mode.ir_search import (
    EpistemicPlanSearch,
    causal_crossover,
    insert_disambiguation_action,
    mutate_insert_action,
)
from plan_mode.judges import OpenAIJudge
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier
from plan_mode.runtime import EvidenceLedger, TransactionOutcome, TransactionalExecutionManager
from plan_mode.runtime.sandbox import (
    ExecutionSandbox,
    IsolationPolicy,
    SandboxSecurityViolationError,
    SecurityProfile,
)
from plan_mode.session import PlanningSession


def _prov() -> Provenance:
    return Provenance(source_type=SourceType.PLANNER_INFERENCE, confidence=1.0)


def _action(action_id: str, capability: str) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability,
        parameters={},
        provenance=_prov(),
    )


def _prepare_transaction(plan: PlanIR, registry: CapabilityRegistry):
    session = PlanningSession(session_id=f"gate-{plan.plan_id}")
    session.submit_draft(plan)
    result = session.validate_candidate(1, registry, observed_world_state=[])
    assert result.status == ValidationStatus.PASS
    session.select_version(1)
    policy_hash = registry.compute_registry_hash()
    cert = session.authorize_selected(registry, policy_hash=policy_hash)
    session.start_execution(registry, policy_hash=policy_hash, current_world_facts=[])
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy_hash,
    )
    return session, manager, cert


def test_strict_security_profile_requires_kernel_isolation():
    strict = SecurityProfile.get_profile(SecurityProfile.STRICT)
    assert strict.use_bwrap is True
    assert strict.require_bwrap is True
    assert strict.allow_unisolated_fallback is False


def test_transaction_default_owns_ephemeral_workspace_and_fails_closed_if_bwrap_missing():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="noop",
            description="No-op",
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(
        plan_id="secure-default",
        goal_description="Exercise secure transaction default",
        actions=[_action("a1", "noop")],
    )
    session, manager, cert = _prepare_transaction(plan, registry)
    workspace = manager.workspace_dir
    assert workspace is not None
    assert os.path.isdir(workspace)
    assert manager.sandbox.is_fail_closed is True

    isolation_ready = manager.sandbox.kernel_isolation_ready
    ledger = manager.ledger
    summary = manager.execute_and_finalize(cert)

    assert not os.path.exists(workspace), "owned transaction workspace was not wiped"
    if isolation_ready:
        assert summary.outcome == TransactionOutcome.COMMITTED
    else:
        assert summary.outcome == TransactionOutcome.FAILED
        assert any("isolation" in blocker.lower() or "bwrap" in blocker.lower() for blocker in summary.commit_blockers)
        assert not any(record.event_type.value == "ACTION_DISPATCHED" for record in ledger.records)


def test_transaction_rejects_insecure_supplied_sandbox_by_default(tmp_path):
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    plan = PlanIR(plan_id="reject-insecure", goal_description="Reject insecure sandbox", actions=[_action("a1", "noop")])
    session = PlanningSession(session_id="reject-insecure")
    session.submit_draft(plan)
    session.validate_candidate(1, registry, observed_world_state=[])
    session.select_version(1)
    policy_hash = registry.compute_registry_hash()
    session.authorize_selected(registry, policy_hash=policy_hash)
    session.start_execution(registry, policy_hash=policy_hash, current_world_facts=[])

    insecure = ExecutionSandbox(
        IsolationPolicy(
            workspace_dir=str(tmp_path),
            use_bwrap=False,
            require_bwrap=False,
            allow_unisolated_fallback=True,
        )
    )
    with pytest.raises(SandboxSecurityViolationError):
        TransactionalExecutionManager(
            session=session,
            registry=registry,
            ledger=EvidenceLedger(session_id=session.session_id),
            observed_world_state=[],
            policy_hash=policy_hash,
            sandbox=insecure,
        )


def test_production_transaction_rejects_custom_backend_bypass():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    plan = PlanIR(plan_id="backend-bypass", goal_description="Reject backend bypass", actions=[_action("a1", "noop")])
    _, manager, cert = _prepare_transaction(plan, registry)

    def backend(argv, *, timeout_seconds):  # pragma: no cover - must never be called
        raise AssertionError("custom backend executed")

    with pytest.raises(SandboxSecurityViolationError):
        manager.execute_and_finalize(cert, execution_backend=backend)


@pytest.mark.asyncio
async def test_malformed_provider_json_is_unknown_not_pass():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = OpenAIJudge(api_key="test-key", http_client=client)
    verdict = await judge.evaluate(PlanIR(plan_id="malformed", goal_description="x", actions=[]))
    await client.aclose()

    assert verdict.verdict == "UNKNOWN"
    assert verdict.confidence == 0.0
    assert any("missing required" in blocker.lower() for blocker in verdict.blockers)


@pytest.mark.asyncio
async def test_invalid_provider_verdict_enum_is_unknown():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "verdict": "YES",
            "feasibility_0_100": 99,
            "confidence": 0.99,
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = OpenAIJudge(api_key="test-key", http_client=client)
    verdict = await judge.evaluate(PlanIR(plan_id="bad-enum", goal_description="x", actions=[]))
    await client.aclose()
    assert verdict.verdict == "UNKNOWN"


def test_effect_creating_mutations_fail_closed_without_registry():
    parent = PlanIR(plan_id="p", goal_description="p", actions=[_action("a1", "known")])
    other = PlanIR(plan_id="q", goal_description="q", actions=[_action("a2", "magic")])
    magic = _action("insert", "magic")

    assert mutate_insert_action(parent, 0, magic, registry=None).compute_hash() == parent.compute_hash()
    assert insert_disambiguation_action(parent, 0, "magic", {}, registry=None).compute_hash() == parent.compute_hash()
    assert causal_crossover(parent, other, registry=None).compute_hash() == parent.compute_hash()


def test_disambiguation_effects_are_registry_derived_not_caller_invented():
    registry = CapabilityRegistry()
    registered_effect = PredicateCondition(predicate="observed", args=["{item}"])
    registry.register(
        CapabilityEntry(
            name="probe.observe",
            description="Observe item",
            input_schema={"item": {"type": "str", "required": True}},
            positive_effects=[registered_effect],
            verifiers=[
                ObservationVerifier(
                    verifier_id="observe",
                    predicate="observed",
                    target_args_mapping=["{item}"],
                    command_template=["true"],
                )
            ],
            executor_command_template=["true"],
        )
    )
    plan = PlanIR(plan_id="derive", goal_description="derive", actions=[])
    forged = [PredicateCondition(predicate="admin_access", args=[])]
    mutated = insert_disambiguation_action(
        plan,
        0,
        "probe.observe",
        {"item": "alpha"},
        positive_effects=forged,
        registry=registry,
    )
    assert len(mutated.actions) == 1
    assert [effect.predicate for effect in mutated.actions[0].positive_effects] == ["observed"]
    assert mutated.actions[0].positive_effects[0].args == ["alpha"]


def test_judge_drives_closed_world_mutation_then_deterministic_revalidation_and_cost_tracking():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityEntry(
            name="noop",
            description="Grounded no-op",
            executor_command_template=["true"],
        )
    )
    seed = PlanIR(
        plan_id="judge-search",
        goal_description="Replace missing capability",
        actions=[_action("a1", "missing.capability")],
    )
    judge = OpenAIJudge(
        mock_response={
            "verdict": "REWORK",
            "feasibility_0_100": 40.0,
            "confidence": 0.9,
            "suggested_mutations": [
                {
                    "op": "replace_action",
                    "action_index": 0,
                    "capability_name": "noop",
                    "parameters": {},
                }
            ],
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }
    )
    search = EpistemicPlanSearch(registry=registry, judge=judge, seed=7)
    result = search.search_best_plan(seed, max_iterations=2, beam_width=3)

    assert result.is_certified is True
    assert result.validation_status == ValidationStatus.PASS
    assert result.plan.actions[0].capability_name == "noop"
    assert result.cost_summary["calls_count"] >= 1
    assert result.cost_summary["total_prompt_tokens"] >= 100
    assert any(entry.get("judge_provider") == "openai" for entry in result.trajectory)
