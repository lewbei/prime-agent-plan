"""Adversarial Tests for Phase 5: Real Judges & IR-Native Search Engine.

Verifies:
- Multi-provider judge adapters (OpenAI, Anthropic, Gemini, DeepSeek, GroundedEpistemic, Ensemble).
- Structured judge contracts (verdict, feasibility, blockers, token usage, latency).
- Epistemic invariant: Judges NEVER create empirical evidence or fabricate VERIFIED_TRUE facts.
- Judge API failure handling and graceful fallback.
- Capability closed-world invariant: Search only selects registered capabilities.
- Search revalidation: Every candidate is deterministically revalidated by EpistemicCausalValidator.
- Token cost and latency tracking.
- Deterministic search reproducibility.
"""

import asyncio
import copy
import json
import pytest
from typing import Any, Dict, List

from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    ProjectedTruth,
    Provenance,
    SourceType,
    SuccessCriterion,
    WitnessabilityStatus,
    WorldFact,
)
from plan_mode.registry import (
    CapabilityEntry,
    CapabilityRegistry,
    ObservationVerifier,
)
from plan_mode.epistemic_validator import (
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.judges import (
    JudgeVerdict,
    JudgeAdapter,
    OpenAIJudge,
    AnthropicJudge,
    GeminiJudge,
    DeepSeekJudge,
    GroundedEpistemicJudge,
    EnsembleJudge,
    DualJudgeEvaluator,
)
from plan_mode.ir_search import (
    EpistemicPlanSearch,
    mutate_action_parameters,
    insert_disambiguation_action,
    causal_crossover,
    mutate_replace_action,
    mutate_reorder_actions,
    mutate_delete_action,
    mutate_insert_action,
    TokenCostTracker,
    SearchResult,
)


def _cond(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate=predicate, args=args, expected_truth=truth)


def _fact(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> WorldFact:
    return WorldFact(
        predicate=predicate,
        args=args,
        truth=truth,
        projected_truth=ProjectedTruth.SUPPORTED_TRUE if truth == FactTruth.VERIFIED_TRUE else ProjectedTruth.UNSUPPORTED,
        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE, confidence=1.0),
        metadata={"evidence_ref": f"ev_{predicate}"},
    )


def _action(
    action_id: str,
    capability_name: str,
    parameters: dict | None = None,
    preconditions: list[PredicateCondition] | None = None,
    positive_effects: list[PredicateCondition] | None = None,
    negative_effects: list[PredicateCondition] | None = None,
) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability_name,
        parameters=parameters or {},
        preconditions=preconditions or [],
        positive_effects=positive_effects or [],
        negative_effects=negative_effects or [],
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE, source_id=action_id),
    )


@pytest.fixture
def sample_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.create_file",
            description="Create file",
            input_schema={"path": {"type": "str", "required": True}},
            positive_effects=[_cond("file_exists", ["{path}"])],
            verifiers=[ObservationVerifier(verifier_id="v_cf", predicate="file_exists", target_args_mapping=["{path}"], command_template=["test", "-f", "{path}"])],
            executor_command_template=["touch", "{path}"],
        )
    )
    reg.register(
        CapabilityEntry(
            name="fs.delete_file",
            description="Delete file",
            input_schema={"path": {"type": "str", "required": True}},
            negative_effects=[_cond("file_exists", ["{path}"], FactTruth.VERIFIED_FALSE)],
            verifiers=[ObservationVerifier(verifier_id="v_df", predicate="file_exists", target_args_mapping=["{path}"], command_template=["test", "!", "-f", "{path}"])],
            executor_command_template=["rm", "-f", "{path}"],
        )
    )
    reg.register(
        CapabilityEntry(
            name="net.fetch",
            description="Fetch data from url",
            input_schema={"url": {"type": "str", "required": True}},
            positive_effects=[_cond("data_fetched", ["{url}"])],
            verifiers=[ObservationVerifier(verifier_id="v_nf", predicate="data_fetched", target_args_mapping=["{url}"], command_template=["true"])],
            executor_command_template=["true"],
        )
    )
    return reg


# ---------------------------------------------------------------------------
# Test 1: Structured Judge Output Contract
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_judge_structured_output_contract():
    """Judge verdict must adhere strictly to structured contract (verdict, blockers, tokens, latency)."""
    plan = PlanIR(
        plan_id="p1_contract",
        goal_description="Create file safely",
        initial_state=[],
        actions=[_action("a1", "fs.create_file", parameters={"path": "/tmp/test.txt"}, positive_effects=[_cond("file_exists", ["/tmp/test.txt"])])],
    )
    judge = GroundedEpistemicJudge()
    verdict = await judge.evaluate(plan)

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.verdict in ("PASS", "FAIL", "UNKNOWN", "REWORK")
    assert isinstance(verdict.feasibility_0_100, float)
    assert 0.0 <= verdict.feasibility_0_100 <= 100.0
    assert isinstance(verdict.confidence, float)
    assert isinstance(verdict.blockers, list)
    assert isinstance(verdict.token_usage, dict)
    assert "prompt_tokens" in verdict.token_usage
    assert "completion_tokens" in verdict.token_usage
    assert "cost_usd" in verdict.token_usage
    assert verdict.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Test 2: Invariant - Judge NEVER Creates Empirical Evidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_judge_never_creates_empirical_evidence():
    """An LLM judge scoring 100% feasibility must NEVER promote FactTruth.UNKNOWN to VERIFIED_TRUE in world state."""
    plan = PlanIR(
        plan_id="p2_no_magic",
        goal_description="Deploy database",
        initial_state=[],
        actions=[_action("a1", "db.deploy", positive_effects=[_cond("db_running", ["prod"])])],
    )
    # Mock judge that praises the plan
    mock_llm_judge = OpenAIJudge(
        mock_response={
            "verdict": "PASS",
            "feasibility_0_100": 100.0,
            "confidence": 1.0,
            "summary": "Plan looks flawless",
            "blockers": [],
        }
    )
    verdict = await mock_llm_judge.evaluate(plan)
    assert verdict.verdict == "PASS"

    # World state must remain strictly unmutated; judge verdict cannot alter fact truth
    validator = EpistemicCausalValidator()
    val_res = validator.validate_plan(plan)
    # Plan remains UNKNOWN because db.deploy is ungrounded / missing verifiers
    assert val_res.status == ValidationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Test 3: Judge API Failure Handling and Graceful Fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_judge_api_failure_fallback_handling():
    """When remote LLM API fails (timeout, network error, 500), judge returns structured fallback with UNKNOWN."""
    plan = PlanIR(
        plan_id="p3_fallback",
        goal_description="Sample goal",
        initial_state=[],
        actions=[],
    )
    failing_judge = AnthropicJudge(mock_error=Exception("Remote API Gateway Timeout (504)"))
    verdict = await failing_judge.evaluate(plan)

    assert verdict.verdict == "UNKNOWN"
    assert verdict.confidence == 0.0
    assert any("504" in b or "gateway timeout" in b.lower() or "error" in b.lower() for b in verdict.blockers)
    assert verdict.token_usage["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Test 4: Multi-Provider Judge Adapters Dispatch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_provider_judge_adapters_dispatch():
    """All provider adapters (OpenAI, Anthropic, Gemini, DeepSeek) return valid structured verdicts."""
    plan = PlanIR(
        plan_id="p4_providers",
        goal_description="Test providers",
        initial_state=[],
        actions=[],
    )
    openai_judge = OpenAIJudge(mock_response={"verdict": "PASS", "feasibility_0_100": 90.0, "confidence": 0.9, "summary": "OpenAI OK"})
    anthropic_judge = AnthropicJudge(mock_response={"verdict": "PASS", "feasibility_0_100": 85.0, "confidence": 0.85, "summary": "Anthropic OK"})
    gemini_judge = GeminiJudge(mock_response={"verdict": "PASS", "feasibility_0_100": 88.0, "confidence": 0.88, "summary": "Gemini OK"})
    deepseek_judge = DeepSeekJudge(mock_response={"verdict": "PASS", "feasibility_0_100": 92.0, "confidence": 0.92, "summary": "DeepSeek OK"})

    res_openai = await openai_judge.evaluate(plan)
    res_anthropic = await anthropic_judge.evaluate(plan)
    res_gemini = await gemini_judge.evaluate(plan)
    res_deepseek = await deepseek_judge.evaluate(plan)

    assert res_openai.provider == "openai"
    assert res_anthropic.provider == "anthropic"
    assert res_gemini.provider == "gemini"
    assert res_deepseek.provider == "deepseek"
    assert res_openai.feasibility_0_100 == 90.0
    assert res_anthropic.feasibility_0_100 == 85.0


# ---------------------------------------------------------------------------
# Test 5: Ensemble Judge Median Consensus
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ensemble_judge_median_consensus():
    """Ensemble judge aggregates multiple judges and takes the conservative median feasibility."""
    plan = PlanIR(
        plan_id="p5_ensemble",
        goal_description="Test ensemble",
        initial_state=[],
        actions=[],
    )
    j1 = OpenAIJudge(mock_response={"verdict": "PASS", "feasibility_0_100": 95.0, "confidence": 0.95})
    j2 = AnthropicJudge(mock_response={"verdict": "REWORK", "feasibility_0_100": 50.0, "confidence": 0.60})
    j3 = GeminiJudge(mock_response={"verdict": "PASS", "feasibility_0_100": 80.0, "confidence": 0.80})

    ensemble = EnsembleJudge(judges=[j1, j2, j3])
    verdict = await ensemble.evaluate(plan)

    # Median of [50, 80, 95] is 80.0
    assert verdict.feasibility_0_100 == 80.0
    assert len(verdict.individual_verdicts) == 3


# ---------------------------------------------------------------------------
# Test 6: Capability Closed-World Invariant in IR Search
# ---------------------------------------------------------------------------
def test_search_rejects_unregistered_capability_mutation(sample_registry):
    """Search mutation attempting to use an ungrounded/unregistered capability is rejected."""
    plan = PlanIR(
        plan_id="p6_closed_world",
        goal_description="Copy file",
        initial_state=[],
        actions=[_action("a1", "fs.create_file", parameters={"path": "/tmp/a.txt"}, positive_effects=[_cond("file_exists", ["/tmp/a.txt"])])],
    )
    # Attempt mutation to non-existent capability 'magic_teleport_tool'
    mutated = mutate_replace_action(
        plan_ir=plan,
        action_index=0,
        new_capability_name="magic_teleport_tool",
        parameters={},
        registry=sample_registry,
    )
    # Mutation should fail or return unmutated plan because magic_teleport_tool is unregistered
    validator = EpistemicCausalValidator()
    val_res = validator.validate_plan(mutated, registry=sample_registry)
    assert val_res.status == ValidationStatus.PASS  # Original plan is preserved, unregistered mutation rejected


# ---------------------------------------------------------------------------
# Test 7: Search Revalidation - Every Candidate Is Deterministically Validated
# ---------------------------------------------------------------------------
def test_search_revalidates_every_candidate_through_causal_validator(sample_registry):
    """EpistemicPlanSearch searches candidate space and returns only deterministically verified plans."""
    # Plan has unknown precondition 'file_exists(/tmp/src.txt)'
    broken_plan = PlanIR(
        plan_id="p7_search",
        goal_description="Fetch and process",
        initial_state=[],
        actions=[
            _action(
                "a1",
                "fs.delete_file",
                parameters={"path": "/tmp/src.txt"},
                preconditions=[_cond("file_exists", ["/tmp/src.txt"])],  # Missing initial precondition!
                negative_effects=[_cond("file_exists", ["/tmp/src.txt"], FactTruth.VERIFIED_FALSE)],
            )
        ],
        success_criteria=[SuccessCriterion(criterion_id="c1", description="Cleaned", condition=_cond("file_exists", ["/tmp/src.txt"], FactTruth.VERIFIED_FALSE))],
    )
    searcher = EpistemicPlanSearch(registry=sample_registry)
    result = searcher.search_best_plan(seed_plan=broken_plan, max_iterations=5, beam_width=3)

    # Output plan must have been revalidated through causal validator
    assert isinstance(result, SearchResult)
    validator = EpistemicCausalValidator()
    val_res = validator.validate_plan(result.plan, registry=sample_registry)
    assert val_res.status in (ValidationStatus.PASS, ValidationStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# Test 8: IR Mutation Operators Preserve Canonical IR Validity
# ---------------------------------------------------------------------------
def test_search_mutation_operators_preserve_ir_validity(sample_registry):
    """Parameter mutation, action reordering, and action deletion produce valid canonical PlanIR objects."""
    init_fact = _fact("file_exists", ["/tmp/x.txt"])
    plan = PlanIR(
        plan_id="p8_mutations",
        goal_description="Multi-step file ops",
        initial_state=[init_fact],
        actions=[
            _action("a1", "fs.create_file", parameters={"path": "/tmp/a.txt"}, positive_effects=[_cond("file_exists", ["/tmp/a.txt"])]),
            _action("a2", "fs.delete_file", parameters={"path": "/tmp/x.txt"}, negative_effects=[_cond("file_exists", ["/tmp/x.txt"], FactTruth.VERIFIED_FALSE)]),
        ],
    )
    # 1. Parameter mutation
    m_param = mutate_action_parameters(plan, 0, {"path": "/tmp/mutated.txt"})
    assert m_param.actions[0].parameters["path"] == "/tmp/mutated.txt"

    # 2. Reordering
    m_reorder = mutate_reorder_actions(plan, 0, 1)
    assert m_reorder.actions[0].action_id == "a2"
    assert m_reorder.actions[1].action_id == "a1"

    # 3. Deletion
    m_del = mutate_delete_action(plan, 1)
    assert len(m_del.actions) == 1
    assert m_del.actions[0].action_id == "a1"


# ---------------------------------------------------------------------------
# Test 9: Token Cost and Latency Tracking
# ---------------------------------------------------------------------------
def test_token_cost_and_latency_tracking():
    """TokenCostTracker aggregates prompt tokens, completion tokens, dollar cost, and latency."""
    tracker = TokenCostTracker()
    tracker.record_usage(provider="openai", model="gpt-4o", prompt_tokens=1000, completion_tokens=500, latency_ms=450.0)
    tracker.record_usage(provider="anthropic", model="claude-3-5-sonnet", prompt_tokens=2000, completion_tokens=1000, latency_ms=800.0)

    summary = tracker.get_summary()
    assert summary["total_prompt_tokens"] == 3000
    assert summary["total_completion_tokens"] == 1500
    assert summary["total_cost_usd"] > 0.0
    assert summary["total_latency_ms"] == 1250.0


# ---------------------------------------------------------------------------
# Test 10: Deterministic Search Reproducibility with Seed
# ---------------------------------------------------------------------------
def test_search_reproducibility_with_seed(sample_registry):
    """Running search with the same random seed produces deterministic, identical plan candidates."""
    plan = PlanIR(
        plan_id="p10_reproduce",
        goal_description="Reproducibility test",
        initial_state=[],
        actions=[_action("a1", "fs.create_file", parameters={"path": "/tmp/a.txt"}, positive_effects=[_cond("file_exists", ["/tmp/a.txt"])])],
    )
    searcher_1 = EpistemicPlanSearch(registry=sample_registry, seed=42)
    searcher_2 = EpistemicPlanSearch(registry=sample_registry, seed=42)

    result_1 = searcher_1.search_best_plan(plan, max_iterations=3, beam_width=2)
    result_2 = searcher_2.search_best_plan(plan, max_iterations=3, beam_width=2)

    assert result_1.plan.compute_hash() == result_2.plan.compute_hash()


# ---------------------------------------------------------------------------
# Test 11: Unconfigured Provider Judge Returns UNKNOWN, Never Fake PASS
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unconfigured_provider_judge_returns_unknown_never_fake_pass(monkeypatch):
    """When no API key is provided, judge must return UNKNOWN with 0 cost, never fabricate a fake PASS."""
    # Clear all potential provider keys from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    plan = PlanIR(
        plan_id="p11_no_fake_pass",
        goal_description="Test unconfigured judge",
        initial_state=[],
        actions=[],
    )
    unconfigured_judge = OpenAIJudge(api_key=None)
    verdict = await unconfigured_judge.evaluate(plan)

    assert verdict.verdict == "UNKNOWN"
    assert verdict.confidence == 0.0
    assert verdict.token_usage["prompt_tokens"] == 0
    assert verdict.token_usage["cost_usd"] == 0.0
    assert any("not configured" in b.lower() or "api key" in b.lower() for b in verdict.blockers)


# ---------------------------------------------------------------------------
# Test 12: Real HTTP Client Protocol & Real Token Calculation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_real_http_client_protocol_and_token_calculation():
    """Judge invokes HTTP client transport, extracts real token usage, and computes cost."""
    import httpx

    # Mock custom HTTP transport simulating a real OpenAI chat completion response
    def handler(request: httpx.Request):
        payload = json.loads(request.content)
        assert "model" in payload
        assert "messages" in payload
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "verdict": "PASS",
                            "feasibility_0_100": 95.0,
                            "confidence": 0.95,
                            "blockers": [],
                            "summary": "Live verified response"
                        })
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 300,
                "total_tokens": 1500
            }
        }
        return httpx.Response(200, json=response_data)

    mock_transport = httpx.MockTransport(handler)
    mock_client = httpx.AsyncClient(transport=mock_transport)

    plan = PlanIR(
        plan_id="p12_http_dispatch",
        goal_description="Test real client dispatch",
        initial_state=[],
        actions=[_action("a1", "fs.create_file", parameters={"path": "/tmp/a.txt"}, positive_effects=[_cond("file_exists", ["/tmp/a.txt"])])],
    )
    judge = OpenAIJudge(api_key="sk-test-live-key", http_client=mock_client)
    verdict = await judge.evaluate(plan)

    assert verdict.verdict == "PASS"
    assert verdict.feasibility_0_100 == 95.0
    assert verdict.token_usage["prompt_tokens"] == 1200
    assert verdict.token_usage["completion_tokens"] == 300
    # Cost for 1200 in ($2.50/M) + 300 out ($10.00/M) = $0.003 + $0.003 = $0.006
    assert verdict.token_usage["cost_usd"] == 0.006
    assert verdict.latency_ms > 0.0


# ---------------------------------------------------------------------------
# Test 13: Closed-World Enforced on Insert and Disambiguation
# ---------------------------------------------------------------------------
def test_closed_world_enforced_on_insert_and_disambiguation(sample_registry):
    """insert_disambiguation_action and mutate_insert_action reject unregistered capabilities when registry is supplied."""
    plan = PlanIR(
        plan_id="p13_closed_world",
        goal_description="Test closed world on insert",
        initial_state=[],
        actions=[_action("a1", "fs.create_file", parameters={"path": "/tmp/a.txt"}, positive_effects=[_cond("file_exists", ["/tmp/a.txt"])])],
    )

    # 1. Disambiguation with unregistered capability
    m_unregistered = insert_disambiguation_action(
        plan_ir=plan,
        target_action_index=0,
        probe_capability_name="unregistered_probe",
        parameters={},
        registry=sample_registry,
    )
    assert len(m_unregistered.actions) == 1, "Unregistered disambiguation probe was incorrectly inserted!"

    # 2. Insert with unregistered action
    unregistered_action = _action("probe_bad", "bad_tool", positive_effects=[_cond("done", [])])
    m_insert_bad = mutate_insert_action(
        plan_ir=plan,
        target_index=0,
        new_action=unregistered_action,
        registry=sample_registry,
    )
    assert len(m_insert_bad.actions) == 1, "Unregistered action was incorrectly inserted!"


# ---------------------------------------------------------------------------
# Test 14: Structured SearchResult Contract and is_certified Status
# ---------------------------------------------------------------------------
def test_search_result_contract_and_is_certified_status(sample_registry):
    """EpistemicPlanSearch returns a structured SearchResult where is_certified is True ONLY if validation is PASS."""
    from plan_mode.ir_search import SearchResult

    valid_plan = PlanIR(
        plan_id="p14_certified",
        goal_description="Create file",
        initial_state=[],
        actions=[_action("a1", "fs.create_file", parameters={"path": "/tmp/test.txt"}, positive_effects=[_cond("file_exists", ["/tmp/test.txt"])])],
        success_criteria=[SuccessCriterion(criterion_id="c1", description="Created", condition=_cond("file_exists", ["/tmp/test.txt"]))],
    )
    searcher = EpistemicPlanSearch(registry=sample_registry)
    result = searcher.search_best_plan(valid_plan)

    assert isinstance(result, SearchResult)
    assert result.is_certified is True
    assert result.validation_status == ValidationStatus.PASS
    assert "total_cost_usd" in result.cost_summary
