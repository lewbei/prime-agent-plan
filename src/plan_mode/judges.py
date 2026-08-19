"""Multi-Provider Plan Judges, Grounded Epistemic Verifiers, and Ensemble Consensus (Phase 5)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import httpx

from plan_mode.epistemic_validator import (
    CausalValidator,
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.ir import FactTruth, PlanIR, WorldFact
from plan_mode.registry import CapabilityRegistry


class JudgeVerdict(BaseModel):
    """Structured evaluation verdict produced by an individual judge."""
    verdict: str = "PASS"  # "PASS", "FAIL", "UNKNOWN", "REWORK"
    feasibility_0_100: float = 100.0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    blockers: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    falsifiable_criteria: bool = True
    suggested_mutations: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Dict[str, Any] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    )
    latency_ms: float = 0.0
    provider: str = "deterministic"
    model: str = "local"
    summary: str = ""
    individual_verdicts: List[JudgeVerdict] = Field(default_factory=list)


class DualJudgeComparison(BaseModel):
    """Comparative analysis contrasting blind vs grounded plan evaluations."""
    blind_verdict: JudgeVerdict
    grounded_verdict: JudgeVerdict
    verdict_concordance: bool
    blind_optimism_detected: bool  # Blind claims PASS, Grounded claims FAIL/UNKNOWN
    blind_pessimism_detected: bool  # Blind claims FAIL, Grounded claims PASS
    confidence_divergence: float
    epistemic_grounding_gap: float


class JudgeAdapter(ABC):
    """Abstract base class for all plan judges."""

    @abstractmethod
    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        """Evaluate plan and return structured JudgeVerdict."""
        pass


class GroundedEpistemicJudge(JudgeAdapter):
    """Evaluates plan via rigorous causal forward validation over 4-state epistemic lattice."""

    def __init__(self, validator: Optional[CausalValidator] = None):
        self.validator = validator or CausalValidator()

    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        t0 = time.time()
        val_res = self.validator.validate_plan(
            plan_ir,
            registry=registry,
            observed_world_state=observed_world_state,
        )
        latency = (time.time() - t0) * 1000.0

        if val_res.status == ValidationStatus.PASS:
            return JudgeVerdict(
                verdict="PASS",
                feasibility_0_100=100.0,
                confidence=1.0,
                summary="All preconditions and invariants causally verified on world state.",
                latency_ms=round(latency, 2),
                provider="grounded_epistemic",
                model="causal_validator",
            )
        elif val_res.status == ValidationStatus.UNKNOWN:
            return JudgeVerdict(
                verdict="UNKNOWN",
                feasibility_0_100=50.0,
                confidence=0.5,
                blockers=val_res.unknown_facts,
                missing_evidence=val_res.unknown_facts,
                summary=f"Plan contains {len(val_res.unknown_facts)} ungrounded UNKNOWN preconditions.",
                latency_ms=round(latency, 2),
                provider="grounded_epistemic",
                model="causal_validator",
            )
        else:
            return JudgeVerdict(
                verdict="FAIL",
                feasibility_0_100=0.0,
                confidence=0.95,
                blockers=val_res.blocker_reasons + val_res.invariants_violated,
                summary=f"Causal validation failed at step '{val_res.failed_step_id}'.",
                latency_ms=round(latency, 2),
                provider="grounded_epistemic",
                model="causal_validator",
            )


class BlindJudge(JudgeAdapter):
    """Simulates standard ungrounded LLM reviewer evaluating plan based solely on surface linguistic coherence."""

    def __init__(self, confidence: float = 0.90):
        self.confidence = confidence

    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        t0 = time.time()
        if not plan_ir.actions:
            return JudgeVerdict(
                verdict="FAIL",
                feasibility_0_100=0.0,
                confidence=0.8,
                summary="No actions scheduled.",
                latency_ms=0.1,
                provider="blind_llm",
                model="heuristic",
            )

        latency = (time.time() - t0) * 1000.0
        return JudgeVerdict(
            verdict="PASS",
            feasibility_0_100=90.0,
            confidence=self.confidence,
            summary="Plan possesses syntactically valid action sequence aligning with stated goal.",
            latency_ms=round(latency, 2),
            provider="blind_llm",
            model="heuristic",
        )


def _resolve_provider_key(env_names: List[str], provider_key: str) -> Optional[str]:
    """Resolve API key from environment variables or ~/.prime/agent/auth.json."""
    for name in env_names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    try:
        auth_path = Path.home() / ".prime" / "agent" / "auth.json"
        if auth_path.exists():
            data = json.loads(auth_path.read_text(encoding="utf-8"))
            cred = data.get(provider_key, {})
            key = cred.get("key") or cred.get("api_key") or cred.get("token")
            if key:
                return str(key).strip()
    except Exception:
        pass
    return None


class BaseLLMJudge(JudgeAdapter):
    """Base class for LLM API judges with real HTTP client dispatch, cost tracking, and error fallback."""

    PROVIDER_NAME: str = "base_llm"
    DEFAULT_MODEL: str = "default_model"
    ENV_KEY_NAMES: List[str] = []
    PROMPT_COST_PER_M: float = 2.50
    COMPLETION_COST_PER_M: float = 10.00
    ENDPOINT_URL: str = ""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        mock_response: Optional[Dict[str, Any]] = None,
        mock_error: Optional[Exception] = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or _resolve_provider_key(self.ENV_KEY_NAMES, self.PROVIDER_NAME)
        self.http_client = http_client
        self.mock_response = mock_response
        self.mock_error = mock_error

    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        t0 = time.time()

        # Check mock error for deterministic adversarial testing
        if self.mock_error is not None:
            latency = (time.time() - t0) * 1000.0
            return JudgeVerdict(
                verdict="UNKNOWN",
                feasibility_0_100=0.0,
                confidence=0.0,
                blockers=[f"{self.PROVIDER_NAME} API error: {str(self.mock_error)}"],
                summary=f"Failed to evaluate plan due to {self.PROVIDER_NAME} API error.",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
                latency_ms=round(latency, 2),
                provider=self.PROVIDER_NAME,
                model=self.model,
            )

        # Use mock response if provided
        if self.mock_response is not None:
            latency = (time.time() - t0) * 1000.0
            resp = self.mock_response
            prompt_toks = resp.get("prompt_tokens", 800)
            comp_toks = resp.get("completion_tokens", 200)
            cost = (prompt_toks / 1_000_000.0 * self.PROMPT_COST_PER_M) + (comp_toks / 1_000_000.0 * self.COMPLETION_COST_PER_M)
            return JudgeVerdict(
                verdict=resp.get("verdict", "PASS"),
                feasibility_0_100=float(resp.get("feasibility_0_100", 85.0)),
                confidence=float(resp.get("confidence", 0.9)),
                blockers=resp.get("blockers", []),
                contradictions=resp.get("contradictions", []),
                missing_evidence=resp.get("missing", []),
                falsifiable_criteria=resp.get("falsifiable_criteria", True),
                suggested_mutations=resp.get("suggested_mutations", []),
                token_usage={"prompt_tokens": prompt_toks, "completion_tokens": comp_toks, "cost_usd": round(cost, 6)},
                latency_ms=round(latency, 2),
                provider=self.PROVIDER_NAME,
                model=self.model,
                summary=resp.get("summary", f"{self.PROVIDER_NAME} evaluation completed."),
            )

        # Fail-closed if no API key and no client configured
        if not self.api_key and self.http_client is None:
            latency = (time.time() - t0) * 1000.0
            return JudgeVerdict(
                verdict="UNKNOWN",
                feasibility_0_100=0.0,
                confidence=0.0,
                blockers=[f"Provider credential/API key not configured for {self.PROVIDER_NAME}"],
                summary=f"{self.PROVIDER_NAME} judge cannot evaluate plan: API key not configured.",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
                latency_ms=round(latency, 2),
                provider=self.PROVIDER_NAME,
                model=self.model,
            )

        # Real HTTP dispatch to provider endpoint
        try:
            return await self._dispatch_api_request(plan_ir, goal_description, timeout, t0)
        except Exception as e:
            latency = (time.time() - t0) * 1000.0
            return JudgeVerdict(
                verdict="UNKNOWN",
                feasibility_0_100=0.0,
                confidence=0.0,
                blockers=[f"{self.PROVIDER_NAME} HTTP dispatch failed: {str(e)}"],
                summary=f"Failed to evaluate plan due to {self.PROVIDER_NAME} network/API failure.",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
                latency_ms=round(latency, 2),
                provider=self.PROVIDER_NAME,
                model=self.model,
            )

    async def _dispatch_api_request(
        self,
        plan_ir: PlanIR,
        goal_description: str,
        timeout: float,
        t0: float,
    ) -> JudgeVerdict:
        """Format and send HTTP request to LLM provider API endpoint."""
        prompt_content = f"Goal: {goal_description or plan_ir.goal_description}\n\nPlan IR:\n{plan_ir.model_dump_json(indent=2)}"
        system_msg = (
            "You are an adversarial plan reviewer. Evaluate feasibility, contradictions, and missing evidence. "
            "Respond ONLY with a valid JSON object matching this schema: "
            '{"verdict": "PASS" | "FAIL" | "UNKNOWN" | "REWORK", "feasibility_0_100": float, "confidence": float, "blockers": list, "contradictions": list, "summary": str}'
        )

        headers = self._get_request_headers()
        payload = self._get_request_payload(system_msg, prompt_content)

        client = self.http_client or httpx.AsyncClient(timeout=timeout)
        try:
            resp = await client.post(self.ENDPOINT_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        finally:
            if self.http_client is None:
                await client.aclose()

        latency = (time.time() - t0) * 1000.0
        return self._parse_provider_response(data, latency)

    def _get_request_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _get_request_payload(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
        }

    def _parse_provider_response(self, data: Dict[str, Any], latency_ms: float) -> JudgeVerdict:
        usage = data.get("usage", {})
        prompt_toks = usage.get("prompt_tokens", 0)
        comp_toks = usage.get("completion_tokens", 0)
        cost = (prompt_toks / 1_000_000.0 * self.PROMPT_COST_PER_M) + (comp_toks / 1_000_000.0 * self.COMPLETION_COST_PER_M)

        choices = data.get("choices", [])
        if choices and "message" in choices[0]:
            content_str = choices[0]["message"].get("content", "{}")
            try:
                parsed = json.loads(content_str)
                return JudgeVerdict(
                    verdict=parsed.get("verdict", "PASS"),
                    feasibility_0_100=float(parsed.get("feasibility_0_100", 85.0)),
                    confidence=float(parsed.get("confidence", 0.9)),
                    blockers=parsed.get("blockers", []),
                    contradictions=parsed.get("contradictions", []),
                    missing_evidence=parsed.get("missing", []),
                    falsifiable_criteria=parsed.get("falsifiable_criteria", True),
                    suggested_mutations=parsed.get("suggested_mutations", []),
                    token_usage={"prompt_tokens": prompt_toks, "completion_tokens": comp_toks, "cost_usd": round(cost, 6)},
                    latency_ms=round(latency_ms, 2),
                    provider=self.PROVIDER_NAME,
                    model=self.model,
                    summary=parsed.get("summary", f"{self.PROVIDER_NAME} live response received."),
                )
            except Exception:
                pass

        return JudgeVerdict(
            verdict="UNKNOWN",
            feasibility_0_100=0.0,
            confidence=0.0,
            blockers=["Failed to parse structured JSON from provider response"],
            token_usage={"prompt_tokens": prompt_toks, "completion_tokens": comp_toks, "cost_usd": round(cost, 6)},
            latency_ms=round(latency_ms, 2),
            provider=self.PROVIDER_NAME,
            model=self.model,
        )


class OpenAIJudge(BaseLLMJudge):
    """OpenAI GPT-4o / GPT-4o-mini plan judge adapter."""
    PROVIDER_NAME = "openai"
    DEFAULT_MODEL = "gpt-4o"
    ENV_KEY_NAMES = ["OPENAI_API_KEY"]
    PROMPT_COST_PER_M = 2.50
    COMPLETION_COST_PER_M = 10.00
    ENDPOINT_URL = "https://api.openai.com/v1/chat/completions"


class AnthropicJudge(BaseLLMJudge):
    """Anthropic Claude 3.5 Sonnet plan judge adapter."""
    PROVIDER_NAME = "anthropic"
    DEFAULT_MODEL = "claude-3-5-sonnet"
    ENV_KEY_NAMES = ["ANTHROPIC_API_KEY"]
    PROMPT_COST_PER_M = 3.00
    COMPLETION_COST_PER_M = 15.00
    ENDPOINT_URL = "https://api.anthropic.com/v1/messages"

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _get_request_payload(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": 1024,
            "system": system_msg,
            "messages": [{"role": "user", "content": user_msg}],
        }

    def _parse_provider_response(self, data: Dict[str, Any], latency_ms: float) -> JudgeVerdict:
        usage = data.get("usage", {})
        prompt_toks = usage.get("input_tokens", 0)
        comp_toks = usage.get("output_tokens", 0)
        cost = (prompt_toks / 1_000_000.0 * self.PROMPT_COST_PER_M) + (comp_toks / 1_000_000.0 * self.COMPLETION_COST_PER_M)

        content_list = data.get("content", [])
        if content_list and "text" in content_list[0]:
            try:
                parsed = json.loads(content_list[0]["text"])
                return JudgeVerdict(
                    verdict=parsed.get("verdict", "PASS"),
                    feasibility_0_100=float(parsed.get("feasibility_0_100", 85.0)),
                    confidence=float(parsed.get("confidence", 0.9)),
                    blockers=parsed.get("blockers", []),
                    contradictions=parsed.get("contradictions", []),
                    missing_evidence=parsed.get("missing", []),
                    token_usage={"prompt_tokens": prompt_toks, "completion_tokens": comp_toks, "cost_usd": round(cost, 6)},
                    latency_ms=round(latency_ms, 2),
                    provider=self.PROVIDER_NAME,
                    model=self.model,
                    summary=parsed.get("summary", "Anthropic evaluation completed."),
                )
            except Exception:
                pass
        return super()._parse_provider_response(data, latency_ms)


class GeminiJudge(BaseLLMJudge):
    """Google Gemini 2.0 Flash / Pro plan judge adapter."""
    PROVIDER_NAME = "gemini"
    DEFAULT_MODEL = "gemini-2.0-flash"
    ENV_KEY_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    PROMPT_COST_PER_M = 0.10
    COMPLETION_COST_PER_M = 0.40

    @property
    def ENDPOINT_URL(self) -> str:  # type: ignore
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def _get_request_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _get_request_payload(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        return {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_msg}\n\n{user_msg}"}]}
            ],
            "generationConfig": {"response_mime_type": "application/json"},
        }

    def _parse_provider_response(self, data: Dict[str, Any], latency_ms: float) -> JudgeVerdict:
        meta = data.get("usageMetadata", {})
        prompt_toks = meta.get("promptTokenCount", 0)
        comp_toks = meta.get("candidatesTokenCount", 0)
        cost = (prompt_toks / 1_000_000.0 * self.PROMPT_COST_PER_M) + (comp_toks / 1_000_000.0 * self.COMPLETION_COST_PER_M)

        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                try:
                    parsed = json.loads(parts[0]["text"])
                    return JudgeVerdict(
                        verdict=parsed.get("verdict", "PASS"),
                        feasibility_0_100=float(parsed.get("feasibility_0_100", 85.0)),
                        confidence=float(parsed.get("confidence", 0.9)),
                        blockers=parsed.get("blockers", []),
                        contradictions=parsed.get("contradictions", []),
                        missing_evidence=parsed.get("missing", []),
                        token_usage={"prompt_tokens": prompt_toks, "completion_tokens": comp_toks, "cost_usd": round(cost, 6)},
                        latency_ms=round(latency_ms, 2),
                        provider=self.PROVIDER_NAME,
                        model=self.model,
                        summary=parsed.get("summary", "Gemini evaluation completed."),
                    )
                except Exception:
                    pass
        return super()._parse_provider_response(data, latency_ms)


class DeepSeekJudge(BaseLLMJudge):
    """DeepSeek V3 / R1 reasoning plan judge adapter."""
    PROVIDER_NAME = "deepseek"
    DEFAULT_MODEL = "deepseek-chat"
    ENV_KEY_NAMES = ["DEEPSEEK_API_KEY"]
    PROMPT_COST_PER_M = 0.14
    COMPLETION_COST_PER_M = 0.28
    ENDPOINT_URL = "https://api.deepseek.com/chat/completions"


class EnsembleJudge(JudgeAdapter):
    """Ensemble judge aggregating multi-provider verdicts via median consensus."""

    def __init__(self, judges: Optional[List[JudgeAdapter]] = None):
        self.judges = judges or [GroundedEpistemicJudge(), OpenAIJudge()]

    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        t0 = time.time()
        tasks = [
            j.evaluate(
                plan_ir,
                goal_description=goal_description,
                registry=registry,
                observed_world_state=observed_world_state,
                timeout=timeout,
            )
            for j in self.judges
        ]
        verdicts: List[JudgeVerdict] = await asyncio.gather(*tasks)

        if not verdicts:
            return JudgeVerdict(verdict="UNKNOWN", feasibility_0_100=0.0, confidence=0.0)

        # Compute median feasibility (conservative lower median)
        feasibilities = sorted([v.feasibility_0_100 for v in verdicts])
        median_feas = feasibilities[(len(feasibilities) - 1) // 2]

        # Select representative median verdict
        med_verdict = min(verdicts, key=lambda v: abs(v.feasibility_0_100 - median_feas))

        total_prompt_toks = sum(v.token_usage.get("prompt_tokens", 0) for v in verdicts)
        total_comp_toks = sum(v.token_usage.get("completion_tokens", 0) for v in verdicts)
        total_cost = sum(v.token_usage.get("cost_usd", 0.0) for v in verdicts)
        latency = (time.time() - t0) * 1000.0

        all_blockers = []
        for v in verdicts:
            all_blockers.extend(v.blockers)

        return JudgeVerdict(
            verdict=med_verdict.verdict,
            feasibility_0_100=median_feas,
            confidence=round(sum(v.confidence for v in verdicts) / len(verdicts), 2),
            blockers=sorted(set(all_blockers)),
            summary=f"Ensemble median feasibility: {median_feas:.1f} across {len(verdicts)} judges.",
            token_usage={"prompt_tokens": total_prompt_toks, "completion_tokens": total_comp_toks, "cost_usd": round(total_cost, 6)},
            latency_ms=round(latency, 2),
            provider="ensemble",
            model=f"{len(verdicts)}_judges",
            individual_verdicts=verdicts,
        )


class DualJudgeEvaluator:
    """Dispatches both blind and grounded judges and computes multi-dimensional disagreement metrics."""

    def __init__(
        self,
        blind_judge: Optional[BlindJudge] = None,
        grounded_judge: Optional[GroundedEpistemicJudge] = None,
    ):
        self.blind_judge = blind_judge or BlindJudge()
        self.grounded_judge = grounded_judge or GroundedEpistemicJudge()

    async def evaluate_plan_async(
        self,
        plan_ir: PlanIR,
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ) -> DualJudgeComparison:
        blind = await self.blind_judge.evaluate(plan_ir)
        grounded = await self.grounded_judge.evaluate(plan_ir, registry=registry, observed_world_state=observed_world_state)

        concordance = (blind.verdict == grounded.verdict)
        blind_optimism = (blind.verdict == "PASS" and grounded.verdict in ("FAIL", "UNKNOWN"))
        blind_pessimism = (blind.verdict in ("FAIL", "UNKNOWN") and grounded.verdict == "PASS")
        conf_div = round(abs(blind.confidence - grounded.confidence), 4)
        epistemic_gap = 0.0 if concordance else round(1.0 - (1.0 - conf_div) / 2.0, 4)

        return DualJudgeComparison(
            blind_verdict=blind,
            grounded_verdict=grounded,
            verdict_concordance=concordance,
            blind_optimism_detected=blind_optimism,
            blind_pessimism_detected=blind_pessimism,
            confidence_divergence=conf_div,
            epistemic_grounding_gap=epistemic_gap,
        )

    def evaluate_plan(
        self,
        plan_ir: PlanIR,
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ) -> DualJudgeComparison:
        """Synchronous wrapper for compatibility."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.evaluate_plan_async(plan_ir, registry=registry, observed_world_state=observed_world_state),
                    ).result()
            else:
                return loop.run_until_complete(
                    self.evaluate_plan_async(plan_ir, registry=registry, observed_world_state=observed_world_state)
                )
        except Exception:
            return asyncio.run(
                self.evaluate_plan_async(plan_ir, registry=registry, observed_world_state=observed_world_state)
            )
