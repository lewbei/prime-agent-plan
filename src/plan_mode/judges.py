"""Multi-provider plan judges with strict, fail-closed structured verdicts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from plan_mode.epistemic_validator import CausalValidator, ValidationStatus
from plan_mode.ir import PlanIR, WorldFact
from plan_mode.registry import CapabilityRegistry

JudgeVerdictValue = Literal["PASS", "FAIL", "UNKNOWN", "REWORK"]


class JudgeVerdict(BaseModel):
    """Strict evaluation verdict produced by an individual judge.

    The safe default is UNKNOWN.  A missing or malformed provider field must
    never silently become PASS or high confidence.
    """

    verdict: JudgeVerdictValue = "UNKNOWN"
    feasibility_0_100: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    blockers: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    falsifiable_criteria: bool = True
    suggested_mutations: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Dict[str, Any] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
        }
    )
    latency_ms: float = 0.0
    provider: str = "deterministic"
    model: str = "local"
    summary: str = ""
    individual_verdicts: List["JudgeVerdict"] = Field(default_factory=list)


class DualJudgeComparison(BaseModel):
    blind_verdict: JudgeVerdict
    grounded_verdict: JudgeVerdict
    verdict_concordance: bool
    blind_optimism_detected: bool
    blind_pessimism_detected: bool
    confidence_divergence: float
    epistemic_grounding_gap: float


class JudgeAdapter(ABC):
    @abstractmethod
    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        pass


class GroundedEpistemicJudge(JudgeAdapter):
    """Deterministic judge backed by the causal/epistemic validator."""

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
        if val_res.status == ValidationStatus.UNKNOWN:
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
    """Explicit heuristic baseline; never confused with a live provider."""

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
        if not plan_ir.actions:
            return JudgeVerdict(
                verdict="FAIL",
                feasibility_0_100=0.0,
                confidence=0.8,
                summary="No actions scheduled.",
                latency_ms=0.1,
                provider="blind_heuristic",
                model="heuristic",
            )
        return JudgeVerdict(
            verdict="PASS",
            feasibility_0_100=90.0,
            confidence=self.confidence,
            summary="Surface-level heuristic baseline judges the action sequence coherent.",
            latency_ms=0.1,
            provider="blind_heuristic",
            model="heuristic",
        )


def _resolve_provider_key(env_names: List[str], provider_key: str) -> Optional[str]:
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        auth_path = Path.home() / ".prime" / "agent" / "auth.json"
        if auth_path.exists():
            data = json.loads(auth_path.read_text(encoding="utf-8"))
            credentials = data.get(provider_key, {})
            key = credentials.get("key") or credentials.get("api_key") or credentials.get("token")
            if key:
                return str(key).strip()
    except Exception:
        pass
    return None


class BaseLLMJudge(JudgeAdapter):
    """Base class for real HTTP provider judges with strict response parsing."""

    PROVIDER_NAME: str = "base_llm"
    DEFAULT_MODEL: str = "default_model"
    ENV_KEY_NAMES: List[str] = []
    PROMPT_COST_PER_M: float = 2.50
    COMPLETION_COST_PER_M: float = 10.00
    ENDPOINT_URL: str = ""
    REQUIRED_RESPONSE_FIELDS = frozenset({"verdict", "feasibility_0_100", "confidence"})

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

    def _token_usage(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        cost = (
            prompt_tokens / 1_000_000.0 * self.PROMPT_COST_PER_M
            + completion_tokens / 1_000_000.0 * self.COMPLETION_COST_PER_M
        )
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": round(cost, 6),
        }

    def _unknown(
        self,
        reason: str,
        latency_ms: float,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> JudgeVerdict:
        return JudgeVerdict(
            verdict="UNKNOWN",
            feasibility_0_100=0.0,
            confidence=0.0,
            blockers=[reason],
            summary=reason,
            token_usage=self._token_usage(prompt_tokens, completion_tokens),
            latency_ms=round(latency_ms, 2),
            provider=self.PROVIDER_NAME,
            model=self.model,
        )

    def _verdict_from_payload(
        self,
        payload: Any,
        latency_ms: float,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> JudgeVerdict:
        if not isinstance(payload, dict):
            return self._unknown(
                "Provider returned a non-object structured verdict.",
                latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        missing = sorted(self.REQUIRED_RESPONSE_FIELDS - set(payload))
        if missing:
            return self._unknown(
                f"Provider response missing required judge fields: {', '.join(missing)}",
                latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        try:
            return JudgeVerdict(
                verdict=payload["verdict"],
                feasibility_0_100=payload["feasibility_0_100"],
                confidence=payload["confidence"],
                blockers=payload.get("blockers", []),
                contradictions=payload.get("contradictions", []),
                missing_evidence=payload.get("missing_evidence", payload.get("missing", [])),
                falsifiable_criteria=payload.get("falsifiable_criteria", True),
                suggested_mutations=payload.get("suggested_mutations", []),
                token_usage=self._token_usage(prompt_tokens, completion_tokens),
                latency_ms=round(latency_ms, 2),
                provider=self.PROVIDER_NAME,
                model=self.model,
                summary=payload.get("summary", f"{self.PROVIDER_NAME} evaluation completed."),
            )
        except (ValidationError, TypeError, ValueError) as exc:
            return self._unknown(
                f"Invalid structured judge response: {exc}",
                latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    async def evaluate(
        self,
        plan_ir: PlanIR,
        goal_description: str = "",
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        timeout: float = 30.0,
    ) -> JudgeVerdict:
        t0 = time.time()

        if self.mock_error is not None:
            return self._unknown(
                f"{self.PROVIDER_NAME} API error: {self.mock_error}",
                (time.time() - t0) * 1000.0,
            )

        if self.mock_response is not None:
            response = dict(self.mock_response)
            prompt_tokens = int(response.pop("prompt_tokens", 800))
            completion_tokens = int(response.pop("completion_tokens", 200))
            return self._verdict_from_payload(
                response,
                (time.time() - t0) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        if not self.api_key and self.http_client is None:
            return self._unknown(
                f"Provider credential/API key not configured for {self.PROVIDER_NAME}",
                (time.time() - t0) * 1000.0,
            )

        try:
            return await self._dispatch_api_request(plan_ir, goal_description, timeout, t0)
        except Exception as exc:
            return self._unknown(
                f"{self.PROVIDER_NAME} HTTP dispatch failed: {exc}",
                (time.time() - t0) * 1000.0,
            )

    async def _dispatch_api_request(
        self,
        plan_ir: PlanIR,
        goal_description: str,
        timeout: float,
        t0: float,
    ) -> JudgeVerdict:
        prompt_content = (
            f"Goal: {goal_description or plan_ir.goal_description}\n\n"
            f"Plan IR:\n{plan_ir.model_dump_json(indent=2)}"
        )
        system_msg = (
            "You are an adversarial plan reviewer. Evaluate feasibility, contradictions, and missing evidence. "
            "Respond ONLY with a JSON object containing required fields verdict, feasibility_0_100, confidence, "
            "and optional blockers, contradictions, missing_evidence, suggested_mutations, summary. "
            "Never claim empirical facts merely because the plan declares them."
        )

        client = self.http_client or httpx.AsyncClient(timeout=timeout)
        try:
            response = await client.post(
                self.ENDPOINT_URL,
                headers=self._get_request_headers(),
                json=self._get_request_payload(system_msg, prompt_content),
            )
            response.raise_for_status()
            data = response.json()
        finally:
            if self.http_client is None:
                await client.aclose()

        return self._parse_provider_response(data, (time.time() - t0) * 1000.0)

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                return self._verdict_from_payload(
                    parsed,
                    latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        return self._unknown(
            "Failed to parse structured JSON from provider response",
            latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OpenAIJudge(BaseLLMJudge):
    PROVIDER_NAME = "openai"
    DEFAULT_MODEL = "gpt-4o"
    ENV_KEY_NAMES = ["OPENAI_API_KEY"]
    PROMPT_COST_PER_M = 2.50
    COMPLETION_COST_PER_M = 10.00
    ENDPOINT_URL = "https://api.openai.com/v1/chat/completions"


class AnthropicJudge(BaseLLMJudge):
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
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)
        content_list = data.get("content", []) if isinstance(data, dict) else []
        if content_list and isinstance(content_list[0], dict):
            text = content_list[0].get("text")
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                return self._verdict_from_payload(
                    parsed,
                    latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        return self._unknown(
            "Failed to parse structured JSON from Anthropic response",
            latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class GeminiJudge(BaseLLMJudge):
    PROVIDER_NAME = "gemini"
    DEFAULT_MODEL = "gemini-2.0-flash"
    ENV_KEY_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    PROMPT_COST_PER_M = 0.10
    COMPLETION_COST_PER_M = 0.40

    @property
    def ENDPOINT_URL(self) -> str:  # type: ignore[override]
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

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
        metadata = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
        prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
        completion_tokens = int(metadata.get("candidatesTokenCount", 0) or 0)
        candidates = data.get("candidates", []) if isinstance(data, dict) else []
        if candidates and isinstance(candidates[0], dict):
            content = candidates[0].get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            if parts and isinstance(parts[0], dict):
                text = parts[0].get("text")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = None
                    return self._verdict_from_payload(
                        parsed,
                        latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
        return self._unknown(
            "Failed to parse structured JSON from Gemini response",
            latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class DeepSeekJudge(BaseLLMJudge):
    PROVIDER_NAME = "deepseek"
    DEFAULT_MODEL = "deepseek-chat"
    ENV_KEY_NAMES = ["DEEPSEEK_API_KEY"]
    PROMPT_COST_PER_M = 0.14
    COMPLETION_COST_PER_M = 0.28
    ENDPOINT_URL = "https://api.deepseek.com/chat/completions"


class EnsembleJudge(JudgeAdapter):
    """Aggregate independent judge verdicts without creating empirical truth."""

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
        verdicts = await asyncio.gather(
            *[
                judge.evaluate(
                    plan_ir,
                    goal_description=goal_description,
                    registry=registry,
                    observed_world_state=observed_world_state,
                    timeout=timeout,
                )
                for judge in self.judges
            ]
        )
        if not verdicts:
            return JudgeVerdict()

        feasibilities = sorted(verdict.feasibility_0_100 for verdict in verdicts)
        median_feasibility = feasibilities[(len(feasibilities) - 1) // 2]
        representative = min(
            verdicts,
            key=lambda verdict: abs(verdict.feasibility_0_100 - median_feasibility),
        )
        prompt_tokens = sum(v.token_usage.get("prompt_tokens", 0) for v in verdicts)
        completion_tokens = sum(v.token_usage.get("completion_tokens", 0) for v in verdicts)
        total_cost = sum(v.token_usage.get("cost_usd", 0.0) for v in verdicts)
        blockers = sorted({blocker for verdict in verdicts for blocker in verdict.blockers})

        return JudgeVerdict(
            verdict=representative.verdict,
            feasibility_0_100=median_feasibility,
            confidence=round(sum(v.confidence for v in verdicts) / len(verdicts), 2),
            blockers=blockers,
            summary=f"Ensemble median feasibility: {median_feasibility:.1f} across {len(verdicts)} judges.",
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(total_cost, 6),
            },
            latency_ms=round((time.time() - t0) * 1000.0, 2),
            provider="ensemble",
            model=f"{len(verdicts)}_judges",
            individual_verdicts=verdicts,
        )


class DualJudgeEvaluator:
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
        grounded = await self.grounded_judge.evaluate(
            plan_ir,
            registry=registry,
            observed_world_state=observed_world_state,
        )
        concordance = blind.verdict == grounded.verdict
        confidence_divergence = round(abs(blind.confidence - grounded.confidence), 4)
        return DualJudgeComparison(
            blind_verdict=blind,
            grounded_verdict=grounded,
            verdict_concordance=concordance,
            blind_optimism_detected=(
                blind.verdict == "PASS" and grounded.verdict in ("FAIL", "UNKNOWN")
            ),
            blind_pessimism_detected=(
                blind.verdict in ("FAIL", "UNKNOWN") and grounded.verdict == "PASS"
            ),
            confidence_divergence=confidence_divergence,
            epistemic_grounding_gap=(
                0.0
                if concordance
                else round(1.0 - (1.0 - confidence_divergence) / 2.0, 4)
            ),
        )

    def evaluate_plan(
        self,
        plan_ir: PlanIR,
        registry: Optional[CapabilityRegistry] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ) -> DualJudgeComparison:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.evaluate_plan_async(
                            plan_ir,
                            registry=registry,
                            observed_world_state=observed_world_state,
                        ),
                    ).result()
            return loop.run_until_complete(
                self.evaluate_plan_async(
                    plan_ir,
                    registry=registry,
                    observed_world_state=observed_world_state,
                )
            )
        except Exception:
            return asyncio.run(
                self.evaluate_plan_async(
                    plan_ir,
                    registry=registry,
                    observed_world_state=observed_world_state,
                )
            )
