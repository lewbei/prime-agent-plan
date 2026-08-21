"""Inherent same-model, same-thinking probabilistic self-verification.

Prime never chooses the verifier transport by credential precedence.  The
implementation model determines the provider family, the implementation
thinking profile is preserved, individual provider requests are bounded, and
an explicit call/concurrency budget prevents verifier scaling from turning a
small Best-of-N decision into an unbounded network job.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import CausalValidator, PlanValidationResult, ValidationStatus
from plan_mode.ir import PlanIR, WorldFact
from plan_mode.registry import CapabilityNotFoundError, CapabilityRegistry

DEFAULT_SELF_VERIFICATION_N_EVALUATIONS = 2
DEFAULT_SELF_VERIFICATION_PIVOTS = 1
DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_VERIFIER_MAX_WORKERS = 16
DEFAULT_VERIFIER_MAX_CALLS = 128

IMPLEMENTATION_MODEL_META_KEYS = (
    "implementation_model",
    "generator_model",
    "model",
    "model_id",
)
IMPLEMENTATION_MODEL_ENV_KEYS = (
    "PRIME_IMPLEMENTATION_MODEL",
    "PRIME_MODEL",
    "PLAN_MODEL",
)
IMPLEMENTATION_THINKING_META_KEYS = (
    "implementation_thinking",
    "thinking_profile",
    "thinking",
    "thinking_level",
    "reasoning_effort",
    "thinking_budget",
)
IMPLEMENTATION_THINKING_ENV_KEYS = (
    "PRIME_IMPLEMENTATION_THINKING",
    "PRIME_THINKING",
    "PLAN_THINKING",
)

DEFAULT_VERIFICATION_CRITERIA: Dict[str, str] = {
    "Goal satisfaction": "Does the candidate actually address the requested goal without silently dropping requirements?",
    "Causal coherence": "Are the actions ordered coherently with their stated preconditions and effects?",
    "Evidence discipline": "Does the candidate avoid treating assumptions, predictions, or missing evidence as empirical truth?",
    "Executability": "Is the candidate concrete, capability-grounded, and independently verifiable at runtime?",
    "Recovery readiness": "Does the candidate avoid unnecessary irreversible effects and preserve a viable recovery path?",
}


class SelfVerificationUnavailableError(RuntimeError):
    """Raised when inherent self-verification cannot faithfully use its runtime profile."""


def resolve_implementation_model(
    explicit_model: Optional[str] = None,
    *,
    session: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    if explicit_model and str(explicit_model).strip():
        return str(explicit_model).strip()
    if session:
        meta = session.get("meta") if isinstance(session, Mapping) else None
        for source in (meta, session):
            if not isinstance(source, Mapping):
                continue
            for key in IMPLEMENTATION_MODEL_META_KEYS:
                value = source.get(key)
                if value and str(value).strip():
                    return str(value).strip()
    env = environ if environ is not None else os.environ
    for key in IMPLEMENTATION_MODEL_ENV_KEYS:
        value = env.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def normalize_thinking_profile(value: Any = None) -> Dict[str, Any]:
    if value is None:
        return {"mode": "default"}
    if isinstance(value, Mapping):
        profile = {str(k): v for k, v in value.items() if v is not None}
        if not profile:
            return {"mode": "default"}
        if "mode" not in profile:
            if "thinking_budget" in profile:
                profile["mode"] = "budget"
            elif "thinking_level" in profile or "reasoning_effort" in profile:
                profile["mode"] = "level"
            else:
                profile["mode"] = "native"
        if "thinking_level" in profile and "reasoning_effort" in profile:
            if str(profile["thinking_level"]).lower() != str(profile["reasoning_effort"]).lower():
                raise ValueError("thinking_level and reasoning_effort disagree")
        return profile
    if isinstance(value, bool):
        level = "high" if value else "off"
        return {"mode": "level", "thinking_level": level, "reasoning_effort": level}
    if isinstance(value, int):
        if value < 0:
            raise ValueError("thinking budget must be >= 0")
        return {"mode": "budget", "thinking_budget": value}
    raw = str(value).strip()
    if not raw or raw.lower() in {"default", "provider_default", "auto"}:
        return {"mode": "default"}
    if raw.startswith("{"):
        try:
            return normalize_thinking_profile(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON thinking profile") from exc
    if raw.isdigit():
        return {"mode": "budget", "thinking_budget": int(raw)}
    level = raw.lower()
    return {"mode": "level", "thinking_level": level, "reasoning_effort": level}


def resolve_implementation_thinking(
    explicit_thinking: Any = None,
    *,
    session: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    if explicit_thinking is not None:
        return normalize_thinking_profile(explicit_thinking)
    if session:
        meta = session.get("meta") if isinstance(session, Mapping) else None
        for source in (meta, session):
            if not isinstance(source, Mapping):
                continue
            if "implementation_thinking" in source:
                return normalize_thinking_profile(source.get("implementation_thinking"))
            if "thinking_profile" in source:
                return normalize_thinking_profile(source.get("thinking_profile"))
            shaped: Dict[str, Any] = {}
            for key in ("thinking_level", "reasoning_effort", "thinking_budget", "max_tokens"):
                if source.get(key) is not None:
                    shaped[key] = source.get(key)
            if shaped:
                return normalize_thinking_profile(shaped)
            if source.get("thinking") is not None:
                return normalize_thinking_profile(source.get("thinking"))
    env = environ if environ is not None else os.environ
    for key in IMPLEMENTATION_THINKING_ENV_KEYS:
        if env.get(key) is not None:
            return normalize_thinking_profile(env.get(key))
    return {"mode": "default"}


def _callable_accepts_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if keyword in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _strip_reasoning_overrides(extra_body: Mapping[str, Any] | None) -> Dict[str, Any]:
    extra = copy.deepcopy(dict(extra_body or {}))
    extra.pop("reasoning_effort", None)
    extra.pop("thinking", None)
    chat_kwargs = extra.get("chat_template_kwargs")
    if isinstance(chat_kwargs, Mapping):
        cleaned = dict(chat_kwargs)
        cleaned.pop("enable_thinking", None)
        if cleaned:
            extra["chat_template_kwargs"] = cleaned
        else:
            extra.pop("chat_template_kwargs", None)
    google = extra.get("google")
    if isinstance(google, Mapping):
        cleaned_google = dict(google)
        cleaned_google.pop("thinking_config", None)
        if cleaned_google:
            extra["google"] = cleaned_google
        else:
            extra.pop("google", None)
    return extra


def _apply_openai_thinking(
    kwargs: Dict[str, Any],
    profile: Mapping[str, Any],
    *,
    model: str,
    is_deepseek: bool,
) -> Dict[str, Any]:
    out = dict(kwargs)
    extra = _strip_reasoning_overrides(out.get("extra_body"))
    out.pop("reasoning_effort", None)
    mode = str(profile.get("mode", "default"))
    level = profile.get("thinking_level", profile.get("reasoning_effort"))
    budget = profile.get("thinking_budget")
    max_tokens = profile.get("max_tokens")
    if max_tokens is not None:
        out["max_tokens"] = int(max_tokens)
    if mode == "default":
        if extra:
            out["extra_body"] = extra
        else:
            out.pop("extra_body", None)
        return out
    if budget is not None:
        if model.lower().startswith("gemini"):
            google = dict(extra.get("google") or {})
            google["thinking_config"] = {"thinking_budget": int(budget)}
            extra["google"] = google
            out["extra_body"] = extra
            return out
        raise SelfVerificationUnavailableError(
            f"backend cannot faithfully apply exact thinking_budget={budget} for model {model}"
        )
    if level is None:
        raise SelfVerificationUnavailableError("thinking profile has no enforceable level/effort")
    level = str(level).lower()
    if is_deepseek:
        if level in {"off", "none", "disabled", "minimal"}:
            extra["thinking"] = {"type": "disabled"}
            extra.pop("reasoning_effort", None)
        else:
            extra["thinking"] = {"type": "enabled"}
            extra["reasoning_effort"] = level
        out["extra_body"] = extra
        return out
    out["reasoning_effort"] = level
    if extra:
        out["extra_body"] = extra
    else:
        out.pop("extra_body", None)
    return out


class _CompletionsProxy:
    def __init__(self, target: Any, profile: Mapping[str, Any], model: str, is_deepseek: bool):
        self._target = target
        self._profile = profile
        self._model = model
        self._is_deepseek = is_deepseek

    def create(self, *args: Any, **kwargs: Any) -> Any:
        configured = _apply_openai_thinking(
            kwargs, self._profile, model=self._model, is_deepseek=self._is_deepseek
        )
        return self._target.create(*args, **configured)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatProxy:
    def __init__(self, target: Any, profile: Mapping[str, Any], model: str, is_deepseek: bool):
        self._target = target
        self.completions = _CompletionsProxy(target.completions, profile, model, is_deepseek)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ModelsProxy:
    def __init__(self, target: Any, profile: Mapping[str, Any]):
        self._target = target
        self._profile = profile

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.get("config")
        if config is None:
            raise SelfVerificationUnavailableError(
                "native Gemini verifier supplied no generation config; cannot enforce same thinking"
            )
        current_thinking = getattr(config, "thinking_config", None)
        if current_thinking is None:
            raise SelfVerificationUnavailableError(
                "native Gemini verifier exposes no thinking_config; cannot enforce same thinking"
            )
        thinking_type = type(current_thinking)
        mode = str(self._profile.get("mode", "default"))
        level = self._profile.get("thinking_level", self._profile.get("reasoning_effort"))
        budget = self._profile.get("thinking_budget")
        try:
            if mode == "default":
                replacement = thinking_type()
            elif budget is not None:
                replacement = thinking_type(thinking_budget=int(budget))
            elif level is not None:
                replacement = thinking_type(thinking_level=str(level).lower())
            else:
                raise SelfVerificationUnavailableError(
                    "native Gemini thinking profile has no enforceable level/budget"
                )
        except Exception as exc:
            if isinstance(exc, SelfVerificationUnavailableError):
                raise
            raise SelfVerificationUnavailableError(
                f"native Gemini client cannot apply inherited thinking profile: {exc}"
            ) from exc
        updates: Dict[str, Any] = {"thinking_config": replacement}
        if self._profile.get("max_tokens") is not None:
            updates["max_output_tokens"] = int(self._profile["max_tokens"])
        if hasattr(config, "model_copy"):
            config = config.model_copy(update=updates)
        else:
            for key, value in updates.items():
                setattr(config, key, value)
        kwargs["config"] = config
        return self._target.generate_content(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ThinkingClientProxy:
    def __init__(self, client: Any, profile: Mapping[str, Any], model: str):
        self._client = client
        self._profile = dict(profile)
        self._model = model
        self._is_deepseek = bool(getattr(client, "_llm_verifier_deepseek", False))
        if hasattr(client, "chat"):
            self.chat = _ChatProxy(client.chat, self._profile, model, self._is_deepseek)
        if hasattr(client, "models"):
            self.models = _ModelsProxy(client.models, self._profile)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _wrap_client_for_thinking(client: Any, profile: Mapping[str, Any], model: str) -> Any:
    if hasattr(client, "chat") or hasattr(client, "models"):
        return _ThinkingClientProxy(client, profile, model)
    raise SelfVerificationUnavailableError(
        f"verifier client type {type(client).__name__} cannot guarantee inherited thinking"
    )


def _model_family(model: str) -> str:
    lower = model.lower()
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith("gemini"):
        return "gemini"
    return "openai_compatible"


def _create_model_family_client(model: str, request_timeout_seconds: float) -> Any:
    """Construct only a transport capable of serving *model*; never use key precedence."""
    family = _model_family(model)
    timeout = float(request_timeout_seconds)
    if timeout <= 0:
        raise ValueError("request_timeout_seconds must be > 0")

    if family == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise SelfVerificationUnavailableError("DEEPSEEK_API_KEY is required for a DeepSeek implementation model")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SelfVerificationUnavailableError("openai client dependency unavailable") from exc
        client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
            timeout=timeout,
            max_retries=0,
        )
        client._llm_verifier_model = model
        client._llm_verifier_deepseek = True
        return client

    if family == "gemini":
        key = os.environ.get("VERTEX_API_KEY", "").strip()
        if not key:
            raise SelfVerificationUnavailableError(
                "VERTEX_API_KEY is required for Gemini self-verification because token logprobs require Vertex AI"
            )
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            http_options = types.HttpOptions(timeout=max(1, int(timeout * 1000)))
            return genai.Client(vertexai=True, api_key=key, http_options=http_options)
        except Exception as exc:  # pragma: no cover - optional provider dependency
            raise SelfVerificationUnavailableError(
                f"could not create bounded Gemini/Vertex verifier client: {exc}"
            ) from exc

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and base_url is None:
        raise SelfVerificationUnavailableError(
            "OPENAI_API_KEY or OPENAI_BASE_URL is required for an OpenAI-compatible implementation model"
        )
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SelfVerificationUnavailableError("openai client dependency unavailable") from exc
    kwargs: Dict[str, Any] = {
        "api_key": key or "EMPTY",
        "timeout": timeout,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    client._llm_verifier_model = model
    return client


def _validate_client_family(client: Any, model: str) -> None:
    family = _model_family(model)
    if family == "deepseek" and not bool(getattr(client, "_llm_verifier_deepseek", False)):
        raise SelfVerificationUnavailableError("DeepSeek model was paired with a non-DeepSeek verifier transport")
    if family == "gemini" and not hasattr(client, "models"):
        raise SelfVerificationUnavailableError("Gemini model was paired with a non-Gemini verifier transport")
    if family == "openai_compatible" and not hasattr(client, "chat"):
        raise SelfVerificationUnavailableError("OpenAI-compatible model was paired with an incompatible verifier transport")


def _estimated_verifier_calls(candidate_count: int, criteria_count: int, n_evaluations: int, pivots: int) -> int:
    # Conservative upper bound for ring pass plus pivot rounds, before each
    # directed pair is expanded over criteria and repeated evaluations.
    pair_upper = candidate_count + candidate_count * min(pivots, candidate_count)
    return pair_upper * criteria_count * n_evaluations


class ProbabilisticSelection(BaseModel):
    selected_index: int
    ranking: List[int] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    verifier_model: Optional[str] = None
    thinking_profile: Dict[str, Any] = Field(default_factory=dict)
    n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS
    pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS
    estimated_calls: int = 0


class PlanSelfVerificationResult(BaseModel):
    selected_index: Optional[int] = None
    selected_plan: Optional[PlanIR] = None
    ranking: List[int] = Field(default_factory=list)
    scores: List[Optional[float]] = Field(default_factory=list)
    candidate_statuses: Dict[int, ValidationStatus] = Field(default_factory=dict)
    eligible_indices: List[int] = Field(default_factory=list)
    validation_result: Optional[PlanValidationResult] = None
    validation_status: ValidationStatus = ValidationStatus.UNKNOWN
    is_certified: bool = False
    requires_rework: bool = True
    generator_model: Optional[str] = None
    verifier_model: Optional[str] = None
    generator_thinking: Dict[str, Any] = Field(default_factory=dict)
    verifier_thinking: Dict[str, Any] = Field(default_factory=dict)
    is_self_verification: bool = False
    is_same_thinking: bool = False
    n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS
    pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS
    criteria: Dict[str, str] = Field(default_factory=dict)


class ProbabilisticSelfVerifier:
    def __init__(self, select_fn: Optional[Callable[..., Any]] = None, client: Any = None):
        self._select_fn = select_fn
        self._client = client

    def _resolve_select_fn(self) -> Callable[..., Any]:
        if self._select_fn is not None:
            return self._select_fn
        try:
            import llm_verifier  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SelfVerificationUnavailableError(
                "Install the probabilistic backend with: pip install 'plan[verification]'"
            ) from exc
        return llm_verifier.select

    @staticmethod
    def _create_upstream_client(model: str, request_timeout_seconds: float) -> Any:
        return _create_model_family_client(model, request_timeout_seconds)

    def select(
        self,
        *,
        problem: str,
        candidates: List[str],
        criteria: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        thinking_profile: Any = None,
        n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS,
        client: Any = None,
        request_timeout_seconds: float = DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
        max_workers: int = DEFAULT_VERIFIER_MAX_WORKERS,
        max_verifier_calls: int = DEFAULT_VERIFIER_MAX_CALLS,
    ) -> ProbabilisticSelection:
        if not candidates:
            raise ValueError("Best-of-N verification requires at least one candidate")
        if not model or not str(model).strip():
            raise SelfVerificationUnavailableError(
                "Active implementation-model identity is unavailable; refusing to choose a different verifier model"
            )
        if n_evaluations < 1 or pivots < 1:
            raise ValueError("n_evaluations and pivots must be >= 1")
        if max_workers < 1 or max_verifier_calls < 1:
            raise ValueError("verifier worker/call budgets must be >= 1")

        verifier_model = str(model).strip()
        profile = normalize_thinking_profile(thinking_profile)
        criterion_map = criteria or DEFAULT_VERIFICATION_CRITERIA
        estimated_calls = _estimated_verifier_calls(
            len(candidates), len(criterion_map), n_evaluations, pivots
        )
        if estimated_calls > max_verifier_calls:
            raise SelfVerificationUnavailableError(
                f"self-verification would require up to {estimated_calls} calls, exceeding budget {max_verifier_calls}"
            )

        select_fn = self._resolve_select_fn()
        kwargs: Dict[str, Any] = {
            "problem": problem,
            "candidates": candidates,
            "criteria": criterion_map,
            "model": verifier_model,
            "n_evaluations": n_evaluations,
            "pivots": min(pivots, len(candidates)),
        }

        selected_client = client if client is not None else self._client
        if self._select_fn is None:
            if selected_client is None:
                selected_client = self._create_upstream_client(
                    verifier_model, request_timeout_seconds
                )
            _validate_client_family(selected_client, verifier_model)
            kwargs["client"] = _wrap_client_for_thinking(selected_client, profile, verifier_model)
            if _callable_accepts_keyword(select_fn, "max_workers"):
                kwargs["max_workers"] = min(max_workers, max(1, estimated_calls))
            if _callable_accepts_keyword(select_fn, "on_error"):
                kwargs["on_error"] = "raise"
            if _callable_accepts_keyword(select_fn, "progress"):
                kwargs["progress"] = False
        else:
            if selected_client is not None:
                kwargs["client"] = _wrap_client_for_thinking(selected_client, profile, verifier_model)
            if _callable_accepts_keyword(select_fn, "thinking_profile"):
                kwargs["thinking_profile"] = dict(profile)
            if _callable_accepts_keyword(select_fn, "max_workers"):
                kwargs["max_workers"] = min(max_workers, max(1, estimated_calls))
            if _callable_accepts_keyword(select_fn, "on_error"):
                kwargs["on_error"] = "raise"

        raw = select_fn(**kwargs)
        selected_index = int(getattr(raw, "index", getattr(raw, "selected_index", -1)))
        if selected_index < 0 or selected_index >= len(candidates):
            raise ValueError("Verifier returned an out-of-range candidate index")
        raw_ranking = list(getattr(raw, "ranking", []))
        ranking = [int(i) for i in raw_ranking] if raw_ranking else [selected_index]
        if any(i < 0 or i >= len(candidates) for i in ranking):
            raise ValueError("Verifier returned an out-of-range ranking index")
        scores = [float(score) for score in list(getattr(raw, "scores", []))]
        if scores and len(scores) != len(candidates):
            raise ValueError("Verifier score vector length does not match candidate count")
        return ProbabilisticSelection(
            selected_index=selected_index,
            ranking=ranking,
            scores=scores,
            verifier_model=verifier_model,
            thinking_profile=dict(profile),
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(candidates)),
            estimated_calls=estimated_calls,
        )


class PlanSelfVerifier:
    """Closed-world prefilter -> same-model/same-thinking rank -> deterministic revalidation."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        validator: Optional[CausalValidator] = None,
        verifier: Optional[ProbabilisticSelfVerifier] = None,
        implementation_model: Optional[str] = None,
        implementation_thinking: Any = None,
    ):
        self.registry = registry
        self.validator = validator or CausalValidator()
        self.verifier = verifier or ProbabilisticSelfVerifier()
        self.implementation_model = implementation_model
        self.implementation_thinking = implementation_thinking

    def _validate(
        self,
        plan: PlanIR,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]],
    ) -> PlanValidationResult:
        for action in plan.actions:
            try:
                self.registry.get(action.capability_name)
            except CapabilityNotFoundError:
                return PlanValidationResult(
                    status=ValidationStatus.UNKNOWN,
                    failed_step_id=action.action_id,
                    unknown_facts=[f"unregistered_capability({action.capability_name})"],
                )
        return self.validator.validate_plan(
            plan,
            registry=self.registry,
            observed_world_state=observed_world_state,
        )

    def select(
        self,
        candidates: List[PlanIR],
        *,
        goal_description: Optional[str] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        criteria: Optional[Dict[str, str]] = None,
        implementation_model: Optional[str] = None,
        implementation_thinking: Any = None,
        generator_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS,
    ) -> PlanSelfVerificationResult:
        if not candidates:
            raise ValueError("Plan self-verification requires at least one candidate")

        validations = [self._validate(plan, observed_world_state) for plan in candidates]
        statuses = {i: result.status for i, result in enumerate(validations)}
        pass_indices = [i for i, result in enumerate(validations) if result.status == ValidationStatus.PASS]
        unknown_indices = [i for i, result in enumerate(validations) if result.status == ValidationStatus.UNKNOWN]
        eligible_indices = pass_indices if pass_indices else unknown_indices

        effective_model = resolve_implementation_model(
            generator_model or implementation_model or self.implementation_model
        )
        if verifier_model and effective_model and verifier_model != effective_model:
            raise ValueError("inherent self-verification requires verifier_model == implementation model")
        effective_thinking = resolve_implementation_thinking(
            implementation_thinking if implementation_thinking is not None else self.implementation_thinking
        )

        if not eligible_indices:
            first = validations[0]
            return PlanSelfVerificationResult(
                candidate_statuses=statuses,
                eligible_indices=[],
                validation_result=first,
                validation_status=first.status,
                is_certified=False,
                requires_rework=True,
                generator_model=effective_model,
                verifier_model=effective_model,
                generator_thinking=dict(effective_thinking),
                verifier_thinking=dict(effective_thinking),
                is_self_verification=bool(effective_model),
                is_same_thinking=True,
                n_evaluations=n_evaluations,
                pivots=pivots,
                criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
            )

        if not effective_model:
            raise SelfVerificationUnavailableError(
                "Active implementation-model identity is unavailable; cannot perform same-model self-verification"
            )

        rendered = [candidates[i].model_dump_json() for i in eligible_indices]
        problem = goal_description or candidates[eligible_indices[0]].goal_description
        soft = self.verifier.select(
            problem=problem,
            candidates=rendered,
            criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
            model=effective_model,
            thinking_profile=effective_thinking,
            n_evaluations=n_evaluations,
            pivots=pivots,
        )
        selected_global = eligible_indices[soft.selected_index]
        ranking_global = [eligible_indices[i] for i in soft.ranking]
        scores_global: List[Optional[float]] = [None] * len(candidates)
        for local_index, score in enumerate(soft.scores):
            scores_global[eligible_indices[local_index]] = score

        final_validation = self._validate(candidates[selected_global], observed_world_state)
        certified = final_validation.status == ValidationStatus.PASS
        return PlanSelfVerificationResult(
            selected_index=selected_global,
            selected_plan=candidates[selected_global],
            ranking=ranking_global,
            scores=scores_global,
            candidate_statuses=statuses,
            eligible_indices=eligible_indices,
            validation_result=final_validation,
            validation_status=final_validation.status,
            is_certified=certified,
            requires_rework=not certified,
            generator_model=effective_model,
            verifier_model=effective_model,
            generator_thinking=dict(effective_thinking),
            verifier_thinking=dict(effective_thinking),
            is_self_verification=True,
            is_same_thinking=True,
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(eligible_indices)),
            criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
        )

    def select_same_model(
        self,
        candidates: List[PlanIR],
        *,
        model: Optional[str] = None,
        thinking: Any = None,
        goal_description: Optional[str] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        criteria: Optional[Dict[str, str]] = None,
        n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS,
    ) -> PlanSelfVerificationResult:
        return self.select(
            candidates,
            goal_description=goal_description,
            observed_world_state=observed_world_state,
            criteria=criteria,
            implementation_model=model,
            implementation_thinking=thinking,
            n_evaluations=n_evaluations,
            pivots=pivots,
        )
