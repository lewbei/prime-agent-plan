"""Probabilistic Best-of-N self-verification inherited by Prime candidate ranking.

Inspired by LLM-as-a-Verifier (arXiv:2607.05391). Self-verification is an
inherent runtime behavior, but the model identity is not hard-coded: whichever
model generated/implemented the candidates is the default verifier model too.
Probabilistic scores rank candidates but never create empirical facts or certify
execution. Certification remains Prime's deterministic/runtime responsibility.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import CausalValidator, PlanValidationResult, ValidationStatus
from plan_mode.ir import PlanIR, WorldFact
from plan_mode.registry import CapabilityNotFoundError, CapabilityRegistry

DEFAULT_SELF_VERIFICATION_N_EVALUATIONS = 2
DEFAULT_SELF_VERIFICATION_PIVOTS = 1

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

DEFAULT_VERIFICATION_CRITERIA: Dict[str, str] = {
    "Goal satisfaction": "Does the candidate actually address the requested goal without silently dropping requirements?",
    "Causal coherence": "Are the actions ordered coherently with their stated preconditions and effects?",
    "Evidence discipline": "Does the candidate avoid treating assumptions, predictions, or missing evidence as empirical truth?",
    "Executability": "Is the candidate concrete, capability-grounded, and independently verifiable at runtime?",
    "Recovery readiness": "Does the candidate avoid unnecessary irreversible effects and preserve a viable recovery path?",
}


class SelfVerificationUnavailableError(RuntimeError):
    """Raised when inherent self-verification cannot resolve or call its backend."""


def resolve_implementation_model(
    explicit_model: Optional[str] = None,
    *,
    session: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Resolve the active implementation model without choosing one for the caller.

    Priority: explicit runtime model -> session meta/top-level identity -> runtime
    environment. Returning ``None`` is intentional: Prime must then use the
    deterministic fallback instead of silently switching to a different model.
    """
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


class ProbabilisticSelection(BaseModel):
    selected_index: int
    ranking: List[int] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    verifier_model: Optional[str] = None
    n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS
    pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS


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
    is_self_verification: bool = False
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

    def select(
        self,
        *,
        problem: str,
        candidates: List[str],
        criteria: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS,
        client: Any = None,
    ) -> ProbabilisticSelection:
        if not candidates:
            raise ValueError("Best-of-N verification requires at least one candidate")
        if not model or not str(model).strip():
            raise SelfVerificationUnavailableError(
                "Active implementation-model identity is unavailable; refusing to choose a different verifier model"
            )
        if n_evaluations < 1:
            raise ValueError("n_evaluations must be >= 1")
        if pivots < 1:
            raise ValueError("pivots must be >= 1")

        verifier_model = str(model).strip()
        select_fn = self._resolve_select_fn()
        kwargs: Dict[str, Any] = {
            "problem": problem,
            "candidates": candidates,
            "criteria": criteria or DEFAULT_VERIFICATION_CRITERIA,
            "model": verifier_model,
            "n_evaluations": n_evaluations,
            "pivots": min(pivots, len(candidates)),
        }
        selected_client = client if client is not None else self._client
        if selected_client is not None:
            kwargs["client"] = selected_client
        raw = select_fn(**kwargs)

        selected_index = int(getattr(raw, "index"))
        if selected_index < 0 or selected_index >= len(candidates):
            raise ValueError("Verifier returned an out-of-range candidate index")
        raw_ranking = list(getattr(raw, "ranking", []))
        ranking = [int(i) for i in raw_ranking] if raw_ranking else [selected_index]
        if any(i < 0 or i >= len(candidates) for i in ranking):
            raise ValueError("Verifier returned an out-of-range ranking index")
        raw_scores = list(getattr(raw, "scores", []))
        scores = [float(score) for score in raw_scores]
        if scores and len(scores) != len(candidates):
            raise ValueError("Verifier score vector length does not match candidate count")
        return ProbabilisticSelection(
            selected_index=selected_index,
            ranking=ranking,
            scores=scores,
            verifier_model=verifier_model,
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(candidates)),
        )


class PlanSelfVerifier:
    """Closed-world prefilter -> inherited same-model rank -> deterministic revalidation."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        validator: Optional[CausalValidator] = None,
        verifier: Optional[ProbabilisticSelfVerifier] = None,
        implementation_model: Optional[str] = None,
    ):
        self.registry = registry
        self.validator = validator or CausalValidator()
        self.verifier = verifier or ProbabilisticSelfVerifier()
        self.implementation_model = implementation_model

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

        effective_generator_model = resolve_implementation_model(
            generator_model or implementation_model or self.implementation_model
        )
        effective_verifier_model = verifier_model or effective_generator_model

        if not eligible_indices:
            first = validations[0]
            return PlanSelfVerificationResult(
                candidate_statuses=statuses,
                eligible_indices=[],
                validation_result=first,
                validation_status=first.status,
                is_certified=False,
                requires_rework=True,
                generator_model=effective_generator_model,
                verifier_model=effective_verifier_model,
                is_self_verification=bool(
                    effective_generator_model
                    and effective_generator_model == effective_verifier_model
                ),
                n_evaluations=n_evaluations,
                pivots=pivots,
                criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
            )

        if not effective_generator_model:
            raise SelfVerificationUnavailableError(
                "Active implementation-model identity is unavailable; cannot perform same-model self-verification"
            )

        rendered = [candidates[i].model_dump_json() for i in eligible_indices]
        problem = goal_description or candidates[eligible_indices[0]].goal_description
        soft = self.verifier.select(
            problem=problem,
            candidates=rendered,
            criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
            model=effective_verifier_model,
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
            generator_model=effective_generator_model,
            verifier_model=effective_verifier_model,
            is_self_verification=effective_generator_model == effective_verifier_model,
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(eligible_indices)),
            criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
        )

    def select_same_model(
        self,
        candidates: List[PlanIR],
        *,
        model: Optional[str] = None,
        goal_description: Optional[str] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        criteria: Optional[Dict[str, str]] = None,
        n_evaluations: int = DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        pivots: int = DEFAULT_SELF_VERIFICATION_PIVOTS,
    ) -> PlanSelfVerificationResult:
        """Compatibility alias; normal ``select`` already inherits the implementation model."""
        return self.select(
            candidates,
            goal_description=goal_description,
            observed_world_state=observed_world_state,
            criteria=criteria,
            implementation_model=model,
            n_evaluations=n_evaluations,
            pivots=pivots,
        )
