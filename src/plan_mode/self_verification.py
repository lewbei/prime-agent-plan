"""Optional probabilistic Best-of-N self-verification for plans and trajectories.

This module is inspired by LLM-as-a-Verifier (arXiv:2607.05391) and its
self-verification workflow. It is deliberately advisory: probabilistic LLM
scores rank candidates but never create empirical facts, change FactTruth, or
certify execution. Plan certification remains the responsibility of the
closed-world deterministic validator and runtime empirical witnesses.

The optional ``llm-verifier`` package is loaded lazily. Core Prime runtime and
CI therefore do not require a verifier provider or API credential.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import CausalValidator, PlanValidationResult, ValidationStatus
from plan_mode.ir import PlanIR, WorldFact
from plan_mode.registry import CapabilityNotFoundError, CapabilityRegistry


DEFAULT_VERIFICATION_CRITERIA: Dict[str, str] = {
    "Goal satisfaction": "Does the candidate actually address the requested goal without silently dropping requirements?",
    "Causal coherence": "Are the actions ordered coherently with their stated preconditions and effects?",
    "Evidence discipline": "Does the candidate avoid treating assumptions, predictions, or missing evidence as empirical truth?",
    "Executability": "Is the candidate concrete, capability-grounded, and independently verifiable at runtime?",
    "Recovery readiness": "Does the candidate avoid unnecessary irreversible effects and preserve a viable recovery path?",
}


class SelfVerificationUnavailableError(RuntimeError):
    """Raised when the optional probabilistic verifier backend is unavailable."""


class ProbabilisticSelection(BaseModel):
    """Provider-neutral result from soft Best-of-N verification."""

    selected_index: int
    ranking: List[int] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    verifier_model: Optional[str] = None
    n_evaluations: int = 1
    pivots: int = 1


class PlanSelfVerificationResult(BaseModel):
    """Best-of-N result after the hard deterministic gate is re-applied."""

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
    n_evaluations: int = 1
    pivots: int = 1
    criteria: Dict[str, str] = Field(default_factory=dict)


class ProbabilisticSelfVerifier:
    """Thin adapter over ``llm_verifier.select`` with an injectable test seam.

    The upstream verifier computes fine-grained probabilistic scores and uses a
    pivot tournament for Best-of-N selection. Prime treats the returned score
    only as a preference signal.
    """

    def __init__(self, select_fn: Optional[Callable[..., Any]] = None):
        self._select_fn = select_fn

    def _resolve_select_fn(self) -> Callable[..., Any]:
        if self._select_fn is not None:
            return self._select_fn
        try:
            import llm_verifier  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise SelfVerificationUnavailableError(
                "Probabilistic self-verification requires the optional 'verification' extra: "
                "pip install 'plan[verification]'"
            ) from exc
        return llm_verifier.select

    def select(
        self,
        *,
        problem: str,
        candidates: List[str],
        criteria: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        n_evaluations: int = 4,
        pivots: int = 2,
    ) -> ProbabilisticSelection:
        if not candidates:
            raise ValueError("Best-of-N verification requires at least one candidate")
        if n_evaluations < 1:
            raise ValueError("n_evaluations must be >= 1")
        if pivots < 1:
            raise ValueError("pivots must be >= 1")

        select_fn = self._resolve_select_fn()
        kwargs: Dict[str, Any] = {
            "problem": problem,
            "candidates": candidates,
            "criteria": criteria or DEFAULT_VERIFICATION_CRITERIA,
            "n_evaluations": n_evaluations,
            "pivots": min(pivots, len(candidates)),
        }
        if model:
            kwargs["model"] = model
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
            verifier_model=model,
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(candidates)),
        )


class PlanSelfVerifier:
    """Select among PlanIR candidates without weakening Prime's truth boundary.

    Policy:
    * deterministic FAIL candidates are never shown to the LLM verifier;
    * if at least one deterministic PASS exists, only PASS candidates compete;
    * otherwise UNKNOWN candidates may be ranked for repair/rework only;
    * the selected plan is deterministically revalidated after ranking;
    * ``is_certified`` is true only for deterministic PASS.
    """

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        validator: Optional[CausalValidator] = None,
        verifier: Optional[ProbabilisticSelfVerifier] = None,
    ):
        self.registry = registry
        self.validator = validator or CausalValidator()
        self.verifier = verifier or ProbabilisticSelfVerifier()

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
        generator_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        n_evaluations: int = 4,
        pivots: int = 2,
    ) -> PlanSelfVerificationResult:
        if not candidates:
            raise ValueError("Plan self-verification requires at least one candidate")

        validations = [self._validate(plan, observed_world_state) for plan in candidates]
        statuses = {i: result.status for i, result in enumerate(validations)}
        pass_indices = [i for i, result in enumerate(validations) if result.status == ValidationStatus.PASS]
        unknown_indices = [i for i, result in enumerate(validations) if result.status == ValidationStatus.UNKNOWN]

        eligible_indices = pass_indices if pass_indices else unknown_indices
        if not eligible_indices:
            first = validations[0]
            return PlanSelfVerificationResult(
                candidate_statuses=statuses,
                eligible_indices=[],
                validation_result=first,
                validation_status=first.status,
                is_certified=False,
                requires_rework=True,
                generator_model=generator_model,
                verifier_model=verifier_model,
                is_self_verification=bool(generator_model and verifier_model and generator_model == verifier_model),
                n_evaluations=n_evaluations,
                pivots=pivots,
                criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
            )

        rendered = [candidates[i].model_dump_json() for i in eligible_indices]
        problem = goal_description or candidates[eligible_indices[0]].goal_description
        soft = self.verifier.select(
            problem=problem,
            candidates=rendered,
            criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
            model=verifier_model,
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
            generator_model=generator_model,
            verifier_model=verifier_model,
            is_self_verification=bool(generator_model and verifier_model and generator_model == verifier_model),
            n_evaluations=n_evaluations,
            pivots=min(pivots, len(eligible_indices)),
            criteria=criteria or DEFAULT_VERIFICATION_CRITERIA,
        )
