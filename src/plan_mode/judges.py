"""Dual Divergence Judges: Blind LLM Judge vs Grounded Epistemic Judge."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import (
    CausalValidator,
    EpistemicCausalValidator,
    ValidationStatus,
)
from plan_mode.ir import FactTruth, PlanIR


class JudgeVerdict(BaseModel):
    """Evaluation verdict produced by an individual judge."""
    verdict: str  # "PASS", "FAIL", "UNKNOWN"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    blockers: List[str] = Field(default_factory=list)
    rationale: str = ""


class DualJudgeComparison(BaseModel):
    """Comparative analysis contrasting blind vs grounded plan evaluations."""
    blind_verdict: JudgeVerdict
    grounded_verdict: JudgeVerdict
    verdict_concordance: bool
    blind_optimism_detected: bool  # Blind claims PASS, Grounded claims FAIL/UNKNOWN
    blind_pessimism_detected: bool  # Blind claims FAIL, Grounded claims PASS
    confidence_divergence: float
    epistemic_grounding_gap: float


class BlindJudge:
    """Simulates standard ungrounded LLM reviewer evaluating plan based solely on surface linguistic coherence."""

    def evaluate(self, plan_ir: PlanIR) -> JudgeVerdict:
        # Blind heuristics: if actions exist and have names, assume PASS with high confidence
        if not plan_ir.actions:
            return JudgeVerdict(verdict="FAIL", confidence=0.8, rationale="No actions scheduled.")

        # Surface plausibility
        return JudgeVerdict(
            verdict="PASS",
            confidence=0.90,
            rationale="Plan possesses syntactically valid action sequence aligning with stated goal.",
        )


class GroundedEpistemicJudge:
    """Evaluates plan via rigorous causal forward validation over 4-state epistemic lattice."""

    def __init__(self, validator: Optional[CausalValidator] = None):
        self.validator = validator or CausalValidator()

    def evaluate(self, plan_ir: PlanIR, registry: Optional[CapabilityRegistry] = None) -> JudgeVerdict:
        val_res = self.validator.validate_plan(plan_ir, registry=registry)
        
        if val_res.status == ValidationStatus.PASS:
            return JudgeVerdict(
                verdict="PASS",
                confidence=1.0,
                rationale="All preconditions and invariants causally verified on world state.",
            )
        elif val_res.status == ValidationStatus.UNKNOWN:
            return JudgeVerdict(
                verdict="UNKNOWN",
                confidence=0.5,
                blockers=val_res.unknown_facts,
                rationale=f"Plan contains {len(val_res.unknown_facts)} ungrounded UNKNOWN preconditions.",
            )
        else:
            return JudgeVerdict(
                verdict="FAIL",
                confidence=0.95,
                blockers=val_res.blocker_reasons + val_res.invariants_violated,
                rationale=f"Causal validation failed at step '{val_res.failed_step_id}'.",
            )


class DualJudgeEvaluator:
    """Dispatches both judges and computes multi-dimensional disagreement metrics."""

    def __init__(
        self,
        blind_judge: Optional[BlindJudge] = None,
        grounded_judge: Optional[GroundedEpistemicJudge] = None,
    ):
        self.blind_judge = blind_judge or BlindJudge()
        self.grounded_judge = grounded_judge or GroundedEpistemicJudge()

    def evaluate_plan(self, plan_ir: PlanIR, registry: Optional[CapabilityRegistry] = None) -> DualJudgeComparison:
        blind = self.blind_judge.evaluate(plan_ir)
        grounded = self.grounded_judge.evaluate(plan_ir, registry=registry)

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
