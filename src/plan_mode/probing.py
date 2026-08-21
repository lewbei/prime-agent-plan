"""Safe Value of Information (VOI) Probing Engine with argv Pipeline Validation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from plan_mode.fact_identity import canonical_fact_identity
from plan_mode.ir import FactTruth


class UnsafeProbeError(Exception):
    """Raised when a diagnostic probe violates safety invariants or contains dangerous side effects."""
    pass


class DiagnosticProbe(BaseModel):
    """Declarative read-only diagnostic probe using structured argv pipelines."""
    probe_id: str
    target_predicate: str
    target_args: List[Any] = Field(default_factory=list)
    argv_pipeline: List[List[str]] = Field(default_factory=list)
    execution_cost: float = Field(default=0.01, ge=0.0)
    side_effect_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    permission_cost: float = Field(default=0.0, ge=0.0)
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    expected_output_parser: str = "exit_code_zero"
    parser_pattern: Optional[str] = None

    @model_validator(mode="after")
    def validate_safety_boundaries(self) -> DiagnosticProbe:
        if self.side_effect_risk > 0.1:
            raise UnsafeProbeError(
                f"Probe '{self.probe_id}' has side_effect_risk={self.side_effect_risk} > 0.1. "
                "Diagnostic probes must be strictly read-only."
            )
        if not self.argv_pipeline:
            raise UnsafeProbeError(f"Probe '{self.probe_id}' must define at least one argv stage.")
        for stage in self.argv_pipeline:
            if not stage or not stage[0]:
                raise UnsafeProbeError(f"Probe '{self.probe_id}' contains an empty command stage.")
        return self

    @property
    def fact_key(self) -> str:
        """Canonical typed key matching ``WorldFact.fact_key`` exactly."""
        return canonical_fact_identity(self.target_predicate, self.args_normalized)

    @property
    def args_normalized(self) -> List[Any]:
        return self.target_args


class ProbeCandidate(BaseModel):
    """Candidate probe evaluated under Value of Information (VOI) ranking."""
    probe: DiagnosticProbe
    expected_utility_delta: float
    voi_score: float


class VOIProbingEngine:
    """Calculates VOI and dispatches sandboxed diagnostic probes to resolve UNKNOWN state."""

    def __init__(self):
        self.probes: Dict[str, DiagnosticProbe] = {}

    def register_probe(self, probe: DiagnosticProbe) -> None:
        self.probes[probe.probe_id] = probe

    def get_probe(self, probe_id: str) -> DiagnosticProbe:
        if probe_id not in self.probes:
            raise KeyError(f"Probe '{probe_id}' not found.")
        return self.probes[probe_id]

    def rank_probes_for_unknowns(
        self,
        unknown_facts: List[str],
        plan_criticality_map: Optional[Dict[str, float]] = None,
    ) -> List[ProbeCandidate]:
        criticality = plan_criticality_map or {}
        candidates: List[ProbeCandidate] = []

        for p in self.probes.values():
            key = p.fact_key
            if key in unknown_facts:
                delta_u = criticality.get(key, 1.0)
                voi = delta_u - p.execution_cost - p.side_effect_risk - p.permission_cost
                candidates.append(
                    ProbeCandidate(
                        probe=p,
                        expected_utility_delta=delta_u,
                        voi_score=round(voi, 4),
                    )
                )

        candidates.sort(key=lambda c: c.voi_score, reverse=True)
        return candidates

    def select_best_probes(
        self,
        unknown_facts: List[str],
        max_probes: int = 5,
        min_voi_threshold: float = 0.0,
    ) -> List[DiagnosticProbe]:
        ranked = self.rank_probes_for_unknowns(unknown_facts)
        selected: List[DiagnosticProbe] = []
        for cand in ranked:
            if cand.voi_score >= min_voi_threshold and len(selected) < max_probes:
                selected.append(cand.probe)
        return selected

    def parse_probe_output(
        self,
        probe: DiagnosticProbe,
        stdout: str,
        returncode: int,
    ) -> FactTruth:
        parser = probe.expected_output_parser

        if parser == "exit_code_zero":
            return FactTruth.VERIFIED_TRUE if returncode == 0 else FactTruth.VERIFIED_FALSE

        if parser == "regex":
            if not probe.parser_pattern:
                return FactTruth.UNKNOWN
            match = re.search(probe.parser_pattern, stdout.strip())
            return FactTruth.VERIFIED_TRUE if match else FactTruth.VERIFIED_FALSE

        if parser == "non_empty":
            return FactTruth.VERIFIED_TRUE if stdout.strip() else FactTruth.VERIFIED_FALSE

        if parser == "integer":
            cleaned = stdout.strip()
            if cleaned.isdigit():
                return FactTruth.VERIFIED_TRUE if int(cleaned) > 0 else FactTruth.VERIFIED_FALSE
            return FactTruth.VERIFIED_FALSE

        if parser == "json":
            try:
                data = json.loads(stdout)
                return FactTruth.VERIFIED_TRUE if data else FactTruth.VERIFIED_FALSE
            except Exception:
                return FactTruth.VERIFIED_FALSE

        return FactTruth.UNKNOWN
