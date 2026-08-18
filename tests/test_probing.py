"""Tests for Safe Value of Information (VOI) Probing Engine."""

import pytest
from plan_mode.ir import FactTruth
from plan_mode.probing import (
    DiagnosticProbe,
    ProbeCandidate,
    VOIProbingEngine,
    UnsafeProbeError,
)


@pytest.fixture
def probing_engine() -> VOIProbingEngine:
    engine = VOIProbingEngine()
    
    # Register safe read probes
    p1 = DiagnosticProbe(
        probe_id="probe_file_exists",
        target_predicate="file_exists",
        target_args=["/mnt/data/records.db"],
        argv_pipeline=[["test", "-f", "/mnt/data/records.db"]],
        execution_cost=0.01,
        side_effect_risk=0.0,
        permission_cost=0.0,
        expected_output_parser="exit_code_zero",
    )
    
    p2 = DiagnosticProbe(
        probe_id="probe_disk_space",
        target_predicate="free_space_bytes",
        target_args=["/mnt/data"],
        argv_pipeline=[["df", "--output=avail", "-B1", "/mnt/data"], ["tail", "-n", "1"]],
        execution_cost=0.05,
        side_effect_risk=0.0,
        permission_cost=0.0,
        expected_output_parser="integer",
    )
    
    engine.register_probe(p1)
    engine.register_probe(p2)
    return engine


def test_voi_ranking_and_selection(probing_engine: VOIProbingEngine):
    unknowns = ["file_exists(/mnt/data/records.db)", "free_space_bytes(/mnt/data)"]
    criticality = {
        "file_exists(/mnt/data/records.db)": 1.0,
        "free_space_bytes(/mnt/data)": 0.8,
    }
    
    candidates = probing_engine.rank_probes_for_unknowns(unknowns, plan_criticality_map=criticality)
    assert len(candidates) == 2
    # p1 has higher delta U (1.0) - cost (0.01) = 0.99 > 0.8 - 0.05 = 0.75
    assert candidates[0].probe.probe_id == "probe_file_exists"
    assert candidates[0].voi_score > candidates[1].voi_score

    selected = probing_engine.select_best_probes(unknowns, max_probes=1)
    assert len(selected) == 1
    assert selected[0].probe_id == "probe_file_exists"


def test_unsafe_probe_rejection():
    """Ensure probes with dangerous side effect risk or shell injection strings are rejected."""
    with pytest.raises(UnsafeProbeError):
        # side_effect_risk > 0.1 is rejected as unsafe diagnostic probe
        DiagnosticProbe(
            probe_id="probe_dangerous_cleanup",
            target_predicate="temp_cleaned",
            target_args=[],
            argv_pipeline=[["rm", "-rf", "/tmp/cache"]],
            side_effect_risk=0.5,
        )


def test_probe_output_parsing_exit_code(probing_engine: VOIProbingEngine):
    p = probing_engine.get_probe("probe_file_exists")
    
    # Returncode 0 -> VERIFIED_TRUE
    truth = probing_engine.parse_probe_output(p, stdout="", returncode=0)
    assert truth == FactTruth.VERIFIED_TRUE

    # Returncode 1 -> VERIFIED_FALSE
    truth = probing_engine.parse_probe_output(p, stdout="", returncode=1)
    assert truth == FactTruth.VERIFIED_FALSE


def test_probe_output_parsing_regex(probing_engine: VOIProbingEngine):
    regex_probe = DiagnosticProbe(
        probe_id="probe_service_status",
        target_predicate="service_status",
        target_args=["nginx"],
        argv_pipeline=[["systemctl", "is-active", "nginx"]],
        expected_output_parser="regex",
        parser_pattern=r"^active\s*$",
    )
    
    truth_match = probing_engine.parse_probe_output(regex_probe, stdout="active\n", returncode=0)
    assert truth_match == FactTruth.VERIFIED_TRUE

    truth_no_match = probing_engine.parse_probe_output(regex_probe, stdout="inactive (dead)\n", returncode=3)
    assert truth_no_match == FactTruth.VERIFIED_FALSE
