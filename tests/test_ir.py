"""Unit and adversarial tests for Canonical Plan IR and Provenance Tracking."""

import pytest
import time
from plan_mode.ir import (
    FactTruth,
    WitnessabilityStatus,
    SourceType,
    Provenance,
    WorldFact,
    PredicateCondition,
    HardConstraint,
    SuccessCriterion,
    ActionIR,
    PlanIR,
    render_markdown_view,
)


def test_fact_truth_lattice_values():
    assert FactTruth.VERIFIED_TRUE.value == "VERIFIED_TRUE"
    assert FactTruth.VERIFIED_FALSE.value == "VERIFIED_FALSE"
    assert FactTruth.UNKNOWN.value == "UNKNOWN"
    assert FactTruth.CONFLICT.value == "CONFLICT"


def test_world_fact_canonical_key():
    prov = Provenance(
        source_type=SourceType.OBSERVED_WORLD_STATE,
        source_id="probe_001",
        confidence=1.0,
        rationale="Direct observation from ls command",
    )
    fact = WorldFact(
        predicate="file_exists",
        args=["/tmp/data.csv"],
        truth=FactTruth.VERIFIED_TRUE,
        witnessability=WitnessabilityStatus.WITNESSABLE,
        provenance=prov,
    )
    assert fact.fact_key == "file_exists(/tmp/data.csv)"
    assert fact.is_fresh() is True


def test_world_fact_ttl_expiration():
    prov = Provenance(source_type=SourceType.OBSERVED_WORLD_STATE)
    fact = WorldFact(
        predicate="lock_acquired",
        args=["resource_A"],
        truth=FactTruth.VERIFIED_TRUE,
        ttl_seconds=0.01,
        created_at=time.time() - 0.05,  # 50ms ago
        provenance=prov,
    )
    assert fact.is_fresh() is False


def test_plan_ir_serialization_and_hashing():
    prov = Provenance(source_type=SourceType.USER_REQUIREMENT)
    fact1 = WorldFact(
        predicate="service_status",
        args=["nginx", "running"],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=prov,
    )
    fact2 = WorldFact(
        predicate="port_open",
        args=[80],
        truth=FactTruth.UNKNOWN,
        provenance=Provenance(source_type=SourceType.EXPLICIT_ASSUMPTION),
    )

    action1 = ActionIR(
        action_id="act_001",
        capability_name="system.check_port",
        parameters={"port": 80},
        preconditions=[
            PredicateCondition(predicate="service_status", args=["nginx", "running"], expected_truth=FactTruth.VERIFIED_TRUE)
        ],
        positive_effects=[
            PredicateCondition(predicate="port_open", args=[80], expected_truth=FactTruth.VERIFIED_TRUE)
        ],
        negative_effects=[],
        is_idempotent=True,
        provenance=prov,
    )

    plan = PlanIR(
        plan_id="plan_test_001",
        goal_description="Verify web server connectivity",
        initial_state=[fact1, fact2],
        actions=[action1],
        hard_constraints=[
            HardConstraint(
                constraint_id="hc_001",
                description="Port must be standard HTTP",
                condition=PredicateCondition(predicate="port_open", args=[80]),
                provenance=prov,
            )
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="sc_001",
                description="Server port verified open",
                condition=PredicateCondition(predicate="port_open", args=[80]),
            )
        ],
    )

    plan_hash = plan.compute_hash()
    assert isinstance(plan_hash, str)
    assert len(plan_hash) == 64  # SHA-256 hex digest

    # Verify deterministic hash
    assert plan.compute_hash() == plan_hash


def test_render_markdown_view():
    prov = Provenance(source_type=SourceType.USER_REQUIREMENT, rationale="User requested deploy")
    plan = PlanIR(
        plan_id="plan_md_001",
        goal_description="Deploy updated microservice",
        initial_state=[
            WorldFact(
                predicate="cluster_available",
                args=["k8s-prod"],
                truth=FactTruth.VERIFIED_TRUE,
                provenance=prov,
            )
        ],
        actions=[
            ActionIR(
                action_id="step_1",
                capability_name="k8s.apply_manifest",
                parameters={"manifest": "app.yaml"},
                preconditions=[
                    PredicateCondition(predicate="cluster_available", args=["k8s-prod"])
                ],
                positive_effects=[
                    PredicateCondition(predicate="deployment_ready", args=["app-v2"])
                ],
                provenance=prov,
            )
        ],
        success_criteria=[
            SuccessCriterion(
                criterion_id="sc_1",
                description="Deployment is ready",
                condition=PredicateCondition(predicate="deployment_ready", args=["app-v2"]),
            )
        ],
    )

    md = render_markdown_view(plan)
    assert "# Plan IR: `plan_md_001`" in md
    assert "Deploy updated microservice" in md
    assert "cluster_available(k8s-prod)" in md
    assert "k8s.apply_manifest" in md
    assert "deployment_ready(app-v2)" in md


def test_adversarial_ir_validation():
    """Ensure invalid confidence or missing mandatory fields raise ValidationError."""
    with pytest.raises(Exception):
        Provenance(source_type=SourceType.PLANNER_INFERENCE, confidence=1.5)  # > 1.0 invalid

    with pytest.raises(Exception):
        Provenance(source_type=SourceType.PLANNER_INFERENCE, confidence=-0.1)  # < 0.0 invalid


def test_plan_ir_hash_invariance_to_metadata_ordering():
    """Ensure hash is deterministic and ignores transient metadata changes."""
    prov = Provenance(source_type=SourceType.USER_REQUIREMENT)
    f1 = WorldFact(predicate="service", args=["web"], truth=FactTruth.VERIFIED_TRUE, provenance=prov)
    
    plan_a = PlanIR(
        plan_id="plan_iso",
        goal_description="test goal",
        initial_state=[f1],
        metadata={"run_counter": 1, "debug_tag": "test"}
    )
    plan_b = PlanIR(
        plan_id="plan_iso",
        goal_description="test goal",
        initial_state=[f1],
        metadata={"debug_tag": "test", "run_counter": 999}  # different metadata and order
    )
    assert plan_a.compute_hash() == plan_b.compute_hash()


def test_world_fact_multiple_args_canonical_key():
    prov = Provenance(source_type=SourceType.OBSERVED_WORLD_STATE)
    fact = WorldFact(
        predicate="distance_between",
        args=["locA", "locB", 42.5],
        truth=FactTruth.VERIFIED_TRUE,
        provenance=prov,
    )
    assert fact.fact_key == "distance_between(locA,locB,42.5)"
