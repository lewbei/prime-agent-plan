"""Tests for 4-State Fact Lattice, State Transitions, and Causal Validator."""

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
)
from plan_mode.registry import CapabilityRegistry, CapabilityEntry, ObservationVerifier
from plan_mode.epistemic_validator import (
    CausalValidator,
    EpistemicCausalValidator,
    PlanValidationResult,
    ValidationStatus,
    merge_fact_truth,
)


def test_lattice_truth_merge():
    # Identical
    assert merge_fact_truth(FactTruth.VERIFIED_TRUE, FactTruth.VERIFIED_TRUE) == FactTruth.VERIFIED_TRUE
    assert merge_fact_truth(FactTruth.VERIFIED_FALSE, FactTruth.VERIFIED_FALSE) == FactTruth.VERIFIED_FALSE
    
    # Knowledge upgrade
    assert merge_fact_truth(FactTruth.UNKNOWN, FactTruth.VERIFIED_TRUE) == FactTruth.VERIFIED_TRUE
    assert merge_fact_truth(FactTruth.UNKNOWN, FactTruth.VERIFIED_FALSE) == FactTruth.VERIFIED_FALSE
    
    # Conflict detection
    assert merge_fact_truth(FactTruth.VERIFIED_TRUE, FactTruth.VERIFIED_FALSE) == FactTruth.CONFLICT
    assert merge_fact_truth(FactTruth.CONFLICT, FactTruth.VERIFIED_TRUE) == FactTruth.CONFLICT


def test_causal_validator_simple_pass():
    prov = Provenance(source_type=SourceType.OBSERVED_WORLD_STATE)

    f1 = WorldFact(predicate="file_exists", args=["/tmp/src.txt"], truth=FactTruth.VERIFIED_TRUE, provenance=prov, metadata={"evidence_ref": "ev_src"})
    f2 = WorldFact(predicate="dest_clean", args=["/tmp/dest.txt"], truth=FactTruth.VERIFIED_TRUE, provenance=prov, metadata={"evidence_ref": "ev_dst"})

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.copy",
            description="Copy file",
            input_schema={"src": "str", "dst": "str"},
            positive_effects=[PredicateCondition(predicate="file_exists", args=["{dst}"])],
            negative_effects=[PredicateCondition(predicate="dest_clean", args=["{dst}"], expected_truth=FactTruth.VERIFIED_FALSE)],
            verifiers=[
                ObservationVerifier(verifier_id="v_exists", predicate="file_exists", target_args_mapping=["{dst}"]),
                ObservationVerifier(verifier_id="v_clean", predicate="dest_clean", target_args_mapping=["{dst}"]),
            ],
        )
    )

    act1 = ActionIR(
        action_id="step_1",
        capability_name="fs.copy",
        parameters={"src": "/tmp/src.txt", "dst": "/tmp/dest.txt"},
        preconditions=[
            PredicateCondition(predicate="file_exists", args=["/tmp/src.txt"]),
            PredicateCondition(predicate="dest_clean", args=["/tmp/dest.txt"]),
        ],
        positive_effects=[
            PredicateCondition(predicate="file_exists", args=["/tmp/dest.txt"])
        ],
        negative_effects=[
            PredicateCondition(predicate="dest_clean", args=["/tmp/dest.txt"], expected_truth=FactTruth.VERIFIED_FALSE)
        ],
        provenance=prov,
    )

    plan = PlanIR(
        plan_id="plan_pass_001",
        goal_description="Copy source file to destination",
        initial_state=[f1, f2],
        actions=[act1],
        hard_constraints=[],
        success_criteria=[],
    )

    validator = CausalValidator()
    result = validator.validate_plan(plan, registry=reg)

    assert result.status == ValidationStatus.PASS
    assert len(result.blocker_reasons) == 0
    final_fact = result.intermediate_states[-1]["file_exists(/tmp/dest.txt)"]
    assert final_fact.witnessability == WitnessabilityStatus.WITNESSABLE
    assert final_fact.truth == FactTruth.UNKNOWN
    assert final_fact.metadata.get("predicted_truth") == FactTruth.VERIFIED_TRUE.value


def test_causal_validator_unknown_precondition_yields_unknown():
    prov = Provenance(source_type=SourceType.PLANNER_INFERENCE)
    
    # Initial state is unknown for file_exists
    f1 = WorldFact(predicate="file_exists", args=["/tmp/src.txt"], truth=FactTruth.UNKNOWN, provenance=prov)
    
    act1 = ActionIR(
        action_id="step_1",
        capability_name="fs.copy",
        parameters={"src": "/tmp/src.txt", "dst": "/tmp/dest.txt"},
        preconditions=[
            PredicateCondition(predicate="file_exists", args=["/tmp/src.txt"]),
        ],
        positive_effects=[
            PredicateCondition(predicate="file_exists", args=["/tmp/dest.txt"])
        ],
        provenance=prov,
    )
    
    plan = PlanIR(
        plan_id="plan_unknown_001",
        goal_description="Attempt copy with unknown precondition",
        initial_state=[f1],
        actions=[act1],
        hard_constraints=[],
        success_criteria=[
            SuccessCriterion(
                criterion_id="sc_001",
                description="Dest file exists",
                condition=PredicateCondition(predicate="file_exists", args=["/tmp/dest.txt"]),
            )
        ],
    )
    
    validator = CausalValidator()
    result = validator.validate_plan(plan)
    
    assert result.status == ValidationStatus.UNKNOWN
    assert "file_exists(/tmp/src.txt)" in result.unknown_facts
    assert len(result.criteria_satisfied) == 0


def test_causal_validator_failed_precondition_yields_fail():
    prov = Provenance(source_type=SourceType.OBSERVED_WORLD_STATE)
    
    # Precondition is verified FALSE
    f1 = WorldFact(predicate="file_exists", args=["/tmp/src.txt"], truth=FactTruth.VERIFIED_FALSE, provenance=prov, metadata={"evidence_ref": "ev_src"})
    
    act1 = ActionIR(
        action_id="step_1",
        capability_name="fs.copy",
        parameters={"src": "/tmp/src.txt", "dst": "/tmp/dest.txt"},
        preconditions=[
            PredicateCondition(predicate="file_exists", args=["/tmp/src.txt"], expected_truth=FactTruth.VERIFIED_TRUE),
        ],
        positive_effects=[
            PredicateCondition(predicate="file_exists", args=["/tmp/dest.txt"])
        ],
        provenance=prov,
    )
    
    plan = PlanIR(
        plan_id="plan_fail_001",
        goal_description="Fail due to missing source file",
        initial_state=[f1],
        actions=[act1],
        hard_constraints=[],
        success_criteria=[
            SuccessCriterion(
                criterion_id="sc_001",
                description="Dest file exists",
                condition=PredicateCondition(predicate="file_exists", args=["/tmp/dest.txt"]),
            )
        ],
    )
    
    validator = CausalValidator()
    result = validator.validate_plan(plan)
    
    assert result.status == ValidationStatus.FAIL
    assert result.failed_step_id == "step_1"
    assert result.failed_predicate == "file_exists(/tmp/src.txt)"
    assert len(result.blocker_reasons) > 0


def test_causal_validator_hard_constraint_violation():
    prov = Provenance(source_type=SourceType.USER_REQUIREMENT)
    
    f1 = WorldFact(predicate="service_running", args=["web"], truth=FactTruth.VERIFIED_TRUE, provenance=prov, metadata={"evidence_ref": "ev_web"})
    
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="system.stop_service",
            description="Stop service",
            input_schema={"name": "str"},
            negative_effects=[PredicateCondition(predicate="service_running", args=["{name}"], expected_truth=FactTruth.VERIFIED_FALSE)],
            verifiers=[ObservationVerifier(verifier_id="v_stop", predicate="service_running", target_args_mapping=["{name}"])],
        )
    )

    # Action stops service
    act1 = ActionIR(
        action_id="step_1",
        capability_name="system.stop_service",
        parameters={"name": "web"},
        preconditions=[],
        negative_effects=[
            PredicateCondition(predicate="service_running", args=["web"], expected_truth=FactTruth.VERIFIED_FALSE)
        ],
        provenance=prov,
    )
    
    # Invariant requires service_running == TRUE
    hc = HardConstraint(
        constraint_id="hc_no_downtime",
        description="Web service must not be stopped",
        condition=PredicateCondition(predicate="service_running", args=["web"], expected_truth=FactTruth.VERIFIED_TRUE),
        provenance=prov,
    )
    
    plan = PlanIR(
        plan_id="plan_hc_violation",
        goal_description="Violate uptime constraint",
        initial_state=[f1],
        actions=[act1],
        hard_constraints=[hc],
    )
    
    validator = CausalValidator()
    result = validator.validate_plan(plan, registry=reg)
    
    assert result.status == ValidationStatus.FAIL
    assert "hc_no_downtime" in result.invariants_violated


def test_causal_validator_fact_ttl_decay_during_execution():
    prov = Provenance(source_type=SourceType.OBSERVED_WORLD_STATE)
    now = time.time()
    
    # Fact expired 10 seconds ago
    f1 = WorldFact(
        predicate="auth_token_valid",
        args=["session_1"],
        truth=FactTruth.VERIFIED_TRUE,
        ttl_seconds=5.0,
        created_at=now - 20.0,
        updated_at=now - 15.0,
        provenance=prov,
        metadata={"evidence_ref": "ev_auth"},
    )
    
    act1 = ActionIR(
        action_id="step_1",
        capability_name="api.fetch_data",
        preconditions=[
            PredicateCondition(predicate="auth_token_valid", args=["session_1"], expected_truth=FactTruth.VERIFIED_TRUE)
        ],
        provenance=prov,
    )
    
    plan = PlanIR(
        plan_id="plan_ttl_decay",
        goal_description="Use expired token",
        initial_state=[f1],
        actions=[act1],
    )
    
    validator = CausalValidator()
    result = validator.validate_plan(plan, current_time=now)
    
    # Decayed to UNKNOWN
    assert result.status == ValidationStatus.UNKNOWN
    assert "auth_token_valid(session_1)" in result.unknown_facts
