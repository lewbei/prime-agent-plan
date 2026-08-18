"""Tests for Runtime Prototype Components: Event Chain, Structured Process Runner, Secret Scrubber, and Executor."""

import pytest
import time
from plan_mode.ir import (
    FactTruth,
    Provenance,
    SourceType,
    WorldFact,
    PredicateCondition,
    ActionIR,
    PlanIR,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry, ObservationVerifier, CompensationAction
from plan_mode.session import PlanningSession
from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType, LedgerTamperError
from plan_mode.runtime.secret_scrubber import SecretScrubber
from plan_mode.runtime.sandbox import ExecutionSandbox
from plan_mode.runtime.executor import ExecutionPlanManager, WitnessStatus, PreconditionFailedError


def test_evidence_ledger_chain_and_tamper_detection():
    ledger = EvidenceLedger(session_id="test_sess_001")
    
    r0 = ledger.append_record(LedgerEventType.SESSION_INIT, {"agent": "runtime_tcb"})
    assert r0.index == 0
    assert len(r0.record_hash) == 64
    
    r1 = ledger.append_record(LedgerEventType.PLAN_SUBMITTED, {"plan_id": "plan_01"})
    assert r1.index == 1
    assert r1.prev_hash == r0.record_hash
    
    r2 = ledger.append_record(LedgerEventType.ACTION_EXECUTED, {"action_id": "act_01", "exit_code": 0})
    assert r2.index == 2
    assert r2.prev_hash == r1.record_hash

    # Integrity verification
    assert ledger.verify_integrity() is True

    # Tamper with historical payload
    ledger.records[1].payload["plan_id"] = "hacked_plan_01"
    assert ledger.verify_integrity() is False


def test_secret_scrubber_redaction():
    scrubber = SecretScrubber()
    
    sample_text = (
        "Connected with AWS key AKIAIOSFODNN7EXAMPLE and github token ghp_1234567890abcdefghijklmnopqrstuvwx\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDcSemACt8x4iTMCda8Yhe3iZaWbvV5XKSTbuAn0M\n"
        "DB password: password = 'SuperSecretDbPassword123!'"
    )
    
    scrubbed = scrubber.scrub_text(sample_text)
    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
    assert "ghp_1234567890abcdefghijklmnopqrstuvwx" not in scrubbed
    assert "[REDACTED_SECRET" in scrubbed


def test_sandbox_argv_pipeline_execution():
    sandbox = ExecutionSandbox()
    
    # Pipeline: echo "hello world\nalpha\nbeta" | grep "hello"
    pipeline = [
        ["echo", "hello world\nalpha\nbeta"],
        ["grep", "hello"],
    ]
    
    res = sandbox.execute_argv_pipeline(pipeline, timeout_seconds=5.0)
    assert res.returncode == 0
    assert "hello world" in res.stdout
    assert "alpha" not in res.stdout
    assert res.timeout_exceeded is False


def test_sandbox_timeout_containment():
    sandbox = ExecutionSandbox()
    
    # Command that sleeps longer than timeout
    pipeline = [["sleep", "2"]]
    res = sandbox.execute_argv_pipeline(pipeline, timeout_seconds=0.2)
    assert res.timeout_exceeded is True


def test_execution_plan_manager_success():
    # Setup capability registry
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="system.echo",
            description="Echoes string",
            is_idempotent=True,
            input_schema={"text": {"type": "str", "required": True}},
            positive_effects=[PredicateCondition(predicate="echoed_text", args=["{text}"])],
            verifiers=[
                ObservationVerifier(
                    verifier_id="v_echo",
                    predicate="echoed_text",
                    target_args_mapping=["text"],
                    command_template=["echo", "$text"],
                    expected_output_pattern=r"hello_run",
                )
            ],
        )
    )

    prov = Provenance(source_type=SourceType.USER_REQUIREMENT)
    plan = PlanIR(
        plan_id="plan_exec_001",
        goal_description="Execute echo and witness result",
        initial_state=[
            WorldFact(predicate="system_ready", args=[], truth=FactTruth.VERIFIED_TRUE, provenance=prov)
        ],
        actions=[
            ActionIR(
                action_id="act_01",
                capability_name="system.echo",
                parameters={"text": "hello_run"},
                preconditions=[PredicateCondition(predicate="system_ready", args=[])],
                positive_effects=[PredicateCondition(predicate="echoed_text", args=["hello_run"])],
                provenance=prov,
            )
        ],
    )

    session = PlanningSession(session_id="sess_exec_001")
    session.submit_draft(plan)
    session.validate_candidate(1, reg)
    session.select_version(1)
    cert = session.authorize_selected(reg, policy_hash="policy_v1")
    session.start_execution(reg, policy_hash="policy_v1")

    ledger = EvidenceLedger(session_id=session.session_id)
    manager = ExecutionPlanManager(session=session, registry=reg, ledger=ledger)

    exec_result = manager.execute_authorized_plan(cert)
    assert exec_result.success is True
    assert len(exec_result.step_results) == 1
    assert exec_result.step_results[0].witness_status == WitnessStatus.WITNESSED_TRUE
    assert ledger.verify_integrity() is True
