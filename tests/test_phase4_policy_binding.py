"""Phase 4 isolation-policy authorization binding tests."""
from __future__ import annotations

from plan_mode.epistemic_validator import ValidationStatus
from plan_mode.ir import ActionIR, PlanIR, Provenance, SourceType
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.runtime import EvidenceLedger, TransactionOutcome, TransactionalExecutionManager
from plan_mode.runtime.isolation_identity import compute_isolation_policy_hash
from plan_mode.runtime.sandbox import ExecutionSandbox, SecurityProfile
from plan_mode.session import PlanningSession


def _plan() -> PlanIR:
    return PlanIR(
        plan_id="isolation-policy-binding",
        goal_description="Bind execution privileges",
        actions=[
            ActionIR(
                action_id="a1",
                capability_name="noop",
                provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE),
            )
        ],
    )


def _selected_session(registry: CapabilityRegistry) -> tuple[PlanningSession, str]:
    session = PlanningSession(session_id="isolation-policy-binding")
    session.submit_draft(_plan())
    assert session.validate_candidate(1, registry, observed_world_state=[]).status == ValidationStatus.PASS
    session.select_version(1)
    return session, registry.compute_registry_hash()


def test_default_authorization_binds_strict_isolation_policy():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    session, policy_hash = _selected_session(registry)
    cert = session.authorize_selected(registry, policy_hash=policy_hash)

    expected = compute_isolation_policy_hash(SecurityProfile.get_profile(SecurityProfile.STRICT))
    assert cert.isolation_policy_hash == expected
    assert session.authorized_isolation_policy_hash == expected
    assert cert.verify_signature(session.secret_key) is True


def test_tampering_isolation_privilege_invalidates_certificate_signature():
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    session, policy_hash = _selected_session(registry)
    cert = session.authorize_selected(registry, policy_hash=policy_hash)

    tampered = cert.model_copy(update={"isolation_policy_hash": "0" * 64})
    assert tampered.verify_signature(session.secret_key) is False


def test_runtime_cannot_upgrade_strict_authorization_to_network_enabled_sandbox(tmp_path):
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    session, policy_hash = _selected_session(registry)
    cert = session.authorize_selected(registry, policy_hash=policy_hash)
    session.start_execution(registry, policy_hash=policy_hash, current_world_facts=[])

    network_policy = SecurityProfile.get_profile(SecurityProfile.NETWORK_ALLOWED).model_copy(
        update={"workspace_dir": str(tmp_path)}
    )
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy_hash,
        sandbox=ExecutionSandbox(network_policy),
    )
    ledger = manager.ledger
    summary = manager.execute_and_finalize(cert)

    assert summary.outcome == TransactionOutcome.FAILED
    assert any("isolation policy drift" in blocker.lower() for blocker in summary.commit_blockers)
    assert not any(record.event_type.value == "ACTION_DISPATCHED" for record in ledger.records)


def test_network_enabled_profile_must_be_explicitly_authorized(tmp_path):
    registry = CapabilityRegistry()
    registry.register(CapabilityEntry(name="noop", description="No-op", executor_command_template=["true"]))
    session, policy_hash = _selected_session(registry)
    network_template = SecurityProfile.get_profile(SecurityProfile.NETWORK_ALLOWED)
    network_hash = compute_isolation_policy_hash(network_template)
    cert = session.authorize_selected(
        registry,
        policy_hash=policy_hash,
        isolation_policy_hash=network_hash,
    )
    session.start_execution(registry, policy_hash=policy_hash, current_world_facts=[])

    runtime_policy = network_template.model_copy(update={"workspace_dir": str(tmp_path)})
    manager = TransactionalExecutionManager(
        session=session,
        registry=registry,
        ledger=EvidenceLedger(session_id=session.session_id),
        observed_world_state=[],
        policy_hash=policy_hash,
        sandbox=ExecutionSandbox(runtime_policy),
    )
    assert manager.isolation_policy_hash == cert.isolation_policy_hash
