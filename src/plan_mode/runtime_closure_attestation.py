"""Runtime-issued observation attestations for commit-critical world facts."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import secrets
from typing import Any

from .runtime_closure_context import (
    ACTIVE_TRANSACTION_ID,
    ACTIVE_WORKSPACE_ID,
    canonical_json,
)

_OBSERVATION_ATTESTATION_KEY = secrets.token_bytes(32)


def _fact_payload(
    fact: Any,
    *,
    transaction_id: str,
    workspace_identity: str,
    certificate_id: str,
    plan_hash: str,
) -> dict[str, Any]:
    provenance = getattr(fact, "provenance", None)
    metadata = getattr(fact, "metadata", None)
    observed_at = metadata.get("observed_at") if isinstance(metadata, dict) else None
    return {
        "transaction_id": transaction_id,
        "workspace_identity": workspace_identity,
        "certificate_id": certificate_id,
        "plan_hash": plan_hash,
        "fact_key": fact.fact_key,
        "truth": fact.truth.value,
        "witnessability": fact.witnessability.value,
        "verifier_id": getattr(provenance, "source_id", None),
        "observed_at": observed_at,
    }


def _sign_payload(payload: dict[str, Any]) -> str:
    return hmac.new(
        _OBSERVATION_ATTESTATION_KEY,
        canonical_json(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _fact_semantic_digest(fact: Any) -> str:
    metadata = dict(getattr(fact, "metadata", None) or {})
    metadata.pop("runtime_attestation", None)
    metadata.pop("attestation_id", None)
    provenance = getattr(fact, "provenance", None)
    payload = {
        "fact_key": fact.fact_key,
        "truth": fact.truth.value,
        "witnessability": fact.witnessability.value,
        "updated_at": getattr(fact, "updated_at", None),
        "provenance": {
            "source_type": getattr(
                getattr(provenance, "source_type", None),
                "value",
                None,
            ),
            "source_id": getattr(provenance, "source_id", None),
        },
        "metadata": metadata,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _attest_fact(
    fact: Any,
    *,
    transaction_id: str,
    workspace_identity: str,
    certificate_id: str,
    plan_hash: str,
) -> None:
    metadata = getattr(fact, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        fact.metadata = metadata
    payload = _fact_payload(
        fact,
        transaction_id=transaction_id,
        workspace_identity=workspace_identity,
        certificate_id=certificate_id,
        plan_hash=plan_hash,
    )
    signature = _sign_payload(payload)
    attestation_id = hashlib.sha256(signature.encode("ascii")).hexdigest()[:24]
    metadata["runtime_attestation"] = {**payload, "signature": signature}
    metadata["attestation_id"] = attestation_id


def _verify_fact_attestation(
    fact: Any,
    *,
    transaction_id: str,
    workspace_identity: str,
    certificate_id: str,
    plan_hash: str,
) -> bool:
    metadata = getattr(fact, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    attestation = metadata.get("runtime_attestation")
    if not isinstance(attestation, dict):
        return False
    signature = attestation.get("signature")
    if not isinstance(signature, str) or not signature:
        return False
    expected_payload = _fact_payload(
        fact,
        transaction_id=transaction_id,
        workspace_identity=workspace_identity,
        certificate_id=certificate_id,
        plan_hash=plan_hash,
    )
    stored_payload = {
        key: attestation.get(key)
        for key in expected_payload
    }
    if stored_payload != expected_payload:
        return False
    return hmac.compare_digest(signature, _sign_payload(expected_payload))


def install_observation_attestation_closure() -> None:
    from .ir import SourceType
    from .runtime import executor as executor_mod
    from .session import CommitGateError, PlanningSession

    manager_cls = executor_mod.ExecutionPlanManager
    raw_execute = manager_cls.execute_authorized_plan
    if not getattr(raw_execute, "_observation_attestation_closure", False):
        def execute_authorized_plan(
            self,
            certificate,
            execution_backend=None,
            custom_action_handler=None,
        ):
            before = {
                key: _fact_semantic_digest(fact)
                for key, fact in self.live_world_state.items()
            }
            result = raw_execute(
                self,
                certificate,
                execution_backend=execution_backend,
                custom_action_handler=custom_action_handler,
            )
            transaction_id = ACTIVE_TRANSACTION_ID.get()
            workspace_identity = ACTIVE_WORKSPACE_ID.get()
            if transaction_id and workspace_identity:
                for key, fact in self.live_world_state.items():
                    provenance = getattr(fact, "provenance", None)
                    source_type = getattr(provenance, "source_type", None)
                    if source_type != SourceType.OBSERVED_WORLD_STATE:
                        continue
                    if before.get(key) == _fact_semantic_digest(fact):
                        continue
                    metadata = getattr(fact, "metadata", None)
                    if not isinstance(metadata, dict) or metadata.get("observed_at") is None:
                        continue
                    if not getattr(provenance, "source_id", None):
                        continue
                    _attest_fact(
                        fact,
                        transaction_id=transaction_id,
                        workspace_identity=workspace_identity,
                        certificate_id=certificate.certificate_id,
                        plan_hash=certificate.plan_hash,
                    )
                result.live_world_state = copy.deepcopy(self.live_world_state)
            return result

        execute_authorized_plan._observation_attestation_closure = True
        manager_cls.execute_authorized_plan = execute_authorized_plan

    raw_commit = PlanningSession.commit_execution
    if getattr(raw_commit, "_observation_attestation_closure", False):
        return

    def commit_execution(self, live_world_state=None) -> None:
        transaction_id = ACTIVE_TRANSACTION_ID.get()
        workspace_identity = ACTIVE_WORKSPACE_ID.get()
        certificate = self.authorization_certificate
        blockers: list[str] = []
        if not transaction_id or not workspace_identity or certificate is None:
            blockers.append("runtime observation attestation context is missing")
        if live_world_state is None:
            facts = []
        elif isinstance(live_world_state, dict):
            facts = list(live_world_state.values())
        else:
            facts = list(live_world_state)
        fact_map = {fact.fact_key: fact for fact in facts}

        if certificate is not None and self.authorized_version is not None:
            plan = self.versions[self.authorized_version].plan_ir
            for criterion in plan.success_criteria:
                if not criterion.is_mandatory:
                    continue
                fact = fact_map.get(criterion.condition.fact_key)
                if fact is None:
                    continue  # the canonical commit gate reports the missing fact
                if not _verify_fact_attestation(
                    fact,
                    transaction_id=transaction_id or "",
                    workspace_identity=workspace_identity or "",
                    certificate_id=certificate.certificate_id,
                    plan_hash=certificate.plan_hash,
                ):
                    blockers.append(
                        f"mandatory criterion '{criterion.criterion_id}' lacks a valid "
                        "runtime-issued observation attestation"
                    )
        if blockers:
            raise CommitGateError(blockers)
        raw_commit(self, live_world_state=live_world_state)

    commit_execution._observation_attestation_closure = True
    PlanningSession.commit_execution = commit_execution
