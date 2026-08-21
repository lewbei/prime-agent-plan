"""Planning Session State Machine, Immutable Plan Versions, and Authorization Certificates."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from plan_mode.epistemic_validator import (
    CausalValidator,
    PlanValidationResult,
    ValidationStatus,
    normalize_trusted_snapshot,
)
from plan_mode.ir import PlanIR, SourceType, WorldFact
from plan_mode.registry import CapabilityRegistry


class SessionState(str, Enum):
    DRAFT = "DRAFT"
    IR_VALID = "IR_VALID"
    FEASIBILITY = "FEASIBILITY"
    SELECTED = "SELECTED"
    AUTHORIZED = "AUTHORIZED"
    EXECUTING = "EXECUTING"
    COMMITTED = "COMMITTED"
    DIAGNOSING = "DIAGNOSING"
    COMPENSATING = "COMPENSATING"
    ROLLED_BACK = "ROLLED_BACK"
    CONTAINMENT_FAILED = "CONTAINMENT_FAILED"
    FAILED = "FAILED"


VALID_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
    SessionState.DRAFT: {SessionState.IR_VALID, SessionState.FAILED},
    SessionState.IR_VALID: {SessionState.FEASIBILITY, SessionState.DRAFT, SessionState.FAILED},
    SessionState.FEASIBILITY: {SessionState.SELECTED, SessionState.DRAFT, SessionState.FAILED},
    SessionState.SELECTED: {SessionState.AUTHORIZED, SessionState.FEASIBILITY, SessionState.DRAFT, SessionState.FAILED},
    SessionState.AUTHORIZED: {SessionState.EXECUTING, SessionState.SELECTED, SessionState.DRAFT, SessionState.FAILED},
    SessionState.EXECUTING: {SessionState.COMMITTED, SessionState.DIAGNOSING, SessionState.COMPENSATING, SessionState.FAILED},
    SessionState.DIAGNOSING: {SessionState.COMPENSATING, SessionState.DRAFT, SessionState.FAILED},
    SessionState.COMPENSATING: {SessionState.ROLLED_BACK, SessionState.CONTAINMENT_FAILED, SessionState.FAILED},
    SessionState.ROLLED_BACK: {SessionState.DRAFT, SessionState.FAILED},
    SessionState.CONTAINMENT_FAILED: {SessionState.FAILED},
    SessionState.COMMITTED: {SessionState.DRAFT},
    SessionState.FAILED: {SessionState.DRAFT},
}


class InvalidStateTransitionError(Exception):
    pass


class CertificateExpiredError(Exception):
    pass


class StateDriftError(Exception):
    pass


class SignatureVerificationError(Exception):
    pass


class VersionNotFoundError(Exception):
    pass


class CommitGateError(Exception):
    def __init__(self, blockers: List[str]):
        self.blockers = blockers
        super().__init__("; ".join(blockers))


def compute_world_state_hash(facts: List[WorldFact]) -> str:
    """Deterministic semantic hash of the empirical state.

    Observation timestamps are intentionally excluded: a fresh re-observation
    of the same typed fact must remain semantically equivalent. TTL duration is
    included, while actual freshness is enforced by ``normalize_trusted_snapshot``
    immediately before execution. An expired fact therefore decays to UNKNOWN
    and changes the hash without making benign re-observation timestamps drift.
    """
    entries = []
    for fact in facts:
        entries.append({
            "predicate": fact.predicate,
            "args": fact.args,
            "truth": fact.truth.value,
            "witnessability": fact.witnessability.value,
            "ttl_seconds": fact.ttl_seconds,
        })
    serialized_entries = sorted(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in entries
    )
    return hashlib.sha256("\n".join(serialized_entries).encode("utf-8")).hexdigest()


class PlanVersion(BaseModel, frozen=True):
    version_number: int
    plan_ir: PlanIR
    validation_result: Optional[PlanValidationResult] = None
    plan_hash: str
    validation_world_state: Optional[List[WorldFact]] = None
    validation_world_state_hash: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class AuthorizationCertificate(BaseModel, frozen=True):
    certificate_id: str
    plan_id: str
    plan_version: int
    plan_hash: str
    world_state_hash: str
    registry_hash: str
    policy_hash: str
    isolation_policy_hash: str
    issued_at: float
    expires_at: float
    signature_hmac: str

    @classmethod
    def create(
        cls,
        plan_ir: PlanIR,
        world_facts: List[WorldFact],
        registry: CapabilityRegistry,
        policy_hash: str,
        isolation_policy_hash: str,
        secret_key: bytes,
        ttl_seconds: float = 60.0,
    ) -> "AuthorizationCertificate":
        now = time.time()
        expires_at = now + ttl_seconds
        cert_id = f"cert_{secrets.token_hex(8)}"
        plan_hash = plan_ir.compute_hash()
        ws_hash = compute_world_state_hash(world_facts)
        reg_hash = registry.compute_registry_hash()
        payload = (
            f"{plan_hash}:{ws_hash}:{reg_hash}:{policy_hash}:"
            f"{isolation_policy_hash}:{expires_at:.6f}"
        ).encode("utf-8")
        signature = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
        return cls(
            certificate_id=cert_id,
            plan_id=plan_ir.plan_id,
            plan_version=plan_ir.version,
            plan_hash=plan_hash,
            world_state_hash=ws_hash,
            registry_hash=reg_hash,
            policy_hash=policy_hash,
            isolation_policy_hash=isolation_policy_hash,
            issued_at=now,
            expires_at=expires_at,
            signature_hmac=signature,
        )

    def verify_signature(self, secret_key: bytes) -> bool:
        payload = (
            f"{self.plan_hash}:{self.world_state_hash}:{self.registry_hash}:{self.policy_hash}:"
            f"{self.isolation_policy_hash}:{self.expires_at:.6f}"
        ).encode("utf-8")
        expected = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        return (current_time if current_time is not None else time.time()) > self.expires_at


class PlanningSession(BaseModel):
    session_id: str
    current_state: SessionState = SessionState.DRAFT
    versions: Dict[int, PlanVersion] = Field(default_factory=dict)

    best_candidate_version: Optional[int] = None
    best_verified_version: Optional[int] = None
    best_unknown_version: Optional[int] = None
    authorized_version: Optional[int] = None
    committed_version: Optional[int] = None

    authorization_certificate: Optional[AuthorizationCertificate] = None
    authorized_policy_hash: Optional[str] = None
    authorized_isolation_policy_hash: Optional[str] = None
    last_execution_success: bool = False
    last_execution_world_state_hash: Optional[str] = None
    last_execution_version: Optional[int] = None
    secret_key: bytes = Field(default_factory=lambda: secrets.token_bytes(32))

    def transition_to(self, target_state: SessionState) -> None:
        if target_state not in VALID_TRANSITIONS.get(self.current_state, set()):
            raise InvalidStateTransitionError(
                f"Illegal state transition from '{self.current_state.value}' to '{target_state.value}'."
            )
        self.current_state = target_state

    def submit_draft(self, plan_ir: PlanIR) -> PlanVersion:
        if self.current_state not in (
            SessionState.DRAFT,
            SessionState.COMMITTED,
            SessionState.ROLLED_BACK,
            SessionState.FAILED,
        ):
            self.transition_to(SessionState.DRAFT)
        next_version = len(self.versions) + 1
        versioned = plan_ir.model_copy(deep=True, update={"version": next_version})
        version_obj = PlanVersion(
            version_number=next_version,
            plan_ir=versioned,
            validation_result=None,
            plan_hash=versioned.compute_hash(),
        )
        self.versions[next_version] = version_obj
        self.best_candidate_version = next_version
        self.transition_to(SessionState.IR_VALID)
        return version_obj

    def validate_candidate(
        self,
        version_number: int,
        registry: CapabilityRegistry,
        validator: Optional[CausalValidator] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
        current_time: Optional[float] = None,
    ) -> PlanValidationResult:
        if version_number not in self.versions:
            raise VersionNotFoundError(f"Version {version_number} not found.")
        if self.current_state != SessionState.IR_VALID:
            self.transition_to(SessionState.IR_VALID)
        version_obj = self.versions[version_number]
        active_validator = validator or CausalValidator()
        effective_now = current_time if current_time is not None else time.time()
        ttl_decay_policy = getattr(active_validator, "default_ttl_decay_to_unknown", True)

        normalized_map = (
            normalize_trusted_snapshot(
                observed_world_state,
                default_ttl_decay_to_unknown=ttl_decay_policy,
                now=effective_now,
            )
            if observed_world_state is not None
            else None
        )
        canonical_list = list(normalized_map.values()) if normalized_map is not None else None
        world_hash = compute_world_state_hash(canonical_list or [])
        result = active_validator.validate_plan(
            version_obj.plan_ir,
            registry=registry,
            observed_world_state=normalized_map,
            current_time=effective_now,
        )
        self.versions[version_number] = PlanVersion(
            version_number=version_obj.version_number,
            plan_ir=version_obj.plan_ir,
            validation_result=result,
            plan_hash=version_obj.plan_hash,
            validation_world_state=canonical_list,
            validation_world_state_hash=world_hash,
            created_at=version_obj.created_at,
        )
        if result.status == ValidationStatus.PASS:
            self.best_verified_version = version_number
        elif result.status == ValidationStatus.UNKNOWN:
            self.best_unknown_version = version_number
        self.transition_to(SessionState.FEASIBILITY)
        return result

    def select_version(self, version_number: int) -> None:
        if version_number not in self.versions:
            raise VersionNotFoundError(f"Version {version_number} not found.")
        version_obj = self.versions[version_number]
        if version_obj.validation_result is None or version_obj.validation_result.status != ValidationStatus.PASS:
            raise ValueError(f"Version {version_number} is not validated as PASS.")
        if self.current_state != SessionState.FEASIBILITY:
            self.transition_to(SessionState.FEASIBILITY)
        self.best_candidate_version = version_number
        self.transition_to(SessionState.SELECTED)

    def authorize_selected(
        self,
        registry: CapabilityRegistry,
        policy_hash: str,
        ttl_seconds: float = 60.0,
        isolation_policy_hash: Optional[str] = None,
    ) -> AuthorizationCertificate:
        if self.current_state != SessionState.SELECTED:
            raise InvalidStateTransitionError(
                f"Cannot authorize when in state '{self.current_state.value}' (expected SELECTED)."
            )
        assert self.best_candidate_version is not None
        version_obj = self.versions[self.best_candidate_version]
        trusted_facts = version_obj.validation_world_state or []

        if isolation_policy_hash is None:
            from plan_mode.runtime.isolation_identity import compute_isolation_policy_hash
            from plan_mode.runtime.sandbox import SecurityProfile

            isolation_policy_hash = compute_isolation_policy_hash(
                SecurityProfile.get_profile(SecurityProfile.STRICT)
            )

        certificate = AuthorizationCertificate.create(
            plan_ir=version_obj.plan_ir,
            world_facts=trusted_facts,
            registry=registry,
            policy_hash=policy_hash,
            isolation_policy_hash=isolation_policy_hash,
            secret_key=self.secret_key,
            ttl_seconds=ttl_seconds,
        )
        self.authorization_certificate = certificate
        self.authorized_version = self.best_candidate_version
        self.authorized_policy_hash = policy_hash
        self.authorized_isolation_policy_hash = isolation_policy_hash
        self.transition_to(SessionState.AUTHORIZED)
        return certificate

    def start_execution(
        self,
        registry: CapabilityRegistry,
        policy_hash: str,
        current_world_facts: Optional[List[WorldFact]] = None,
        current_time: Optional[float] = None,
    ) -> None:
        if self.current_state != SessionState.AUTHORIZED:
            raise InvalidStateTransitionError(
                f"Cannot start execution from state '{self.current_state.value}' (expected AUTHORIZED)."
            )
        certificate = self.authorization_certificate
        if certificate is None:
            raise SignatureVerificationError("Missing authorization certificate.")
        if not certificate.verify_signature(self.secret_key):
            raise SignatureVerificationError("Authorization certificate signature invalid or tampered.")
        if certificate.is_expired(current_time):
            raise CertificateExpiredError(
                f"Certificate {certificate.certificate_id} expired at {certificate.expires_at}."
            )
        if policy_hash != certificate.policy_hash or (
            self.authorized_policy_hash and policy_hash != self.authorized_policy_hash
        ):
            raise StateDriftError(
                f"Policy drift detected: policy hash '{policy_hash}' does not match "
                f"authorized policy '{certificate.policy_hash}'."
            )
        if (
            self.authorized_isolation_policy_hash
            and certificate.isolation_policy_hash != self.authorized_isolation_policy_hash
        ):
            raise StateDriftError("Isolation policy identity changed after authorization.")

        assert self.authorized_version is not None
        version_obj = self.versions[self.authorized_version]
        if version_obj.plan_ir.compute_hash() != certificate.plan_hash:
            raise StateDriftError("Plan semantics changed after authorization.")

        effective_now = current_time if current_time is not None else time.time()
        source_facts = current_world_facts
        if source_facts is None:
            validation_facts = version_obj.validation_world_state or []
            if any(f.ttl_seconds is not None for f in validation_facts):
                raise StateDriftError(
                    "Fresh current_world_facts are required because authorized world state contains TTL-bound facts."
                )
            source_facts = validation_facts

        normalized_map = normalize_trusted_snapshot(source_facts, now=effective_now)
        live_facts = list(normalized_map.values())
        if compute_world_state_hash(live_facts) != certificate.world_state_hash:
            raise StateDriftError(
                "World state drift or freshness decay detected between authorization and execution."
            )
        if registry.compute_registry_hash() != certificate.registry_hash:
            raise StateDriftError("Capability registry changed since certificate issuance.")

        self.last_execution_success = False
        self.last_execution_world_state_hash = None
        self.last_execution_version = None
        self.transition_to(SessionState.EXECUTING)

    def record_execution_result(self, success: bool, world_facts: List[WorldFact]) -> None:
        if self.current_state != SessionState.EXECUTING:
            raise InvalidStateTransitionError(
                f"Cannot record execution result from state '{self.current_state.value}'."
            )
        self.last_execution_success = bool(success)
        self.last_execution_world_state_hash = compute_world_state_hash(world_facts)
        self.last_execution_version = self.authorized_version

    def commit_execution(
        self,
        live_world_state: Optional[Dict[str, WorldFact] | List[WorldFact]] = None,
    ) -> None:
        if self.current_state != SessionState.EXECUTING:
            raise InvalidStateTransitionError(
                f"Cannot commit from state '{self.current_state.value}' (expected EXECUTING)."
            )

        blockers: List[str] = []
        if not self.last_execution_success:
            blockers.append("latest execution was not independently attested as successful")
        if self.last_execution_version != self.authorized_version:
            blockers.append("execution attestation is not bound to the authorized plan version")

        if live_world_state is None:
            blockers.append("live runtime world state is required for commit")
            facts: List[WorldFact] = []
        elif isinstance(live_world_state, dict):
            facts = list(live_world_state.values())
        else:
            facts = list(live_world_state)

        if live_world_state is not None:
            current_hash = compute_world_state_hash(facts)
            if self.last_execution_world_state_hash != current_hash:
                blockers.append("commit world state does not match the attested execution world state")

        if self.authorized_version is None:
            blockers.append("no authorized plan version is bound to this execution")
        else:
            plan = self.versions[self.authorized_version].plan_ir
            if self.authorization_certificate and plan.compute_hash() != self.authorization_certificate.plan_hash:
                blockers.append("authorized plan semantics changed before commit")
            fact_map = {fact.fact_key: fact for fact in facts}
            for criterion in plan.success_criteria:
                if not criterion.is_mandatory:
                    continue
                expected = criterion.condition
                fact = fact_map.get(expected.fact_key)
                if fact is None:
                    blockers.append(f"mandatory criterion '{criterion.criterion_id}' is not observed")
                    continue
                if fact.truth != expected.expected_truth:
                    blockers.append(
                        f"mandatory criterion '{criterion.criterion_id}' expected "
                        f"{expected.expected_truth.value} but observed {fact.truth.value}"
                    )
                    continue
                if fact.provenance.source_type != SourceType.OBSERVED_WORLD_STATE:
                    blockers.append(
                        f"mandatory criterion '{criterion.criterion_id}' is not grounded in observed runtime evidence"
                    )

        if blockers:
            raise CommitGateError(blockers)

        self.committed_version = self.authorized_version
        self.transition_to(SessionState.COMMITTED)
