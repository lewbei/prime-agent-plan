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
    EpistemicCausalValidator,
    PlanValidationResult,
    ValidationStatus,
)
from plan_mode.ir import PlanIR, WorldFact
from plan_mode.registry import CapabilityRegistry


class SessionState(str, Enum):
    """Transactional session states for planning, verification, and execution."""
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
    FAILED = "FAILED"


# Allowed state transitions
VALID_TRANSITIONS: Dict[SessionState, Set[SessionState]] = {
    SessionState.DRAFT: {SessionState.IR_VALID, SessionState.FAILED},
    SessionState.IR_VALID: {SessionState.FEASIBILITY, SessionState.DRAFT, SessionState.FAILED},
    SessionState.FEASIBILITY: {SessionState.SELECTED, SessionState.DRAFT, SessionState.FAILED},
    SessionState.SELECTED: {SessionState.AUTHORIZED, SessionState.FEASIBILITY, SessionState.DRAFT, SessionState.FAILED},
    SessionState.AUTHORIZED: {SessionState.EXECUTING, SessionState.SELECTED, SessionState.DRAFT, SessionState.FAILED},
    SessionState.EXECUTING: {SessionState.COMMITTED, SessionState.DIAGNOSING, SessionState.COMPENSATING, SessionState.FAILED},
    SessionState.DIAGNOSING: {SessionState.COMPENSATING, SessionState.DRAFT, SessionState.FAILED},
    SessionState.COMPENSATING: {SessionState.ROLLED_BACK, SessionState.FAILED},
    SessionState.ROLLED_BACK: {SessionState.DRAFT, SessionState.FAILED},
    SessionState.COMMITTED: {SessionState.DRAFT},
    SessionState.FAILED: {SessionState.DRAFT},
}


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state machine transition is attempted."""
    pass


class CertificateExpiredError(Exception):
    """Raised when an authorization certificate has passed its TTL."""
    pass


class StateDriftError(Exception):
    """Raised when the world state has drifted from the authorized certificate state."""
    pass


class SignatureVerificationError(Exception):
    """Raised when an HMAC cryptographic signature check fails."""
    pass


class VersionNotFoundError(Exception):
    """Raised when a requested plan version is not in session history."""
    pass


def compute_world_state_hash(facts: List[WorldFact]) -> str:
    """Deterministic SHA-256 hash of a collection of world facts."""
    entries = sorted(
        [f"{f.fact_key}:{f.truth.value}:{f.witnessability.value}" for f in facts]
    )
    combined = "\n".join(entries)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class PlanVersion(BaseModel, frozen=True):
    """Immutable snapshot of a plan version and its validation outcome."""
    version_number: int
    plan_ir: PlanIR
    validation_result: Optional[PlanValidationResult] = None
    plan_hash: str
    created_at: float = Field(default_factory=time.time)


class AuthorizationCertificate(BaseModel, frozen=True):
    """Cryptographic authorization token granting execution rights."""
    certificate_id: str
    plan_id: str
    plan_version: int
    plan_hash: str
    world_state_hash: str
    registry_hash: str
    policy_hash: str
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
        secret_key: bytes,
        ttl_seconds: float = 60.0,
    ) -> AuthorizationCertificate:
        now = time.time()
        expires_at = now + ttl_seconds
        cert_id = f"cert_{secrets.token_hex(8)}"
        plan_hash = plan_ir.compute_hash()
        ws_hash = compute_world_state_hash(world_facts)
        reg_hash = registry.compute_registry_hash()

        # Binding: HMAC(PlanHash || WSHash || RegHash || PolicyHash || ExpiresAt)
        payload = f"{plan_hash}:{ws_hash}:{reg_hash}:{policy_hash}:{expires_at:.6f}".encode("utf-8")
        sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

        return cls(
            certificate_id=cert_id,
            plan_id=plan_ir.plan_id,
            plan_version=plan_ir.version,
            plan_hash=plan_hash,
            world_state_hash=ws_hash,
            registry_hash=reg_hash,
            policy_hash=policy_hash,
            issued_at=now,
            expires_at=expires_at,
            signature_hmac=sig,
        )

    def verify_signature(self, secret_key: bytes) -> bool:
        """Verify the HMAC signature of this certificate."""
        payload = f"{self.plan_hash}:{self.world_state_hash}:{self.registry_hash}:{self.policy_hash}:{self.expires_at:.6f}".encode("utf-8")
        expected_sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected_sig)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if certificate has exceeded its validity window."""
        now = current_time if current_time is not None else time.time()
        return now > self.expires_at


class PlanningSession(BaseModel):
    """Stateful runtime session managing plan versions and state machine."""
    session_id: str
    current_state: SessionState = SessionState.DRAFT
    versions: Dict[int, PlanVersion] = Field(default_factory=dict)
    
    # Version pointers
    best_candidate_version: Optional[int] = None
    best_verified_version: Optional[int] = None
    best_unknown_version: Optional[int] = None
    authorized_version: Optional[int] = None
    committed_version: Optional[int] = None
    
    authorization_certificate: Optional[AuthorizationCertificate] = None
    secret_key: bytes = Field(default_factory=lambda: secrets.token_bytes(32))

    def transition_to(self, target_state: SessionState) -> None:
        """Enforce state machine transition graph."""
        if target_state not in VALID_TRANSITIONS.get(self.current_state, set()):
            raise InvalidStateTransitionError(
                f"Illegal state transition from '{self.current_state.value}' to '{target_state.value}'."
            )
        self.current_state = target_state

    def submit_draft(self, plan_ir: PlanIR) -> PlanVersion:
        """Submit a new plan draft, generating an immutable PlanVersion."""
        if self.current_state not in (SessionState.DRAFT, SessionState.COMMITTED, SessionState.ROLLED_BACK, SessionState.FAILED):
            self.transition_to(SessionState.DRAFT)

        next_ver = len(self.versions) + 1
        plan_ir_versioned = plan_ir.model_copy(update={"version": next_ver})
        plan_hash = plan_ir_versioned.compute_hash()

        version_obj = PlanVersion(
            version_number=next_ver,
            plan_ir=plan_ir_versioned,
            validation_result=None,
            plan_hash=plan_hash,
        )
        self.versions[next_ver] = version_obj
        self.best_candidate_version = next_ver
        self.transition_to(SessionState.IR_VALID)
        return version_obj

    def validate_candidate(
        self,
        version_number: int,
        registry: CapabilityRegistry,
        validator: Optional[CausalValidator] = None,
        current_time: Optional[float] = None,
        observed_world_state: Optional[List[WorldFact] | Dict[str, WorldFact]] = None,
    ) -> PlanValidationResult:
        """Validate candidate plan version with CausalValidator."""
        if version_number not in self.versions:
            raise VersionNotFoundError(f"Version {version_number} not found.")

        if self.current_state != SessionState.IR_VALID:
            self.transition_to(SessionState.IR_VALID)

        v_obj = self.versions[version_number]
        val = validator or CausalValidator()
        result = val.validate_plan(
            v_obj.plan_ir,
            registry=registry,
            observed_world_state=observed_world_state,
            current_time=current_time,
        )

        updated_version = PlanVersion(
            version_number=v_obj.version_number,
            plan_ir=v_obj.plan_ir,
            validation_result=result,
            plan_hash=v_obj.plan_hash,
            created_at=v_obj.created_at,
        )
        self.versions[version_number] = updated_version

        if result.status == ValidationStatus.PASS:
            self.best_verified_version = version_number
        elif result.status == ValidationStatus.UNKNOWN:
            self.best_unknown_version = version_number

        self.transition_to(SessionState.FEASIBILITY)
        return result

    def select_version(self, version_number: int) -> None:
        """Select a validated plan version for execution."""
        if version_number not in self.versions:
            raise VersionNotFoundError(f"Version {version_number} not found.")

        v_obj = self.versions[version_number]
        if v_obj.validation_result is None or v_obj.validation_result.status != ValidationStatus.PASS:
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
    ) -> AuthorizationCertificate:
        """Issue cryptographic HMAC authorization certificate for selected plan."""
        if self.current_state != SessionState.SELECTED:
            raise InvalidStateTransitionError(
                f"Cannot authorize when in state '{self.current_state.value}' (expected SELECTED)."
            )

        assert self.best_candidate_version is not None
        v_obj = self.versions[self.best_candidate_version]

        cert = AuthorizationCertificate.create(
            plan_ir=v_obj.plan_ir,
            world_facts=v_obj.plan_ir.initial_state,
            registry=registry,
            policy_hash=policy_hash,
            secret_key=self.secret_key,
            ttl_seconds=ttl_seconds,
        )
        self.authorization_certificate = cert
        self.authorized_version = self.best_candidate_version
        self.transition_to(SessionState.AUTHORIZED)
        return cert

    def start_execution(
        self,
        registry: CapabilityRegistry,
        policy_hash: str,
        current_world_facts: Optional[List[WorldFact]] = None,
        current_time: Optional[float] = None,
    ) -> None:
        """Verify certificate validity and state drift before transitioning to EXECUTING."""
        if self.current_state != SessionState.AUTHORIZED:
            raise InvalidStateTransitionError(
                f"Cannot start execution from state '{self.current_state.value}' (expected AUTHORIZED)."
            )

        cert = self.authorization_certificate
        if cert is None:
            raise SignatureVerificationError("Missing authorization certificate.")

        # 1. Signature check
        if not cert.verify_signature(self.secret_key):
            raise SignatureVerificationError("Authorization certificate signature invalid or tampered.")

        # 2. Expiration check
        if cert.is_expired(current_time):
            raise CertificateExpiredError(f"Certificate {cert.certificate_id} expired at {cert.expires_at}.")

        # 3. State drift check
        assert self.authorized_version is not None
        v_obj = self.versions[self.authorized_version]
        live_facts = current_world_facts if current_world_facts is not None else v_obj.plan_ir.initial_state
        live_ws_hash = compute_world_state_hash(live_facts)

        if live_ws_hash != cert.world_state_hash:
            raise StateDriftError(
                f"World state drift detected! Authorized hash {cert.world_state_hash} != live hash {live_ws_hash}"
            )

        # 4. Registry drift check
        live_reg_hash = registry.compute_registry_hash()
        if live_reg_hash != cert.registry_hash:
            raise StateDriftError("Capability registry changed since certificate issuance.")

        self.transition_to(SessionState.EXECUTING)

    def commit_execution(self) -> None:
        """Mark plan execution as successfully committed."""
        if self.current_state != SessionState.EXECUTING:
            raise InvalidStateTransitionError(
                f"Cannot commit from state '{self.current_state.value}' (expected EXECUTING)."
            )
        self.committed_version = self.authorized_version
        self.transition_to(SessionState.COMMITTED)
