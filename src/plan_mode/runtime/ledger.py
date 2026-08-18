"""Cryptographic Append-Only Hash-Chained Evidence Ledger."""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LedgerTamperError(Exception):
    """Raised when ledger integrity verification fails due to record tampering or hash breakage."""
    pass


class LedgerEventType(str, Enum):
    """Categorical event types recorded in the evidence ledger."""
    SESSION_INIT = "SESSION_INIT"
    PLAN_SUBMITTED = "PLAN_SUBMITTED"
    PLAN_VALIDATED = "PLAN_VALIDATED"
    PROBE_DISPATCHED = "PROBE_DISPATCHED"
    PROBE_COMPLETED = "PROBE_COMPLETED"
    ACTION_AUTHORIZED = "ACTION_AUTHORIZED"
    PRECHECK_EVALUATED = "PRECHECK_EVALUATED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    POSTCHECK_WITNESSED = "POSTCHECK_WITNESSED"
    COMPENSATION_TRIGGERED = "COMPENSATION_TRIGGERED"
    COMPENSATION_EXECUTED = "COMPENSATION_EXECUTED"
    PLAN_COMMITTED = "PLAN_COMMITTED"
    PLAN_ABORTED = "PLAN_ABORTED"


class LedgerRecord(BaseModel):
    """Single immutable block in the append-only ledger chain."""
    index: int
    prev_hash: str
    timestamp: float
    event_type: LedgerEventType
    payload: Dict[str, Any]
    record_hash: str

    @staticmethod
    def calculate_hash(
        index: int,
        prev_hash: str,
        timestamp: float,
        event_type: str,
        payload: Dict[str, Any],
    ) -> str:
        """Deterministic SHA-256 hash of block contents."""
        block = {
            "index": index,
            "prev_hash": prev_hash,
            "timestamp": f"{timestamp:.6f}",
            "event_type": event_type,
            "payload": payload,
        }
        serialized = json.dumps(block, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class EvidenceLedger(BaseModel):
    """Append-only evidence ledger guaranteeing causal auditability and tamper-detection."""
    session_id: str
    records: List[LedgerRecord] = Field(default_factory=list)

    @property
    def genesis_hash(self) -> str:
        """Deterministic genesis anchor for the session."""
        raw = f"GENESIS:{self.session_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def append_record(
        self,
        event_type: LedgerEventType,
        payload: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> LedgerRecord:
        """Append a new verified record to the hash chain."""
        idx = len(self.records)
        prev_hash = self.records[-1].record_hash if self.records else self.genesis_hash
        ts = timestamp if timestamp is not None else time.time()
        rec_hash = LedgerRecord.calculate_hash(
            index=idx,
            prev_hash=prev_hash,
            timestamp=ts,
            event_type=event_type.value,
            payload=payload,
        )

        record = LedgerRecord(
            index=idx,
            prev_hash=prev_hash,
            timestamp=ts,
            event_type=event_type,
            payload=payload,
            record_hash=rec_hash,
        )
        self.records.append(record)
        return record

    def verify_integrity(self) -> bool:
        """Verify unbroken hash chain from genesis to head."""
        expected_prev = self.genesis_hash
        for idx, rec in enumerate(self.records):
            if rec.index != idx:
                return False
            if rec.prev_hash != expected_prev:
                return False
            computed = LedgerRecord.calculate_hash(
                index=rec.index,
                prev_hash=rec.prev_hash,
                timestamp=rec.timestamp,
                event_type=rec.event_type.value,
                payload=rec.payload,
            )
            if rec.record_hash != computed:
                return False
            expected_prev = rec.record_hash
        return True
