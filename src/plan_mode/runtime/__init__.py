"""Runtime Subsystem Prototype Scaffolding: event chain, structured execution, attestation, and transaction control."""

from plan_mode.runtime.ledger import EvidenceLedger, LedgerEventType, LedgerRecord, LedgerTamperError
from plan_mode.runtime.secret_scrubber import SecretScrubber
from plan_mode.runtime.sandbox import ExecutionSandbox, SandboxExecutionResult
from plan_mode.runtime.executor import (
    ExecutionPlanManager,
    WitnessStatus,
    ExecutionSummary,
    ExecutionBackend,
    PreconditionFailedError,
    ExecutionContractMissingError,
)
from plan_mode.runtime.transaction import (
    TransactionalExecutionManager,
    TransactionOutcome,
    TransactionSummary,
    CompensationResult,
)

__all__ = [
    "EvidenceLedger",
    "LedgerEventType",
    "LedgerRecord",
    "LedgerTamperError",
    "SecretScrubber",
    "ExecutionSandbox",
    "SandboxExecutionResult",
    "ExecutionPlanManager",
    "WitnessStatus",
    "ExecutionSummary",
    "ExecutionBackend",
    "PreconditionFailedError",
    "ExecutionContractMissingError",
    "TransactionalExecutionManager",
    "TransactionOutcome",
    "TransactionSummary",
    "CompensationResult",
]
