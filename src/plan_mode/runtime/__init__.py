"""Runtime Subsystem Prototype Scaffolding: in-memory event chain, structured process runner, and secret scrubber."""

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
]
