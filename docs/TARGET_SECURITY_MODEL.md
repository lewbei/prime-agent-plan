# Epistemic Planning & Verification Runtime: Security Model & Isolation Architecture

> [!NOTE]
> **PHASE 4 IMPLEMENTATION COMPLETE & VERIFIED**
> This document describes the **hardened security architecture** enforced by the runtime.
> **Current Codebase Status (Phases 0–4 Verified):**
> - `sandbox.py` implements **Real Linux Namespace & Container Isolation** via Bubblewrap (`bwrap`), POSIX resource limits (`RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NPROC`, `RLIMIT_FSIZE`), network default-deny policies, filesystem workspace jails, path traversal & symlink escape defenses, output size caps, and secret scrubbing.
> - `transaction.py` provides transactional saga compensation, pre-dispatch recovery recording, and fail-closed `CONTAINMENT_FAILED` handling.
> - `ledger.py` is an **in-memory integrity-linked SHA-256 event hash chain** providing chronological auditability for all runtime actions and observations.
> - `session.py` provides HMAC cryptographic authorization certificates strictly bound to canonical PlanIR hashes, trusted validation snapshots, and registry policies.

---

## 1. Executive Summary & Design Principles (Target)

The target architecture is designed around an **epistemic verification runtime** where language model planners, diagnostic probing scripts, and external tool outputs are classified as **untrusted components**.

The target design aims to enforce:
1. **No unverified execution**: Actions must not execute against infrastructure without causally verified preconditions and valid authorization.
2. **Event traceability**: State transitions and observation results are recorded in an integrity-linked event chain.
3. **Structured execution**: Tool and capability commands use structured `argv` arrays rather than shell interpolation.

---

## 2. System Assets & Protection Objectives

| Asset | Description | Target Protection Objective | Current Implementation Status |
| :--- | :--- | :--- | :--- |
| **World State Model ($W_t$)** | Epistemic representation with 4-state truth values (`VERIFIED_TRUE`, `VERIFIED_FALSE`, `UNKNOWN`, `CONFLICT`). | Prevent state poisoning and hallucinated preconditions. | `PROTOTYPE` (4-state lattice defined in `ir.py`; forward simulation in `epistemic_validator.py`). |
| **Capability Registry** | Declarative registry of typed actions, verifiers, and compensations. | Prevent execution of unauthorized or schema-mismatched operations. | `PROTOTYPE` (Schema typing in `registry.py`; full precondition/effect enforcement pending Phase 1/2). |
| **Authorization Certificate ($H_E$)** | HMAC binding plan hash, world state hash, registry hash, and policy hash. | Prevent execution of drifted plans or expired authorizations. | `PROTOTYPE` (In-process HMAC verification in `session.py`; live drift integration pending Phase 3). |
| **Evidence Event Chain** | Integrity-linked hash chain recording probe, verification, transition, and execution events. | Auditability and in-process tamper-detection. | `PROTOTYPE` (In-memory SHA-256 event chain in `ledger.py`; disk persistence / anchoring pending). |
| **Secrets & Credentials** | API tokens, SSH keys, session cookies, database passwords. | Redaction from stdout/stderr streams before persistence. | `IMPLEMENTED` (Regex and high-entropy scrubber in `secret_scrubber.py`). |

---

## 3. Adversary Model & Threat Vectors (Target Analysis)

The target architecture addresses these primary risk vectors:

1. **Untrusted LLM Planner Hallucinations**:
   - Hallucinated preconditions ($P \in \text{Pre}$ asserted without evidence).
   - Arbitrary shell syntax injection via parameter strings.
   - Deceptive plans where destructive actions are presented as benign read probes.

2. **Unverified Probing & Tool Scripts**:
   - Side-effect-inducing probes masquerading as read-only diagnostic actions.
   - Credential leakage in tool output streams.
   - Unbounded execution time.

3. **Concurrency & State Drift**:
   - Time-of-Check to Time-of-Use (TOCTOU) discrepancies where world state changes between planning and execution.

---

## 4. Trust Boundaries & Component Classification

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TARGET TRUSTED COMPUTING BASE (TCB)                  │
│                                                                        │
│  ┌─────────────────────────┐         ┌──────────────────────────────┐  │
│  │   Runtime Controller    │         │   In-Memory Event Chain      │  │
│  │  (Optimistic Concurrency)│        │   (Integrity-Linked Records) │  │
│  └───────────┬─────────────┘         └──────────────┬───────────────┘  │
│              │                                      │                  │
│  ┌───────────┴─────────────┐         ┌──────────────┴───────────────┐  │
│  │    Causal Validator &   │         │    Secret Scrubbing Engine   │  │
│  │  Authorization Engine   │         │    (Pattern-Based Redactor)  │  │
│  └───────────┬─────────────┘         └──────────────┬───────────────┘  │
│              │                                      │                  │
│  ┌───────────┴──────────────────────────────────────┴───────────────┐  │
│  │   Structured Process Runner (argv pipelines, env sanitization)   │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────────┼──────────────────────────────────┘
                                      │ Controlled Dispatch
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   UNTRUSTED INPUTS & MODEL OUTPUTS                     │
│                                                                        │
│  ┌───────────────────────┐  ┌────────────────────┐  ┌───────────────┐  │
│  │  LLM Planner          │  │  Proposer Agent    │  │ Untrusted     │  │
│  │  (Generates PlanIR)   │  │  (Candidate Plans) │  │ Tool Outputs  │  │
│  └───────────────────────┘  └────────────────────┘  └───────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Status:
- **Structured Process Runner (`sandbox.py`)**: Executes explicit `argv` argument vectors with native OS pipes (`stdout -> stdin`) and stripped environment variables. *Does not provide kernel isolation or resource sandboxing.*
- **Event Chain (`ledger.py`)**: Maintains a linked SHA-256 hash sequence ($R_i = \text{SHA256}(R_{i-1} \parallel \text{Payload}_i)$) to detect accidental in-memory modifications. *Does not prevent adversarial memory modification by a compromised host process.*
- **Secret Scrubber (`secret_scrubber.py`)**: Applies pattern matching to sanitize tokens and high-entropy strings from stdout/stderr.
- **Causal Validator (`epistemic_validator.py`)**: Forward-simulates 4-state fact transitions to verify plan feasibility.
