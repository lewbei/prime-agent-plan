# Epistemic Planning & Verification Runtime: Threat & Security Model

## 1. Executive Summary & Core Security Invariant

The Epistemic Planning & Verification Runtime operates under a strict **Zero-Trust Autonomous Execution** paradigm. Language model planners, diagnostic probing scripts, and external tool outputs are classified as **untrusted**. The runtime guarantees that **no action shall be executed against external infrastructure without a cryptographically bound, causally verified, and freshness-validated Authorization Certificate**.

Furthermore, every epistemic assertion, state transition, and execution effect is captured in a tamper-evident, append-only **Hash-Chained Evidence Ledger**.

---

## 2. System Assets & Protection Objectives

| Asset | Description | Integrity & Confidentiality Objective |
| :--- | :--- | :--- |
| **World State Model ($W_t$)** | Epistemic representation of facts with 4-state truth values (`VERIFIED_TRUE`, `VERIFIED_FALSE`, `UNKNOWN`, `CONFLICT`). | Prevent state poisoning and hallucinated preconditions. Ensure causal consistency. |
| **Capability Registry** | Declarative registry of typed actions, verifiers, and compensations. | Prevent execution of unauthorized or schema-mismatched operations. |
| **Authorization Certificate ($H_E$)** | Cryptographic HMAC binding plan hash, world state hash, registry hash, and policy hash. | Prevent replay attacks, execution of drifted plans, and unauthorized elevation. |
| **Evidence Ledger ($R_0 \dots R_n$)** | Append-only hash chain recording every probe, verification, transition, and execution event. | Guarantee auditability, non-repudiation, and post-mortem causal forensics. |
| **Secrets & Credentials** | API tokens, SSH keys, session cookies, database passwords. | Absolute redaction from all stdout/stderr streams, ledger entries, and prompt contexts. |

---

## 3. Adversary Model & Threat Vectors

We consider an active, adaptive adversary capable of:

1. **Untrusted LLM Planner Exploitation**:
   - Hallucinating preconditions ($P \in \text{Pre}$ marked as true when nonexistent).
   - Injecting malicious shell syntax (e.g. `; rm -rf /`, `$(cat /etc/passwd)`) into parameters.
   - Deceptive plan generation where destructive actions are disguised as benign read probes.
   - Epistemic overconfidence (claiming unverified facts are verified).

2. **Compromised / Hostile Probing & Tool Scripts**:
   - Side-effect-inducing probes masquerading as read-only diagnostic actions.
   - Exfiltration of environment variables through stderr/stdout or DNS tunneling.
   - Resource exhaustion via unbounded loops or memory allocation.

3. **Concurrency & Out-of-Band State Drift**:
   - Time-of-Check to Time-of-Use (TOCTOU) race conditions where world state changes between authorization and execution.
   - Zombie execution tasks running after certificate expiration.

---

## 4. Trust Boundaries & Architecture Classification

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRUSTED COMPUTING BASE (TCB)                     │
│                                                                        │
│  ┌─────────────────────────┐         ┌──────────────────────────────┐  │
│  │   Runtime Daemon &      │         │   Cryptographic Evidence     │  │
│  │  Concurrency Controller │         │     Append-Only Ledger       │  │
│  └───────────┬─────────────┘         └──────────────┬───────────────┘  │
│              │                                      │                  │
│  ┌───────────┴─────────────┐         ┌──────────────┴───────────────┐  │
│  │    Causal Validator &   │         │    Secret Scrubbing Engine   │  │
│  │  Authorization Engine   │         │    (High-Entropy Redactor)   │  │
│  └───────────┬─────────────┘         └──────────────┬───────────────┘  │
│              │                                      │                  │
│  ┌───────────┴──────────────────────────────────────┴───────────────┐  │
│  │       Isolated Execution Sandbox (argv pipelines, env sanitization)│  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────────────┼──────────────────────────────────┘
                                      │ Enforces Strict Boundaries
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    UNTRUSTED / ADVERSARIAL COMPONENTS                  │
│                                                                        │
│  ┌───────────────────────┐  ┌────────────────────┐  ┌───────────────┐  │
│  │  LLM Planner / Search │  │  Executor Agent    │  │ Diagnostic    │  │
│  │  (Generates PlanIR)   │  │  (Invokes tools)   │  │ Probing Scripts│  │
│  └───────────────────────┘  └────────────────────┘  └───────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Trusted Components
- **Kernel & Sandbox Subsystem**: Enforces strict execution without shell invocation (`execve` with explicit `argv` arrays), sterile environment variables, process isolation, and CPU/memory/time limits.
- **Runtime Daemon & Concurrency Controller**: Manages state transitions, optimistic lock verification, and certificate lifecycle.
- **Causal Validator**: Deterministic 4-state lattice evaluator ensuring monotonic fact updates and frame axioms.
- **Append-Only Evidence Ledger**: Cryptographically chained SHA-256 ledger recording all verifiable events.
- **Secret Scrubbing Engine**: Multi-stage regex and entropy scanner scrubbing sensitive data before persistence.

### 4.2 Untrusted Components
- **LLM Planner**: All generated plans, assumptions, and causal claims are treated as unverified hypotheses.
- **Executor Agent**: Must present valid certificates for every action; cannot execute raw commands directly.
- **Probing Scripts**: Executed strictly under read-only sandboxed constraints with bounded budgets.

---

## 5. Security Invariants & Guarantees

### 5.1 Bounded Containment Guarantees
- **No Shell Interpolation**: All command executions must use structured `argv_pipeline` (array of arguments). Shell execution (`/bin/sh -c`, `bash -c`, or `eval`) is strictly prohibited in runtime execution.
- **Piped Process Chaining**: When data flows between commands (e.g. `cat file | grep pattern`), execution occurs via native OS pipes (`os.pipe()`) where stdout of process $k$ is connected directly to stdin of process $k+1$.
- **Environment Sanitization**: The execution environment is scrubbed of parent environment variables, passing only an explicit whitelist (`PATH=/usr/bin:/bin`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`).
- **Resource Constraints**: Strict wall-clock timeouts (default: 10s per probe, 60s per action), memory caps, and process tree termination via `SIGKILL` on timeout.

### 5.2 Append-Only Evidence Ledger Integrity
Every state change, probe output, and execution result is recorded as an immutable ledger record $R_i$:
$$R_i = \text{SHA-256}(R_{i-1} \parallel \text{JSON}(\text{Payload}_i))$$
where $R_0 = \text{SHA-256}(\text{"GENESIS"} \parallel \text{SessionID})$.
Any tampering with historical records breaks the hash chain and triggers an immediate panic / lockdown.

### 5.3 Cryptographic Authorization Certificates
Execution requires an HMAC-SHA256 Authorization Certificate:
$$H_E = \text{HMAC}_{K_{\text{runtime}}}(H_{\text{Plan}} \parallel H_{\text{WorldState}} \parallel H_{\text{Registry}} \parallel H_{\text{Policy}} \parallel \text{ExpiresAt})$$
- The certificate is valid only if $\text{CurrentTime} \le \text{ExpiresAt}$.
- The certificate is rejected if the live world state hash $H_{\text{WorldState}}$ has drifted since certificate issuance.

### 5.4 Secret Redaction Rules
All diagnostic outputs, error messages, and payload logs are filtered through the `SecretScrubber`:
1. **High-Entropy Tokens**: Continuous alphanumeric strings with Shannon entropy $> 3.5$ and length $\ge 16$.
2. **Standard Secret Patterns**:
   - AWS Access Keys (`AKIA[0-9A-Z]{16}`)
   - GitHub Personal Access Tokens (`ghp_[0-9a-zA-Z]{36}`, `github_pat_[0-9a-zA-Z_]{82}`)
   - Bearer Tokens (`Bearer [a-zA-Z0-9_\-\.~+/]+=*`)
   - Private Keys (`-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----`)
   - Generic Password/API Key assignments (`(?:api_key|password|secret|token)\s*=\s*['"][^'"]+['"]`)
Scrubbed tokens are replaced with `[REDACTED_SECRET:<type>]`.
