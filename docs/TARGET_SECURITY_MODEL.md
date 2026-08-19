# Epistemic Planning & Verification Runtime — Security Model

> **Current implementation status:** Phases 0–5 are implemented and gate-tested. Phase 6 empirical benchmark evaluation remains pending.
>
> This document describes the security properties actually enforced by the current runtime. It intentionally separates the production transaction boundary from compatibility/development helpers.

## 1. Security objective

The runtime treats planner output, proposed `PlanIR`, judge output, tool output, and capability payloads as untrusted until they cross explicit deterministic trust boundaries.

The production execution path is:

```text
PlanIR
  ↓ deterministic epistemic validation
AuthorizationCertificate
  ↓ HMAC-bound plan/world/registry/policy/isolation identities
TransactionalExecutionManager
  ↓ fail-closed security preflight
Bubblewrap isolation + resource limits
  ↓ real capability execution
Independent observation verifier
  ↓ empirical live_world_state
Commit gate OR verified reverse-order compensation
```

The core rule is that model confidence, judge approval, process exit status, or planner-declared effects cannot independently establish empirical truth or authorize broader execution privileges.

## 2. Trusted computing base and explicit limitations

The current trusted computing base includes:

- deterministic PlanIR/schema validation;
- the capability registry and verifier contracts;
- HMAC authorization logic;
- `TransactionalExecutionManager`;
- the Bubblewrap policy construction in `ExecutionSandbox`;
- the host Linux kernel and Bubblewrap binary;
- the in-process evidence ledger and secret scrubber.

The system **does not** claim protection against a compromised host kernel, malicious code already executing with equivalent privileges inside the controller process, or a compromised Bubblewrap binary. Bubblewrap is used as a low-level Linux namespace/mount sandbox; the policy constructed by this runtime is the security boundary being tested.

The evidence ledger is an **in-memory integrity-linked SHA-256 chain**. It detects record mutation within the modelled process history but is not a remotely anchored append-only audit log and does not survive hostile process-memory compromise.

## 3. Production isolation policy

### 3.1 Fail-closed default

Production `TransactionalExecutionManager` construction with no caller-supplied sandbox:

1. creates a private `EphemeralWorkspace` with mode `0700`;
2. selects `SecurityProfile.STRICT`;
3. binds the workspace into that policy;
4. constructs `ExecutionSandbox`;
5. refuses execution before dispatch if Bubblewrap is unavailable or cannot satisfy the required boundary;
6. destroys its owned workspace when finalization exits.

The strict profile requires:

```text
use_bwrap = true
require_bwrap = true
allow_unisolated_fallback = false
read_only_root = true
allow_network = false
```

A raw-host subprocess is therefore not a production fallback.

### 3.2 Explicit development compatibility path

The low-level `ExecutionSandbox` still supports an explicitly selected permissive development profile for legacy/unit-test scenarios. Portable Python `sitecustomize` hooks provide defense in depth for that mode, but **no kernel-isolation claim is attached to it**.

`TransactionalExecutionManager` rejects a caller-supplied sandbox unless it is fail-closed and workspace-bound. Its `allow_insecure_test_sandbox` escape hatch is explicitly test-only and is used by Phase 3 semantic tests to inject deterministic backends without weakening the production default.

Production transaction execution also rejects arbitrary custom execution backends, because a caller-provided Python callable is not an isolation boundary.

## 4. Kernel boundary

For an isolated production command, Bubblewrap is configured with an explicit user namespace plus mount/process namespaces and a new session:

- `--unshare-user`;
- isolated UID/GID mapping inside the new user namespace;
- read-only host root (`--ro-bind / /`);
- a read-write bind only for the authorized workspace;
- masked sensitive paths where present;
- new PID, IPC, UTS namespaces;
- cgroup namespace isolation where supported (`--unshare-cgroup-try`);
- `--new-session` and `--die-with-parent`;
- network namespace isolation (`--unshare-net`) when networking is not authorized.

`/dev` and `/proc` are reconstructed through Bubblewrap rather than inherited as arbitrary writable host mounts.

## 5. Filesystem boundary

With the strict policy:

- the host root is mounted read-only;
- the ephemeral workspace is the designated writable region;
- sensitive locations such as `/etc/shadow`, `/etc/sudoers`, `/root`, `~/.ssh`, `~/.aws`, and `~/.gnupg` are masked when present;
- working directories are validated against the workspace;
- `validate_path_within_workspace` rejects traversal and symlink targets outside the workspace;
- structured argv execution avoids implicit shell interpolation.

Static path checks and Python hooks are secondary defense-in-depth mechanisms. The production filesystem claim relies on the real mount namespace/read-only-root boundary.

## 6. Network boundary and authorization binding

`SecurityProfile.STRICT` uses `allow_network=false`, which adds a new network namespace. The runtime never falls back to host networking if that namespace cannot be created.

Network-enabled execution requires a different `IsolationPolicy` (`NETWORK_ALLOWED` or another explicit fail-closed policy). Security-relevant isolation settings are canonicalized and SHA-256 fingerprinted by `compute_isolation_policy_hash()`.

The fingerprint covers security privileges and limits including network permission, root mutability, blocked paths, environment whitelist, resource limits, and fail-closed/bwrap settings. The ephemeral `workspace_dir` is deliberately excluded because it is a per-run instance identifier, not a privilege.

`AuthorizationCertificate` HMAC-signs this `isolation_policy_hash` together with:

- canonical plan hash;
- trusted world-state hash;
- capability-registry hash;
- external policy hash;
- certificate expiry.

`TransactionalExecutionManager` recomputes the runtime isolation fingerprint before dispatch and before compensation. A caller therefore cannot authorize a strict network-denied plan and later substitute a network-enabled sandbox without invalidating the authorization identity.

## 7. Resource and environment controls

The sandbox applies POSIX resource limits, with `prlimit` inside the Bubblewrap boundary when available and `setrlimit` defense in depth:

- address-space limit (`RLIMIT_AS`);
- CPU-time limit (`RLIMIT_CPU`);
- process-count limit (`RLIMIT_NPROC`);
- maximum file-size limit (`RLIMIT_FSIZE`);
- wall-clock execution timeout;
- stdout/stderr output-size truncation.

The child environment starts from a small whitelist rather than inheriting the controller environment. API/database credentials injected into the parent process are therefore not automatically inherited. Output is additionally passed through the secret scrubber before being returned/persisted by the runtime.

## 8. Authorization and state-drift boundaries

Before normal execution, the runtime verifies:

- session is in `EXECUTING` state;
- certificate signature and expiry;
- active certificate identity;
- PlanIR canonical hash;
- capability-registry hash;
- current external policy hash;
- trusted live-world-state hash;
- production isolation-policy hash at the transaction boundary.

`PlanIR.initial_state` is not a substitute for a trusted runtime observation snapshot.

## 9. Execution, evidence, commit, and compensation

Process success alone does not create `VERIFIED_TRUE`.

For effectful actions, an independent verifier must match the exact predicate and typed target arguments. Successful observations update `live_world_state` with `SourceType.OBSERVED_WORLD_STATE` provenance.

`ExecutionPlanManager` does not commit. `TransactionalExecutionManager` commits only after the execution result and mandatory success criteria are empirically attested.

If execution or commit finalization fails after an effectful dispatch, registered compensation capabilities run in reverse dispatch order. Compensation itself must execute and have its postconditions independently witnessed. Missing, failed, unobservable, or exception-raising compensation moves the session to `CONTAINMENT_FAILED`.

## 10. Judge/search trust boundary

LLM judges are advisory only.

- provider adapters use real HTTP transport paths;
- missing credentials, HTTP failures, invalid JSON, missing required fields, invalid verdict enums, or out-of-range structured values produce `UNKNOWN`, not synthetic `PASS`;
- judge output cannot mutate empirical world state;
- judge-suggested changes are translated through registry-grounded IR mutation operators;
- effect-creating mutations fail closed without a capability registry;
- every search candidate is deterministically revalidated;
- `SearchResult.is_certified` is true only for deterministic `ValidationStatus.PASS`;
- every action in a certifiable search result must be registered;
- judge token/cost/latency metadata is tracked separately from deterministic certification.

## 11. Security test evidence

The portable Python matrix runs on Python 3.10–3.14 and includes adversarial tests for:

- missing isolation backend fail-closed behavior;
- insecure supplied sandbox rejection;
- production custom-backend rejection;
- automatic ephemeral-workspace ownership/cleanup;
- isolation-policy HMAC binding and privilege-drift rejection;
- traversal and symlink checks;
- environment-secret stripping;
- resource and output limits;
- malformed judge/provider responses;
- registry-closed search mutations and judge-driven deterministic revalidation.

Because standard GitHub-hosted runners prohibit the user/network namespace operations Bubblewrap needs, CI also contains a dedicated `phase4-kernel-isolation` job. That job launches a privileged Linux Docker integration harness, installs Bubblewrap, and executes the actual kernel-integration suite. The integration tests prove:

1. Bubblewrap successfully starts and executes inside the authorized workspace;
2. read-only-root/workspace mounts prevent a process from writing outside the workspace;
3. default network denial cannot silently fall back to host networking.

This dedicated job is evidence for the real namespace/mount boundary rather than a mock or a startup-failure-only test.

## 12. Remaining phase

Phase 6 remains intentionally separate: empirical evaluation on public planning benchmarks, baselines, ablations, false-PASS analysis, recovery/safety metrics, cost/latency analysis, and statistical reporting.

Until Phase 6 is completed, the repository may claim the architecture/security gates through Phase 5, but **not** a completed empirical benchmark or full research-evaluation gate.
