# Prime Agent Plan Mode (`/plan`)

**Make the plan better before the agent starts doing the work.**

Plan Mode is a planning skill for Prime Agent that turns a one-shot LLM plan into an **iterative, checked, auditable planning process**. Instead of accepting the first plan that sounds reasonable, `/plan` repeatedly drafts, critiques, verifies, simulates, compares, and revises candidate plans until the plan is strong enough to pass a release gate.

## Epistemic verification/runtime prototype

The current prototype branch includes:

- Canonical Plan IR with provenance and four-state empirical truth.
- Separate projected causal truth for plan-time reasoning.
- Trusted observed-world snapshot normalization and typed fact identity.
- Capability/effect registry binding with exact typed verifier targets.
- Structured argv execution with independent post-execution attestation.
- Explicit plan/registry/policy/world identity checks before execution.
- Transactional finalization: execution produces evidence only; commit requires an attested successful execution and empirically verified mandatory success criteria.
- Saga compensation for dispatched effectful actions, executed in reverse order through registered compensation capabilities and independently verified postconditions.
- Fail-closed `CONTAINMENT_FAILED` when compensation is missing, invalid, unobservable, blocked by drift/preconditions, fails execution/verification, or raises during recovery.

The structured subprocess runner is **not** a hostile-code sandbox, and the evidence ledger is an **in-memory integrity-linked event chain**, not durable tamper-proof storage. Stronger isolation/security, live model judges/search integration, and public benchmark evaluation remain future phases.
