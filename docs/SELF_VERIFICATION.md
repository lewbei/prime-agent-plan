# Probabilistic Self-Verification for Prime

Self-verification is inherited by Prime on top of the PR #1 runtime already merged into `main`.

PR #2 therefore includes the entire Phase 0–5 architecture from PR #1 plus the new verification-scaling changes. GitHub shows only the new diff relative to `main`.

Default verification is same-model Gemini 3.7 Flash -> Gemini 3.7 Flash. Normal `PlanSelfVerifier.select(candidate_plans)` uses `gemini-3.7-flash` for both generator and verifier with `n_evaluations=2` and `pivots=1`; no separate same-model mode is required.

Probabilistic ranking remains advisory. FAIL is excluded, PASS candidates take precedence over UNKNOWN, the selected plan is deterministically revalidated, and only deterministic PASS can be certified. PR #1 authorization, isolation, execution witnessing, and recovery remain unchanged.

The external logprob backend is lazy (`pip install -e '.[verification]'`) so deterministic Prime remains operational without provider credentials. No benchmark uplift is claimed until a real Gemini provider run is executed.
