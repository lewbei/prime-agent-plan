# Probabilistic Self-Verification for Prime

Self-verification is an inherited Prime behavior layered on the PR #1 runtime already merged into `main`.

PR #2 therefore contains the complete Phase 0–5 architecture from PR #1 plus the verification-scaling changes. The PR diff only shows the new files relative to `main`.

Default candidate verification follows the same-model pattern from LLM-as-a-Verifier:

```text
Gemini 3.7 Flash -> generate candidates
Gemini 3.7 Flash -> rank candidates
Prime deterministic validator -> certify/reject
Prime runtime witness -> verify empirical effects
```

`PlanSelfVerifier.select(candidate_plans)` defaults to `gemini-3.7-flash` for both generator and verifier, with `n_evaluations=2` and `pivots=1`. A separate same-model selection mode is not required; `select_same_model()` is only a compatibility alias.

Probabilistic ranking cannot override deterministic truth: FAIL is excluded, PASS outranks UNKNOWN eligibility, the selected plan is revalidated, and only deterministic PASS can be certified. PR #1 authorization, isolation, witnessing, and recovery remain unchanged.

The external logprob backend is lazy (`pip install -e '.[verification]'`) so deterministic Prime can run without provider credentials. Benchmark uplift is not claimed until a real Gemini provider run is executed.
