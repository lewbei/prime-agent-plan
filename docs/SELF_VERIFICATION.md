# Probabilistic Self-Verification for Prime

Self-verification is an **inherited Prime behavior**, not a separate mode that callers must select.

PR #2 is based on `main` after PR #1 was merged, so it already includes the complete Phase 0–5 runtime from PR #1. GitHub only displays the new verification-scaling diff relative to that inherited base.

Following the same-model setup in **LLM-as-a-Verifier** (arXiv:2607.05391), Prime defaults to:

```text
Gemini 3.7 Flash -> generate candidate implementations/plans
Gemini 3.7 Flash -> probabilistically rank those candidates
Prime deterministic validator -> certify/reject
Prime runtime witness -> verify empirical effects
```

Normal `PlanSelfVerifier.select(candidate_plans)` defaults to `gemini-3.7-flash` for both generator and verifier, with `n_evaluations=2` and `pivots=1`. `select_same_model()` remains only as a deprecated compatibility alias.

The LLM verifier is advisory: deterministic FAIL candidates are excluded, PASS candidates take precedence over UNKNOWN candidates, the selected plan is deterministically revalidated, and only deterministic PASS can set `is_certified=True`. Authorization, isolation, execution witnessing, and recovery inherited from PR #1 remain unchanged.

The external logprob backend is lazy (`pip install -e '.[verification]'`) so deterministic Prime remains operational without provider credentials. Prime does not claim the upstream benchmark uplift until a real Gemini provider run is executed.
