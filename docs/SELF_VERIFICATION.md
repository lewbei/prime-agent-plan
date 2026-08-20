# Probabilistic Self-Verification for Prime

Self-verification is inherited by Prime on top of the PR #1 Phase 0–5 runtime already merged into `main`.

The architectural rule is **same implementation model -> same verifier model**. Prime does not hard-code Gemini, DeepSeek, or any other model:

```text
implementation model M
    -> generate N candidates
implementation model M
    -> probabilistically verify/rank those candidates
Prime deterministic validator
    -> certify/reject
Prime runtime
    -> execute and independently witness effects
```

Examples: Gemini 3.7 Flash -> Gemini 3.7 Flash, DeepSeek V4 Flash -> DeepSeek V4 Flash, or any other supported active model -> itself.

The active implementation-model identity is resolved from runtime/session context: an explicit runtime value, session metadata (`implementation_model`, `generator_model`, `model`, or `model_id`), or `PRIME_IMPLEMENTATION_MODEL` / `PRIME_MODEL` / `PLAN_MODEL`. If no identity is available, Prime deliberately falls back to the original deterministic PR #1 selector rather than silently choosing a verifier model.

Normal callers keep using:

```python
plan.assess_candidates(session, candidate_plans)
```

For example, a harness can record its current model when starting a session:

```python
session = plan.start(
    objective,
    meta={"implementation_model": current_model_id},
)
```

No separate `select_same_model()` mode is required. The default verification-scaling settings remain `n_evaluations=2` and `pivots=1`, following the upstream Best-of-5 self-verification reproduction pattern.

Probabilistic ranking remains advisory. Deterministic failures are not rehabilitated, PASS candidates take precedence over UNKNOWN, selected PlanIR is deterministically revalidated, and only deterministic PASS can be certified. PR #1 authorization, isolation, execution witnessing, commit gates, and recovery remain unchanged.

The external logprob backend is lazy (`pip install -e '.[verification]'`). If its package, credentials, model support, or provider is unavailable, deterministic Prime remains operational and records the fallback. Upstream benchmark gains are not claimed as Prime results.
