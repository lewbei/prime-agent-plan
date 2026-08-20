# Probabilistic Self-Verification for Prime

Self-verification is inherited by Prime on top of the PR #1 Phase 0–5 runtime already merged into `main`.

The architectural rule is **same implementation model + same implementation thinking -> same verifier model + same verifier thinking**. Prime does not hard-code Gemini, DeepSeek, a deeper verifier, or a cheaper verifier:

```text
implementation model M, thinking T
    -> generate N candidates
same model M, same thinking T
    -> probabilistically verify/rank those candidates
Prime deterministic validator
    -> certify/reject
Prime runtime
    -> execute and independently witness effects
```

Examples:

- Gemini 3.7 Flash at `high` -> Gemini 3.7 Flash at `high`.
- Gemini 3.7 Flash at `medium` -> Gemini 3.7 Flash at `medium`.
- DeepSeek V4 Flash at `high` -> DeepSeek V4 Flash at `high`.
- If the implementation uses the provider/model default, the verifier also uses that default rather than introducing a new override.

The active implementation-model identity is resolved from runtime/session context: an explicit runtime value, session metadata (`implementation_model`, `generator_model`, `model`, or `model_id`), or `PRIME_IMPLEMENTATION_MODEL` / `PRIME_MODEL` / `PLAN_MODEL`.

The thinking profile is resolved independently from `implementation_thinking`, `thinking_profile`, `thinking_level`, `reasoning_effort`, `thinking_budget`, or the corresponding `PRIME_IMPLEMENTATION_THINKING` / `PRIME_THINKING` / `PLAN_THINKING` environment values. The canonical profile is then reused unchanged for the verifier.

Normal callers keep using:

```python
plan.assess_candidates(session, candidate_plans)
```

A harness should record both parts of its runtime identity:

```python
session = plan.start(
    objective,
    meta={
        "implementation_model": current_model_id,
        "implementation_thinking": current_thinking_profile,
    },
)
```

No separate self-verification mode is required. Verification scaling remains `n_evaluations=2` and `pivots=1`, following the upstream Best-of-5 reproduction pattern.

## Why Prime overrides upstream verifier defaults

The upstream `llm-verifier` package currently applies provider-specific verifier reasoning defaults. Its DeepSeek path enables its own reasoning-effort policy, while its Gemini path was written with a fixed Gemini thinking configuration. Prime wraps the verifier client and replaces those verifier-only choices with the implementation's inherited thinking profile.

For OpenAI-compatible clients this removes upstream reasoning overrides before applying the inherited `reasoning_effort` / provider-native thinking fields. For native Gemini clients it replaces the generated `thinking_config` with the inherited `thinking_level` or exact legacy `thinking_budget`.

If a backend cannot faithfully express the inherited thinking profile, the verifier call fails closed and `plan.assess_candidates(...)` uses PR #1's deterministic ranking instead. Prime does not silently approximate `high` as `medium`, convert an exact budget to a different budget, or disable thinking.

Probabilistic ranking remains advisory. Deterministic failures are not rehabilitated, PASS candidates take precedence over UNKNOWN, selected PlanIR is deterministically revalidated, and only deterministic PASS can be certified. PR #1 authorization, isolation, execution witnessing, commit gates, and recovery remain unchanged.

The external logprob backend is lazy (`pip install -e '.[verification]'`). Upstream benchmark gains are not claimed as Prime results.
