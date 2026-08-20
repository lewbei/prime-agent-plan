# Probabilistic Self-Verification for Prime

## Status

Self-verification is an **inherited Prime candidate-ranking behavior**, not a separate mode that callers must select. It is not an empirical truth source and not a benchmark claim.

This work is layered on the complete PR #1 runtime already merged into `main`. PR #2 therefore inherits the full Phase 0–5 architecture. GitHub shows only the new verification-scaling files in PR #2 because its diff is relative to the current `main`, which already contains PR #1.

It is inspired by **LLM-as-a-Verifier: A General-Purpose Verification Framework** (arXiv:2607.05391) and the public `llm-as-a-verifier/llm-as-a-verifier` implementation. The upstream project demonstrates **same-model self-verification** on Terminal-Bench 2.1: `deepseek-v4-flash` generates multiple mini-swe-agent trajectories and the same `deepseek-v4-flash` model ranks/verifies those trajectories. Their reported Best-of-5 result improves from 78.7% Pass@1 to 88.0% ± 0.6% after self-verifier selection. Those numbers belong to the upstream project and are **not Prime results**.

Prime mainly uses **Gemini 3.7 Flash** for implementation, so Prime's inherited analogue is:

```text
Gemini 3.7 Flash -> generate candidate implementations/plans
Gemini 3.7 Flash -> probabilistically verify/rank the same model's candidates
Prime deterministic validator -> certify or reject the selected candidate
Prime runtime witness -> verify empirical execution effects
```

## Inherited behavior

Normal selection already uses the same-model configuration; no separate self-verification mode is required:

```python
selector = PlanSelfVerifier(registry=registry, verifier=soft_verifier)
result = selector.select(candidate_plans)
```

Defaults:

```text
generator model:  gemini-3.7-flash
verifier model:   gemini-3.7-flash
n_evaluations K:  2
pivots:            1
```

`K=2` and `pivots=1` match the upstream repository's Best-of-5 Terminal-Bench 2.1 reproduction settings. `select_same_model()` remains only as a deprecated compatibility alias.

## Prime verification boundary

1. `CapabilityRegistry` defines the closed action world.
2. Deterministic `FAIL` candidates never reach the probabilistic verifier.
3. If any deterministic `PASS` candidates exist, `UNKNOWN` candidates do not compete with them.
4. The probabilistic verifier ranks eligible candidates only.
5. The selected candidate is deterministically revalidated.
6. `is_certified=True` only for deterministic `PASS`.
7. Authorization and runtime empirical witnessing remain unchanged.
8. LLM output can never create empirical `VERIFIED_*` facts.

## Gemini backend

The upstream scoring method requires token-level log probabilities. Its Gemini path uses Vertex AI. Prefer an explicit Vertex-backed `google-genai` client:

```python
from google import genai
from plan_mode.self_verification import ProbabilisticSelfVerifier, PlanSelfVerifier

vertex_client = genai.Client(vertexai=True, api_key=VERTEX_API_KEY)
soft_verifier = ProbabilisticSelfVerifier(client=vertex_client)
selector = PlanSelfVerifier(registry=registry, verifier=soft_verifier)
result = selector.select(candidate_plans)
```

The self-verification policy is part of Prime; the external probabilistic/logprob backend remains lazy so deterministic Prime continues operating when provider credentials are unavailable.

```bash
pip install -e '.[verification]'
```

## What we are not claiming

Prime does **not** claim to reproduce the paper's benchmark results or to have already measured the same uplift with Gemini 3.7 Flash. The integration adopts the same-model verification mechanism while preserving Prime's deterministic and empirical truth boundaries.
