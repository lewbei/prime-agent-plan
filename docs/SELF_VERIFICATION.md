# Probabilistic Self-Verification for Prime

## Status

Self-verification is an **inherited Prime behavior**, not a separate mode that callers must select.

PR #2 is based on `main` after PR #1 was merged, so it already contains the complete Phase 0–5 runtime from PR #1. GitHub only displays the new verification-scaling diff relative to that inherited base.

The design is inspired by **LLM-as-a-Verifier** (arXiv:2607.05391). Upstream demonstrates same-model self-verification using `deepseek-v4-flash` for both trajectory generation and verification. Prime uses the direct analogue for its main implementation model:

```text
Gemini 3.7 Flash -> generate candidate implementations/plans
Gemini 3.7 Flash -> rank/verify those candidates
Prime deterministic validator -> certify/reject
Prime runtime witness -> verify empirical execution effects
```

Normal `PlanSelfVerifier.select(candidate_plans)` already defaults to:

```text
generator:      gemini-3.7-flash
verifier:       gemini-3.7-flash
n_evaluations:  2
pivots:         1
```

No separate same-model mode is required. `select_same_model()` remains only as a deprecated compatibility alias.

## Hard boundary

- deterministic `FAIL` candidates never reach the LLM verifier;
- `PASS` candidates take precedence over `UNKNOWN` candidates;
- `UNKNOWN` may only be ranked for rework;
- the selected candidate is deterministically revalidated;
- `is_certified=True` only for deterministic `PASS`;
- LLM scores never create empirical `VERIFIED_*` facts;
- authorization, isolation, execution witnessing, and recovery from PR #1 remain unchanged.

## Gemini backend

The upstream probabilistic scoring path requires token-level log probabilities. For Gemini, use a Vertex-backed `google-genai` client when live verification is enabled:

```python
from google import genai
from plan_mode.self_verification import ProbabilisticSelfVerifier, PlanSelfVerifier

vertex_client = genai.Client(vertexai=True, api_key=VERTEX_API_KEY)
selector = PlanSelfVerifier(
    registry=registry,
    verifier=ProbabilisticSelfVerifier(client=vertex_client),
)
result = selector.select(candidate_plans)
```

The self-verification policy is part of Prime. The external probabilistic/logprob backend is lazy so the deterministic runtime remains operational without provider credentials.

```bash
pip install -e '.[verification]'
```

## Claims

Prime does not claim to reproduce the upstream benchmark uplift yet. The integration adopts the same-model verification mechanism while preserving Prime's deterministic and empirical truth boundaries.
