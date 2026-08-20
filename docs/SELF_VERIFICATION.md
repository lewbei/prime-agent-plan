# Probabilistic Self-Verification for Prime

## Status

This is an **optional test-time selection layer**, not an empirical truth source and not a benchmark claim.

It is inspired by **LLM-as-a-Verifier: A General-Purpose Verification Framework** (arXiv:2607.05391) and the public `llm-as-a-verifier/llm-as-a-verifier` implementation. The upstream project demonstrates **same-model self-verification** on Terminal-Bench 2.1: `deepseek-v4-flash` generates multiple mini-swe-agent trajectories and the same `deepseek-v4-flash` model ranks/verifies those trajectories. Their reported Best-of-5 result improves from 78.7% Pass@1 to 88.0% ± 0.6% after self-verifier selection. Those numbers belong to the upstream project and are **not Prime results**.

Prime mainly uses **Gemini 3.7 Flash** for implementation, so Prime's recommended analogue is:

```text
Gemini 3.7 Flash -> generate candidate implementations/plans
Gemini 3.7 Flash -> probabilistically verify/rank the same model's candidates
Prime deterministic validator -> certify or reject the selected candidate
Prime runtime witness -> verify empirical execution effects
```

The current Google model identifier is `gemini-3.7-flash`.

## Why it fits Prime

LLM-as-a-Verifier treats verification as a test-time scaling axis. Instead of asking an LM judge for one coarse discrete verdict, it uses fine-grained probabilistic scoring, repeated evaluations, criteria decomposition, and a budget-efficient pivot tournament to rank candidate trajectories.

Prime already has a different hard boundary:

1. `CapabilityRegistry` defines the closed action world.
2. The deterministic epistemic validator returns `PASS`, `FAIL`, or `UNKNOWN`.
3. Authorization binds the selected PlanIR and policy state.
4. Runtime execution is independently witnessed.
5. LLM output can never create empirical `VERIFIED_*` facts.

The probabilistic verifier therefore sits **before certification** as a soft ranking mechanism.

## Prime selection pipeline

```text
Gemini 3.7 Flash
        |
        +--> candidate PlanIR / implementation 1
        +--> candidate PlanIR / implementation 2
        +--> ...
        +--> candidate PlanIR / implementation N
                    |
                    v
        deterministic closed-world gate
          FAIL    -> discard
          PASS    -> eligible for certified selection
          UNKNOWN -> eligible only when no PASS exists, for rework ranking
                    |
                    v
        Gemini 3.7 Flash self-verifier
        - fine-grained criteria
        - repeated evaluations
        - probabilistic pivot tournament
                    |
                    v
        selected candidate
                    |
                    v
        deterministic revalidation
          PASS    -> may proceed to authorization
          UNKNOWN -> rework only
          FAIL    -> reject
                    |
                    v
        execution + empirical witnessing
```

## Recommended same-model configuration

The direct convenience API is:

```python
from plan_mode.self_verification import PlanSelfVerifier

result = selector.select_same_model(candidate_plans)
```

This defaults to:

```text
generator model:  gemini-3.7-flash
verifier model:   gemini-3.7-flash
n_evaluations K:  2
pivots:            1
```

`K=2` and `pivots=1` match the upstream repository's Best-of-5 Terminal-Bench 2.1 self-verification reproduction settings. The model is changed from the paper/repository's `deepseek-v4-flash` to `gemini-3.7-flash` because Gemini 3.7 Flash is Prime's primary implementation model.

For difficult work, generate **5 independent candidates** before calling `select_same_model`; for routine work, 3 candidates can be used to reduce cost. Candidate generation remains the responsibility of the agent/harness rather than this selector.

An explicit model can still be supplied:

```python
result = selector.select_same_model(
    candidate_plans,
    model="gemini-3.7-flash",
    n_evaluations=2,
    pivots=1,
)
```

`result.is_self_verification` records that generator and verifier model identities match. This flag is descriptive only; it does not change certification semantics.

## Gemini backend requirement

The upstream `llm-verifier` scoring method requires token-level log probabilities. Its Gemini path uses **Vertex AI**, not the plain Gemini API path, because the verifier must observe score-token probabilities.

When Gemini is the verifier, prefer an explicit Vertex-backed `google-genai` client so environment-key precedence cannot silently select another provider:

```python
from google import genai
from plan_mode.self_verification import ProbabilisticSelfVerifier, PlanSelfVerifier

vertex_client = genai.Client(vertexai=True, api_key=VERTEX_API_KEY)
soft_verifier = ProbabilisticSelfVerifier(client=vertex_client)
selector = PlanSelfVerifier(registry=registry, verifier=soft_verifier)

result = selector.select_same_model(candidate_plans)
```

Alternatively, configure the upstream package with `VERTEX_API_KEY` and no higher-priority incompatible verifier backend.

**Compatibility status:** Prime's adapter and safety semantics are CI-proven with an injected verifier seam. A live Gemini 3.7 Flash + Vertex logprob run is still a provider-compatibility check, not something Prime claims to have benchmarked or reproduced yet.

## Default verification criteria

Prime decomposes the soft evaluation into criteria aligned with its architecture:

- **Goal satisfaction** — does the candidate actually cover the requested goal?
- **Causal coherence** — are actions ordered consistently with preconditions/effects?
- **Evidence discipline** — does it avoid treating assumptions or predictions as empirical truth?
- **Executability** — is it capability-grounded and independently verifiable?
- **Recovery readiness** — does it avoid unnecessary irreversible effects and preserve recovery paths?

These criteria are advisory. A high probabilistic score cannot override a deterministic contradiction.

## Safety invariants

- Deterministic `FAIL` candidates are never rehabilitated by an LLM score.
- If any deterministic `PASS` candidates exist, `UNKNOWN` candidates do not compete with them for certified selection.
- `UNKNOWN` candidates may be ranked only to choose a rework target.
- The selected candidate is deterministically revalidated after probabilistic ranking.
- `is_certified=True` only when the final deterministic status is `PASS`.
- Probabilistic verifier scores never modify `WorldFact`, `FactTruth`, authorization certificates, verifier evidence, or empirical runtime state.
- The verifier may rank or suggest; it may not attest.

## Installation

Core Prime does not depend on the external verifier package.

```bash
pip install -e '.[verification]'
```

The extra pins `llm-verifier>=0.2.0,<0.3.0` so upstream scoring API changes cannot silently alter Prime behavior.

## What we are not claiming

This integration does **not** claim that Prime reproduces the paper's Terminal-Bench, SWE-Bench, robotics, or medical results. It also does not claim that Gemini 3.7 Flash necessarily achieves the same verification uplift reported for DeepSeek V4 Flash. It provides the same-model verification mechanism adapted to Prime's actual implementation model while retaining Prime's deterministic and empirical verification boundaries.
