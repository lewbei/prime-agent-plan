# Probabilistic Self-Verification for Prime

## Status

This is an **optional test-time selection layer**, not an empirical truth source and not a benchmark claim.

It is inspired by **LLM-as-a-Verifier: A General-Purpose Verification Framework** (arXiv:2607.05391) and the public `llm-as-a-verifier/llm-as-a-verifier` implementation. The upstream project reports that the same `deepseek-v4-flash` model can generate multiple Terminal-Bench 2.1 trajectories and verify its own rollouts; their README reports Best-of-5 Pass@1 of 78.7% and self-verifier selection of 88.0% ± 0.6%. Those numbers belong to the upstream project and are **not Prime results**.

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
same or separate generator model
        |
        +--> candidate PlanIR 1
        +--> candidate PlanIR 2
        +--> ...
        +--> candidate PlanIR N
                    |
                    v
        deterministic closed-world gate
          FAIL    -> discard
          PASS    -> eligible for certified selection
          UNKNOWN -> eligible only when no PASS exists, for rework ranking
                    |
                    v
        probabilistic Best-of-N verifier
        - fine-grained criteria
        - repeated evaluations
        - pivot tournament
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

## Same-model self-verification

Self-verification is explicitly supported by passing the same model identifier as both generator and verifier metadata:

```python
from plan_mode.self_verification import PlanSelfVerifier

result = selector.select(
    candidate_plans,
    generator_model="deepseek-v4-flash",
    verifier_model="deepseek-v4-flash",
    n_evaluations=2,
    pivots=1,
)
```

`result.is_self_verification` records that the generator and verifier model identities match. This flag is descriptive only; it does not change certification semantics.

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

This integration does **not** claim that Prime reproduces the paper's Terminal-Bench, SWE-Bench, robotics, or medical results. It only provides a compatible optional mechanism for probabilistic Best-of-N selection while retaining Prime's deterministic and empirical verification boundaries.
