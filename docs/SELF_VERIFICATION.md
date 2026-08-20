# Probabilistic Self-Verification for Prime

Self-verification is an **inherited Prime behavior**, not a separate mode that callers must select.

PR #2 is based on `main` after PR #1 was merged. Therefore PR #2 already contains the complete Phase 0–5 runtime from PR #1; its visible diff only contains the new verification-scaling changes relative to that inherited base.

The design follows the same-model idea from **LLM-as-a-Verifier** (arXiv:2607.05391): upstream uses DeepSeek V4 Flash to generate trajectories and DeepSeek V4 Flash to verify them. Prime uses the direct analogue for its implementation model:

```text
Gemini 3.7 Flash -> generate candidate implementations/plans
Gemini 3.7 Flash -> probabilistically rank those candidates
Prime deterministic validator -> certify/reject
Prime runtime witness -> verify empirical effects
```

Normal selection already defaults to:

```text
generator:      gemini-3.7-flash
verifier:       gemini-3.7-flash
n_evaluations:  2
pivots:         1
```

Callers use ordinary `PlanSelfVerifier.select(candidate_plans)`. `select_same_model()` is retained only as a deprecated compatibility alias.

The LLM verifier remains advisory: deterministic FAIL candidates are excluded, PASS candidates take precedence over UNKNOWN candidates, the selected plan is deterministically revalidated, and only deterministic PASS can set `is_certified=True`. Authorization, isolation, execution witnessing, and recovery inherited from PR #1 remain unchanged.

For live Gemini verification, the upstream logprob scoring path requires a suitable Vertex-backed `google-genai` client. The external backend remains lazy so deterministic Prime stays operational without provider credentials:

```bash
pip install -e '.[verification]'
```

Prime does not claim to reproduce the upstream benchmark uplift until a real Gemini provider run is executed.
