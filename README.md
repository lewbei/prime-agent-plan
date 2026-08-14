# Plan Mode (`/plan`)

An autonomous, multi-round iterative planning engine with formal rubric scoring, STRIPS-style forward simulation, Monte Carlo Tree Search (MCTS) candidate exploration, multi-judge falsifiable consensus, and the Cordis Spatiotemporal Composability runtime ($\Gamma_\infty$ effect monoid, Fiber lifecycle calculus, and transactional plan execution).

---

## 🌟 Key Capabilities

### 1. Multi-Round Iterative Plan Optimization
- **Generator/Critic Separation**: Subagents synthesize and revise plans against an objective; the engine computes deterministic scoring, constraint validation, and search heuristics.
- **Auditable Convergence**: Every iteration, score vector, and diagnostic critique is tracked. Convergence occurs only when the score stabilizes or the release gate is satisfied.

### 2. Comprehensive 97-Check Rubric Engine
- Evaluates plans across structural formatting, dependency ordering, milestone tracking, artifact contracts, error mitigation, and resource budgets.
- Extensible via `RUBRIC.md` overrides.

### 3. STRIPS-Style Simulation & Verification
- **World State Tracking**: Emulates precondition checking, effect application, artifact consumption/production, and forward-reference blocking.
- **Feasibility & Landmark Checking**: Ensures no steps depend on ungrounded artifacts or cyclic handoffs.

### 4. MCTS & Best-of-N Search Engine
- **Search Operations**: `search_expand`, `search_select` (UCB1), `search_backtrack`, and automated candidate tournaments.
- **Recombination & Mutation**: Critique-aware targeted mutations and cross-over plan recombination.

### 5. Multi-Judge Falsifiable Consensus
- **Tri-Judge Ensemble**: Evaluates plan robustness with falsifiable criteria, filtering out vague assertions and computing consensus median scores with release gating (`plan.release()`).

### 6. Cordis Spatiotemporal Composability Engine (Shi, Zhang, Cui 2026)
- **$\Gamma_\infty$ Context Engine**: Dynamic context management with effect propagation and reactive coeffect tracking.
- **Fiber Lifecycle Calculus**: 10-rule state machine managing fiber lifecycles (`PENDING` $\to$ `ACTIVE` $\to$ `COMMITTED` / `ROLLED_BACK` / `TERMINATED`).
- **Twisted Monoid ($M \rtimes \text{Aut}(M)$)**: Exact LIFO inverse unwinding for side effects, file mutations, and tool registrations.
- **Transactional Rollouts**: `execute_plan` with atomic commits / clean rollbacks upon failure, and `speculative_rollout` for isolated subagent scratchpads.
- **Scoped Subagent Realms**: Hermetic environment isolation preventing cross-agent contamination.

---

## 📂 Repository Structure

```
├── SKILL.md                 # Prime Agent Skill definition & slash command protocol
├── pyproject.toml           # Package configuration & build specification
├── README.md                # Project documentation
├── src/
│   ├── plan/                # Top-level Prime Agent skill entrypoint
│   │   └── __init__.py      # Unified re-exports & runtime engine resolver
│   └── plan_mode/           # Core planning & composability engine
│       ├── __init__.py      # Session management, rubric evaluation, simulation & release gates
│       ├── cordis.py        # Cordis spatiotemporal composability engine (Context, Fiber, TwistedMonoid)
│       ├── search_engine.py # MCTS search, UCB1 selection, mutation & recombination
│       ├── judge_client.py  # Multi-judge client & falsifiable consensus
│       └── RUBRIC.md        # 97-check scoring rubric specification
└── tests/
    ├── conftest.py          # Pytest fixtures & environment setup
    ├── test_cordis.py       # 9 comprehensive suites for Cordis runtime
    ├── test_engine.py       # 28 suites for simulation, search, session fold, & templates
    └── test_rubric.py       # 8 suites for rubric parsing, regexes, & constraints
```

---

## 🚀 Usage

### Python API

```python
import plan

# Start or resume a planning session
session = plan.start("Design and deploy a distributed stream processing pipeline")

# Assess draft plan
assessment = plan.assess(session, draft_content)
print(f"Round: {assessment['round']}, Score: {assessment['score']}")

# Search plan candidates with MCTS
candidates = [variant_1, variant_2, variant_3]
best_cand = plan.assess_candidates(session, candidates)

# Cordis transactional plan execution
plan_steps = [
    {"name": "Step 1", "effect": {"path": "out/a.txt", "content": "data"}},
    {"name": "Step 2", "effect": {"path": "out/b.txt", "content": "processed"}}
]
result = await plan.execute_plan(plan_steps, rollback_on_failure=True)
```

### Prime Agent Skill CLI

```bash
# Start plan mode via Prime Agent CLI
plan --objective "Implement end-to-end multi-modal evaluation harness"
```

---

## 🧪 Testing

Run the full test suite with `pytest`:

```bash
pytest -v
```

All 45 test suites pass with 100% coverage across Cordis mechanics, STRIPS simulation, rubric compilation, and MCTS search algorithms.

---

## 📜 License

Private & Confidential. All rights reserved.
