# Plan Mode (`/plan`)

An autonomous, multi-round iterative planning engine with formal rubric scoring, STRIPS-style forward simulation, Monte Carlo Tree Search (MCTS) candidate exploration, multi-judge falsifiable consensus, and the **Cordis Spatiotemporal Composability Runtime** ($\Gamma_\infty$ effect monoid, Fiber lifecycle calculus, and transactional plan execution).

---

## 📚 Theoretical Foundations & Literature Corpus

Plan Mode is built upon empirical synthesis and formal paradigms from over **300 research papers (2024–2026)** across agentic planning, symbolic verification, and composability calculus:

| Domain | Key Papers & Citations | Core Contribution Applied |
|---|---|---|
| **Spatiotemporal Composability** | **Cordis** (*Shi, Zhang, Cui 2026*) | $\Gamma_\infty$ context monoid, 10-rule Fiber lifecycle state machine, Twisted Monoid ($M \rtimes \text{Aut}(M)$) exact LIFO rollback, and hermetic subagent realm isolation. |
| **Mechanical Verification & Benchmarks** | **PlanBench** (*Valmeekam et al., 2409.13373, 2406.13094*)<br>**Natural Plan** (*2406.04520*)<br>**LLM-Modulo** (*2502.12435, 2512.09629*) | Demonstrates that standalone LLM self-review degrades without external mechanical verifiers; implements mechanical anchoring, representation invariance, and non-LLM ground-truth checkers. |
| **State Simulation & Symbolic Grounding** | **SymPlanner** (*2505.01479*)<br>**PyPDDLEngine** (*2603.06064*)<br>**SimPlan** (*2402.11489*) | STRIPS-style forward world-state simulation, artifact production/consumption tracking, forward-reference blocking, and landmark reachability. |
| **Tree Search & Candidate Optimization** | **LATS** (*2310.04406*)<br>**SYMPHONY** (*2601.22623*)<br>**CB-MCTS** (*2603.02154*)<br>**GATS** (*2607.08894*) | Monte Carlo Tree Search over plan space, UCB1 candidate selection, multi-candidate expansion, cost penalties, and transposition tables. |
| **Evolutionary Recombination & Ensembles** | **Mind Evolution** (*2501.09891*)<br>**Plan Ensembles** (*2601.17942*)<br>**Diversity Maintenance** (*2509.22613*) | Evaluator-driven plan recombination (crossover of high-scoring plan prefixes/suffixes), critique-aware mutations, and best-of-N selection tournaments. |
| **Failure Prevention & Memory Distillation** | **PPA-Plan** (*2601.11908*)<br>**RoT** (*2404.05449*)<br>**GNNVerifier** (*2603.14730*) | Pre-emptive failure enumeration (negative constraints), localized root-cause grouping, and automated rule distillation. |
| **Context Management & Budgeting** | **HIPIF** (*2606.10507*)<br>**SafeRun** (*2606.09027*)<br>**BRACE** (*2608.01428*) | Superseded history folding (keeping best and last-two rounds), numeric budget reconciliation, and revision horizon guarding. |
| **Ensemble Judgment & Release Gating** | **Tri-Judge Consensus** (*2510.03469*)<br>**Confidence Gates** (*2602.08948, 2608.10729*) | Independent multi-judge panel with falsifiable criteria validation, median score consensus, and strict release gating. |
| **Self-Evolution & Search Adaptation** | **SERP** (*2603.02772*)<br>**LFS** (*2506.05213*) | Adaptive search width escalation upon score plateaus and strategy self-upgrading. |

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

## 🔧 How to Add into Prime Agent (`prime-agents`)

Prime Agent natively supports Python-backed skills installed either globally or locally per project.

### Method 1: Global Installation (Recommended)

To make `/plan` available across all your Prime Agent sessions and projects on your machine:

```bash
# 1. Create global skills directory if it doesn't exist
mkdir -p ~/.prime/agent/skills

# 2. Clone the repository into ~/.prime/agent/skills/plan
git clone https://github.com/lewbei/prime-agent-plan.git ~/.prime/agent/skills/plan
```

### Method 2: Project-Local Installation

To include `/plan` as part of a specific project repository:

```bash
# 1. From your project root, create .prime/agent/skills
mkdir -p .prime/agent/skills

# 2. Clone or submodule plan into .prime/agent/skills/plan
git clone https://github.com/lewbei/prime-agent-plan.git .prime/agent/skills/plan
# OR add as a submodule:
git submodule add https://github.com/lewbei/prime-agent-plan.git .prime/agent/skills/plan
```

### Method 3: Activating & Verifying

1. **In an interactive Prime Agent session**:
   - Reload installed skills with `/reload`.
   - Invoke `/plan` or `/skill:plan`.
2. **From the IPython kernel**:
   ```python
   import plan
   session = plan.start("Your objective here")
   ```
3. **From the Shell CLI**:
   ```bash
   plan --objective "Your objective here"
   ```

---

## 📂 Repository Structure

```
├── SKILL.md                 # Prime Agent Skill definition & slash command protocol
├── pyproject.toml           # Package configuration & build specification
├── README.md                # Project documentation & literature review
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

## 🚀 API Examples

### 1. Iterative Planning Loop

```python
import plan

# Start or resume a planning session
session = plan.start("Design and deploy a distributed stream processing pipeline")

# Assess draft plan against 97-check rubric + STRIPS simulator
assessment = plan.assess(session, draft_content)
print(f"Round: {assessment['round']}, Score: {assessment['score']}")
print(f"Critiques: {assessment['critiques']}")

# Release gate check
release_status = plan.release(session, min_score=90)
```

### 2. MCTS & Best-of-N Candidate Search

```python
import plan

# Expand candidate variants in plan space
candidates = [variant_1, variant_2, variant_3]
best_cand = plan.assess_candidates(session, candidates)

# Run autonomous MCTS search loop
search_result = await plan.search(session, iterations=4, width=3, mode="mcts")
print(f"Best plan score: {search_result['best_score']}")
```

### 3. Cordis Spatiotemporal Composability & Transactional Rollouts

```python
import plan

# Execute multi-step plan with automatic LIFO rollback upon failure
steps = [
    {"name": "Create config", "effect": {"path": "config.yaml", "content": "env: prod"}},
    {"name": "Build container", "action": build_image},
    {"name": "Deploy service", "action": deploy_service}
]
result = await plan.execute_plan(steps, rollback_on_failure=True)

# Isolated speculative rollout for subagent reasoning
async with plan.speculative_rollout(name="worker_sandbox") as ctx:
    ctx.emit_effect({"path": "temp_draft.md", "content": "# Draft"})
    # All effects automatically unwound cleanly upon exit
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
