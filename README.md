# Prime Agent Plan Mode (`/plan`)

**Make the plan better before the agent starts doing the work.**

Plan Mode is a planning skill for Prime Agent that turns a one-shot LLM plan into an **iterative, checked, auditable planning process**. Instead of accepting the first plan that sounds reasonable, `/plan` repeatedly drafts, critiques, verifies, simulates, compares, and revises candidate plans until the plan is strong enough to pass a release gate.

It is designed for tasks where a weak plan is expensive: software projects, research workflows, deployments, multi-agent work, long-running objectives, migrations, experiments, and other jobs with dependencies, artifacts, risks, deadlines, or rollback requirements.

> **Core idea:** a plan should not be accepted only because it is well-written. It should also be structurally valid, executable under its stated assumptions, measurable, and able to survive adversarial checking.

---

## Why use Plan Mode?

A normal planning prompt often produces one plausible-looking answer and stops. That can hide missing dependencies, vague success criteria, impossible ordering, ungrounded artifacts, or assumptions that only become visible during execution.

Plan Mode adds a planning loop around the agent:

```text
Objective
   ↓
Draft plan
   ↓
Rubric assessment + deterministic checks
   ↓
STRIPS-style simulation
   ↓
Critiques / blockers
   ↓
Generate multiple revisions
   ↓
Score + select the best candidate
   ↓
Independent judge / release gate
   ↓
Released plan
```

The important difference is that **failure creates another planning round instead of being ignored**.

### Normal LLM planning vs Plan Mode

| Normal planning | Plan Mode |
|---|---|
| Usually one draft | Multi-round improvement |
| Self-review is mostly prose | Deterministic checks + rubric scoring |
| Dependencies may be implied | Dependency graph is verified |
| Artifacts can appear from nowhere | Inputs/outputs are tracked and simulated |
| Often one candidate | Best-of-N / tree-search candidate exploration |
| “Looks good” can end the process | Release gate must pass |
| Revision history is easy to lose | Versions, scores, critiques, and judge results are persisted |

---

## What happens when you run `/plan`?

Plan Mode uses the parent agent as the orchestrator and keeps the scoring and verification engine separate from plan generation.

1. **Start an objective** — a planning session is created or resumed.
2. **Draft** — a planner agent produces a complete plan.
3. **Assess** — the engine scores the plan against the planning rubric and mechanical checks.
4. **Verify** — task numbering, dependencies, artifacts, milestones, deadlines, and graph structure are checked.
5. **Simulate** — a STRIPS-style forward simulation checks whether tasks can actually execute in order from the declared state.
6. **Revise** — unresolved critiques become requirements for the next version.
7. **Search** — multiple candidate revisions can be generated and compared using best-of-N, beam search, or MCTS-style search.
8. **Judge** — an independent judge can challenge contradictions, missing inputs, assumptions, and unfalsifiable success criteria.
9. **Release** — the plan is released only after the configured release conditions pass.

Every round is persisted, so improvement can be inspected instead of merely claimed.

---

## Quick start with Prime Agent

### Option A — Install globally

Use this if you want `/plan` available across Prime Agent projects.

```bash
mkdir -p ~/.prime/agent/skills
git clone https://github.com/lewbei/prime-agent-plan.git ~/.prime/agent/skills/plan
```

Then reload the skills in Prime Agent and invoke:

```text
/plan Design and implement a production-ready authentication service with tests, migration steps, rollback, and measurable release criteria.
```

Depending on the Prime Agent interface, the skill can also be invoked as:

```text
/skill:plan <your objective>
```

### Option B — Install only for one project

From the project root:

```bash
mkdir -p .prime/agent/skills
git clone https://github.com/lewbei/prime-agent-plan.git .prime/agent/skills/plan
```

Or keep it as a submodule:

```bash
git submodule add https://github.com/lewbei/prime-agent-plan.git .prime/agent/skills/plan
```

---

## Example

Suppose the objective is:

```text
/plan Migrate a production API from PostgreSQL 15 to PostgreSQL 17 with less than 5 minutes of downtime and a tested rollback path.
```

A useful Plan Mode result should not stop at:

```text
1. Back up the database.
2. Upgrade PostgreSQL.
3. Test the application.
```

The planner is pushed toward a plan containing things such as:

- explicit objective and scope;
- measurable acceptance criteria;
- current-state assumptions and unknowns;
- ordered tasks and dependency references;
- declared inputs and output artifacts;
- checkpoints and go/no-go gates;
- verification commands or deterministic checks;
- failure modes and mitigations;
- rollback and recovery steps;
- resource/time budgets;
- evidence that each stage succeeded;
- replanning triggers if reality differs from the assumptions.

If a later task depends on an artifact that no earlier task produces, or if the task graph is invalid, the mechanical checks can block convergence instead of allowing the plan to pass because the prose sounds convincing.

---

## When should you use it?

Plan Mode is most useful when the objective has **real structure or real consequences**.

Good use cases include:

- software architecture and implementation plans;
- repository-wide changes and refactors;
- deployment and migration planning;
- research projects and experiment pipelines;
- data/ML training and evaluation workflows;
- multi-agent task decomposition;
- long-running goals with checkpoints;
- plans involving external dependencies or artifacts;
- work that needs rollback, recovery, or stop conditions;
- plans where success must be measurable rather than subjective.

For a tiny task such as renaming one variable or answering a simple factual question, `/plan` is probably unnecessary overhead.

---

## The checks are not only LLM opinions

One design goal of this project is to avoid relying entirely on an LLM saying that another LLM's plan is good.

The engine includes deterministic and mechanically checkable components for:

- plan rubric scoring;
- task numbering and duplicate detection;
- dependency validation;
- forward-reference detection;
- cycle detection;
- artifact production/consumption tracking;
- milestone/task consistency;
- deadline arithmetic;
- grounded input checks;
- STRIPS-style state simulation;
- convergence tracking;
- candidate scoring and selection;
- release gating.

LLM-based planning and judging can be used around those checks, but the core structural checks remain separate.

---

## Release gate

`release()` is intended to stop a weak plan from being treated as finished too early.

A release can require all of the following:

```text
converged
AND score >= threshold
AND verify() is clean
AND simulate() executes end-to-end
AND independent judge verdict == "go"
AND judge criteria are falsifiable
```

The exact threshold and judge requirement are configurable through the API.

This does **not** mean the released plan is guaranteed to succeed in the real world. It means the plan has passed the checks represented by the engine and the information available to it. Unknown external facts can still invalidate a plan, so important assumptions should be grounded before execution.

---

## Python API

The skill re-exports the planning engine through `plan`.

### Start, assess, and inspect a session

```python
import plan

session = plan.start(
    "Design and deploy a distributed stream-processing pipeline"
)

assessment = plan.assess(session, draft_plan_text)

print("score:", assessment["score"])
print("status:", assessment["status"])
print("critiques:", assessment["critiques"])

print(plan.status(session))
print(plan.best(session))
```

### Verify a plan mechanically

```python
result = plan.verify(plan_text)

if not result["ok"]:
    for error in result["errors"]:
        print(error)
```

### Simulate task execution

```python
simulation = plan.simulate(plan_text, initial_state={})

print("executable:", simulation["executable_plan"])
print("problems:", simulation["problems"])
print("trace:", simulation["trace"])
```

### Compare several candidate plans

```python
candidates = [draft_a, draft_b, draft_c]
result = plan.assess_candidates(session, candidates)

print(result)
```

### Search plan space

```python
result = await plan.search(
    session,
    iterations=4,
    width=3,
    mode="mcts",
)

print("best score:", result["best_score"])
print("best plan:", result["best_plan"])
```

`search()` supports LLM-based expansion when the required provider is available and deterministic/rule-based fallback paths for offline use.

### Check the release gate

```python
release_status = plan.release(
    session,
    min_score=90,
    require_judge=True,
)

print(release_status)
```

---

## Multi-agent planning pattern

The recommended Prime Agent pattern is not “one giant agent thinks about everything.”

```text
Parent / Orchestrator
├── Planner A ──┐
├── Planner B ──┼──> deterministic assessment / simulation ──> best candidate
├── Planner C ──┘
└── Independent Judge ──> blockers / go / rework
```

The parent agent owns the loop. Planner/reviser subagents generate alternatives. The engine scores and verifies them. A different judge challenges the selected plan before release.

This separation makes it harder for one generation path to both create the mistake and approve the same mistake.

---

## Plan search

For difficult objectives, Plan Mode can search over plan candidates instead of revising only one linear draft.

Available mechanisms include:

- best-of-N candidate assessment;
- UCB1 node selection;
- MCTS-style expansion/backpropagation;
- beam-style search;
- critique-aware mutation;
- plan recombination;
- plateau detection and adaptive search width;
- transposition tracking;
- search reports for auditability.

Useful API entry points:

```python
await plan.search(...)
plan.search_expand(...)
plan.search_select(...)
plan.search_backtrack(...)
plan.search_report(...)
```

---

## Transactional execution and Cordis

The repository also contains the **Cordis Spatiotemporal Composability Runtime** used for isolated contexts, lifecycle management, and reversible effects.

At a practical level, Cordis is intended to support:

- child contexts for subagents;
- isolated speculative rollouts;
- effect journaling;
- LIFO rollback;
- reversible file/tool mutations;
- transactional plan execution;
- cleanup when a rollout fails.

Example:

```python
result = await plan.execute_plan(
    plan_text,
    task_handlers=handlers,
)
```

For speculative evaluation:

```python
result = await plan.speculative_rollout(
    candidate_plan,
    eval_fn,
)
```

Lower-level primitives are also exported:

```python
plan.Context
plan.Fiber
plan.LifecycleState
plan.TwistedMonoid
plan.create_subagent_context(...)
plan.provide_tool(...)
```

You do not need to understand the underlying composability calculus to use `/plan`; it is an advanced runtime layer for integrations that need isolation and rollback semantics.

---

## Persistence and audit trail

Planning sessions are stored under:

```text
plans/<session_id>.json
```

The stored session can include plan versions, scores, critiques, best-plan state, search state, progress records, and judge results.

Useful inspection APIs include:

```python
plan.status(session)
plan.history(session)
plan.best(session)
plan.list_sessions()
```

This is what makes “the plan improved” inspectable: the previous versions and score changes are retained rather than replaced silently.

---

## Replanning during execution

Planning does not have to stop once execution begins.

```python
plan.log_progress(
    session,
    task="database migration dry run",
    status="blocked",
    evidence="extension X is incompatible with PostgreSQL 17",
)
```

A blocked or failed execution step can become a replanning signal. The intended policy is to repair the smallest invalid part of the plan while preserving the valid prefix instead of regenerating everything unnecessarily.

---

## Repository structure

```text
.
├── README.md
├── SKILL.md
├── pyproject.toml
├── src/
│   ├── plan/
│   │   └── __init__.py
│   └── plan_mode/
│       ├── __init__.py
│       ├── cordis.py
│       ├── judge_client.py
│       ├── search_engine.py
│       └── RUBRIC.md
└── tests/
    ├── README.md
    ├── conftest.py
    ├── test_cordis.py
    ├── test_engine.py
    └── test_rubric.py
```

`SKILL.md` contains the Prime Agent orchestration protocol. `src/plan_mode/` contains the deterministic planning engine. `src/plan/` is the public skill entry point and re-exports the main APIs.

---

## Testing

From the repository root:

```bash
python3 -m pytest tests/ -q
```

The test suite covers the rubric, structural verification, simulation, search behavior, history handling, and Cordis runtime behavior. The documented rule-based search tests do not require network access.

After changing the planning engine or rubric, the engine also provides:

```python
plan_mode.selfcheck()
```

for repository-level re-evaluation and regression checking.

---

## Design principles

Plan Mode follows several principles throughout the implementation:

1. **Do not trust the first plan.** Generate, test, and revise it.
2. **Do not let the planner be the only verifier.** Keep deterministic checks and independent judgment separate.
3. **Make success falsifiable.** “Looks good” is not an acceptance criterion.
4. **Track artifacts and dependencies.** A task cannot consume something that was never produced.
5. **Simulate before release.** Structural errors should be found before execution when possible.
6. **Keep history.** Improvement should be auditable.
7. **Prefer local repair over unnecessary restart.** Preserve valid work when only one part fails.
8. **Rollback side effects when speculative execution fails.** Exploration should not silently pollute the parent state.

---

## Research foundations

The implementation is research-driven and incorporates ideas from work on LLM planning, mechanical verification, symbolic simulation, tree search, ensembles, failure prevention, context management, judgment, and composability.

Key influences documented in the repository include:

| Area | Examples used by the project | What is applied |
|---|---|---|
| Mechanical planning verification | PlanBench, Natural Plan, LLM-Modulo | External/mechanical checks instead of prose-only self-review |
| Symbolic state simulation | SymPlanner, PyPDDLEngine, SimPlan | Preconditions, effects, state transitions, artifact flow |
| Plan-space search | LATS, SYMPHONY, CB-MCTS, GATS | Candidate expansion, UCB selection, backtracking, search accounting |
| Plan ensembles and recombination | Mind Evolution, Plan Ensembles, diversity-maintenance work | Multiple candidate plans, evaluator selection, recombination |
| Failure prevention | PPA-Plan, RoT, verifier work | Failure enumeration, critique grouping, learned planning constraints |
| Context and budget management | HIPIF, SafeRun, BRACE | History folding, budget checks, revision control |
| Release judgment | Tri-Judge / confidence-gating work | Independent judgment and acceptance thresholds |
| Composability | Cordis | Context composition, fibers, lifecycle control, reversible effects |

See [`SKILL.md`](SKILL.md) and [`src/plan_mode/RUBRIC.md`](src/plan_mode/RUBRIC.md) for the implementation-facing protocol and literature-linked rubric details.

---

## Current status and limitations

This repository is an **experimental planning and agent-runtime implementation**, not a mathematical guarantee that an objective will succeed.

Important limitations:

- deterministic checks only verify properties they are programmed to inspect;
- a structurally valid plan can still rely on a false real-world assumption;
- LLM candidate quality still depends on the model and context supplied to the planner;
- external judge quality depends on the judge model/provider;
- simulation is an abstraction of the declared plan state, not the full real world;
- transactional rollback only covers effects represented through the runtime's reversible mechanisms.

For high-impact work, treat Plan Mode as a stronger planning and verification layer, not as a replacement for domain-specific testing, security review, human approval, or production safeguards.

---

## Contributing

Useful contributions include:

- new deterministic plan checks;
- stronger simulation rules;
- benchmark cases where plausible plans should fail;
- search-policy improvements;
- judge robustness tests;
- additional reversible effect handlers;
- reproducible planning benchmarks;
- documentation and real-world usage examples.

When changing the engine or rubric, run the test suite and the project self-check before treating the change as complete.

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for full legal text and permissions.


---

## 🔬 Core Formal Architecture

### 1. Deterministic Causal Validator (`causal_validator.py`)
- **Action Schema & AST**: Parses plans into typed `ActionSchema` objects with declared `preconditions`, `add_effects`, `del_effects`, `inputs`, `outputs`, `duration`, and `cost`.
- **STRIPS State Transition Solver**: Emulates forward world-state evolution $\mathcal{S}' = (\mathcal{S} \setminus 	ext{Del}(a)) \cup 	ext{Add}(a)$ with propositional consistency.
- **Causal Link Construction**: Tracks triples $\langle a_i, p, a_j 
angle$ asserting that step $a_i$ achieves condition $p$ consumed by step $a_j$.
- **Clobber Threat Detection**: Identifies any intermediate action $a_k$ ($i < k < j$) that deletes $p$, producing pinpoint diagnostic flaws with automated remedies.
- **Numeric & Resource Budget Solver**: Deterministically checks linear constraints on tasks, durations, and token budgets.

### 2. Complete Cordis Composability Engine (`cordis.py`)
- **$\Gamma_\infty$ Context Monoid**: Dynamic context with dual sync/async effect registration (`effect` / `async_effect`) and dual LIFO rollback (`dispose` / `async_dispose`).
- **Algorithm 1 Effect Iterators with Cancellation**: Step-boundary async generators with cancellation tokens $	au$ and instant LIFO unwinding of in-flight mutations.
- **Theorem 63 Topological Provider Withdrawal**: When a service provider withdraws, all active dependent fibers deactivate in reverse topological order before the binding unmounts.
- **Cryptographic Hash Journaling**: Records SHA-256 pre-state and post-state hashes to detect external state drift during rollbacks.
- **Declarative Component Loader (`ctx.use`)**: Component lifecycle instantiation and reactive coeffect binding.

### 3. Structural AST-Level Evolutionary Search (`ast_search.py`)
- **AST Subgraph Crossover**: Recombines valid causal subgraphs from top parent plans.
- **Flaw-Directed Mutations**: Directly patches flaws identified by `CausalValidator` (inserting missing producers, reordering conflicting steps, renumbering).
- **Population Diversity Tracking**: Computes Jaccard graph edit distance $D(P_a, P_b)$ and applies Pareto multi-objective selection (fitness + diversity bonus).
- **State Transposition Table**: Collapses search space by caching state and DAG hashes.

### 4. RoT Memory Distillation & Context Management (`memory_distiller.py`)
- **Rule of Thought (RoT) Distiller**: Automatically distills negative rules `Rule(id, trigger, pattern, remedy, perspective)` from structural/execution flaws to prevent repeat errors.
- **Tree-of-Experience Rule Namespaces**: Rules are grouped by perspective and updated with environmental success/failure outcomes (2608.09044).
- **HIPIF Context Budgeter**: Compresses superseded rounds into compact semantic diffs while preserving full text for the best and latest rounds.
- **4-Tier Replanning Ladder**: Escalates failures systematically from local task parameter adjustments (Tier 1) to subgraph replanning (Tier 2) to global strategy upgrades (Tier 3) to runtime drift recovery (Tier 4, 2608.14109).

### 5. Agentic Transactions & Aligned Rewind
- **Explore / Commit Separation**: `best()` is the best explored plan; `committed()` is the last successfully released plan. Only `release()` promotes best to committed.
- **Aligned Session Checkpoints**: `checkpoint()` stores a deep session snapshot; `rewind()` restores rounds, search tree, execution log, and world state with a rewind-log entry (AgentRewind 2608.14380).
- **Search Checkpoints**: `search(..., checkpoint_before=True)` records a recoverable checkpoint before expansion begins.
