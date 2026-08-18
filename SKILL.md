---
name: plan
description: "Plan mode - formal planning engine, STRIPS causal validator, evolutionary AST search, Cordis spatiotemporal composability runtime, and iterative self-improving plan synthesis. Use when the user invokes /plan or asks to plan something robustly, refine a plan, or enter plan mode."
---

# Plan Mode (/plan)

Plan mode drafts a formal plan for an objective and iteratively refines it round by round until convergence. Every version, score, and critique is persisted under `plans/<session_id>.json`, guaranteeing auditable convergence, STRIPS causal soundness, and zero unaddressed flaws.

## Theoretical Foundations

- **Cordis Spatiotemporal Composability** (*Shi, Zhang, Cui 2026*): Unified context $\Gamma_\infty$, Provider Stack dynamic resolution, Theorem 6.3 reverse-topological DFS deactivation, hash-verified journaling with drift detection, and Twisted Monoid $(f_1, g_1) \circ (f_2, g2) = (f_1 \circ f_2, g_2 \circ g_1)$ LIFO inverse unwinding.
- **STRIPS Causal Validation** (*SymPlanner 2505.01479, GNNVerifier 2603.14730*): AST causal link tracking $\langle a_i, p, a_j \rangle$, clobber threat detection, closed-world negation, and localized type-mismatch verification.
- **Evolutionary AST Search** (*PlanBench 2409.13373, LATS 2310.04406, SYMPHONY 2601.22623*): AST subgraph crossover with semantic dependency remapping, flaw-directed and exploratory mutations, and Jaccard diversity tracking.
- **Rule of Thought (RoT) Memory Distillation** (*2404.05449*): Causal flaws distilled into persistent structural rules enforced on subsequent plans.
- **Hierarchical Re-planning Ladder** (*RePLan 2401.04157, 2605.25851*): 4-tier escalation (L1 Subgoal Audit $\to$ L2 Structured Search $\to$ L3 Global Redraft $\to$ L4 Runtime Drift Recovery) with HIPIF context compression (*2606.10507*).
- **Execution Contracts** (*ACID-Agent 2608.13900, FlowScout 2608.10039*): released plans must declare verification commands, artifact budgets, parity checks, and symbol contracts; a minimal probe must pass before full implementation.

---

## Standard Protocol (When `/plan` is Invoked)

The parent agent **ORCHESTRATES**; subagents execute reasoning and drafting. The Python engine deterministically scores (rubric + STRIPS causal validator + simulation), searches (AST evolutionary / MCTS / Beam), locks sessions across parallel agents, and enforces release gates.

```
                  ┌─────────────────────────────────────────┐
                  │          Parent Orchestrator            │
                  └──────┬───────────────────────────▲──────┘
                         │                           │
            1. plan.start()                 7. plan.finish()
            2. spawn planner subagent       6. plan.release()
            3. plan.assess()                5. plan.record_judge()
            4. plan.search(mode="ast")
                         │                           ▲
                         ▼                           │
                  ┌─────────────────────────────────────────┐
                  │    Deterministic Validation Engine      │
                  │   - STRIPS Causal Validator (Links)     │
                  │   - Grounded Feasibility (Disk check)   │
                  │   - State Simulation (SymPlanner)       │
                  │   - RoT Learned Structural Rules        │
                  │   - Reentrant Session Lock (POSIX flock)│
                  └─────────────────────────────────────────┘
```

### 1. Initialize Session
```python
import plan
s = plan.start("Objective description", max_rounds=8)
```
Resumes an existing active session for the objective if present under `plans/`.

### 2. Draft Candidate Plan (Planner Subagent)
Spawn a planner subagent (`await rlm('Draft a complete plan for: ...', name='planner')`). The planner drafts the plan with explicit numbered tasks, dependencies, environment inputs, and concrete artifact outputs:
```markdown
# Objective: <Goal>

## Success Criteria
- S1: Numeric criteria 1
- S2: Numeric criteria 2

## Tasks
1. Extract Dataset
   Inputs: raw_data.csv
   Output: extracted.json
2. Transform Features
   Depends on 1
   Inputs: extracted.json
   Output: features.parquet
3. Train Classifier
   Depends on 2
   Inputs: features.parquet
   Output: model.bin
```

Every implementation plan must also contain an `## Execution Contract`. The planner writes a minimal spike file first and lists every function/variable it intends to create, so stubs and undeclared helpers are detectable:
```markdown
## Execution Contract
```json
{
  "probe": {"command": ["python", "spike.py"], "expected_output": "runner=40"},
  "verification_commands": [["python", "-m", "pytest", "-q"], ["ruff", "check", "."]],
  "expected_artifacts": {"runner.py": {"min_lines": 60}},
  "symbols": {"runner.py": {"functions": ["main", "run_profile"], "variables": ["PROFILES"]}},
  "parity_checks": [{"left": "legacy_runner", "right": "runner", "algorithm": "sha256"}]
}
```
```

### 2b. Execution Evidence & Negative Constraints
After implementation, the executor writes an `## Execution Evidence` JSON trace with real command outputs. The verifier must be a different agent and independently check:
- `plan.verify_execution_trace(plan_text, evidence)`
- `plan.verify_negative_constraints(plan_text, evidence)`
- `plan.run_exit_criteria(contract, cwd=repo)`
- `plan.symbol_audit(plan_text, cwd=repo)`

A plan may also declare falsifiers:
```markdown
## Negative Constraints
- NF-1: declared output exists but symbol audit fails.
- NF-2: command exits 0 but stdout lacks `must_contain`.
```

### 3. Assess Draft and Probe Feasibility
```python
res = plan.assess(s, draft_plan_text, note="Initial draft",
                  require_execution_contract=True, run_probe=True)
# res: {"version": 1, "score": 85.0, "delta": None, "critiques": [...], "status": "improving", "continue": True}
```
If `run_probe=True` and the spike fails, `assess()` emits `mech:probe:*` critiques. The plan must be revised until the spike produces the declared output.
`assess()` executes the full validation pipeline:
1. **Rubric Scoring**: Mechanical, domain, and structure checks (`RUBRIC.md`).
2. **Causal Validation**: Validates precondition satisfaction and clobber threats.
3. **Grounded Feasibility**: Verifies undeclared environment inputs against real disk paths.
4. **SymPlanner State Simulation**: Simulates task graph execution against an explicit world state.
5. **RoT Memory Enforcement**: Checks against distilled historical rules.

### 4. Search Plan Space (Evolutionary AST / MCTS)
Run an evolutionary AST search to automatically repair flaws and discover higher-scoring executable plans:
```python
search_res = await plan.search(s, iterations=4, width=3, mode="ast", cwd=Path.cwd())
# Automatically commits the winning plan to the session
```
Or use subagents for parallel candidate expansion with Best-of-N selection:
```python
winner = plan.assess_candidates(s, [draft_a, draft_b, draft_c])
```

### 5. Adversarial Judge Verification
Spawn a dedicated judge subagent (`await rlm('Review plan feasibility adversarially...', name='plan-judge')`) and persist the verdict:
```python
plan.record_judge(s, {
    "verdict": "go",
    "feasibility_0_100": 95,
    "blockers": [],
    "contradictions": [],
    "falsifiable_criteria": True
})
```

### 6. Release & Finish Gate
Enforce strict acceptance thresholds before executing:
```python
# Verifies convergence, score >= min_score, clean verify(), clean simulation, and judge "go"
plan.finish(s, require_release=True, min_score=90.0,
           require_execution_contract=True)
```
Release fails while the execution contract is missing/invalid. After implementation, Prime Agent runs `plan.symbol_audit(plan_text, cwd=repo)` and the independent verifier subagent runs the contract commands before reporting done.

### 7. Transactional Execution (Cordis Runtime)
Execute the released plan transactionally with automatic LIFO rollback on failure:
```python
async def run_task1(ctx):
    # execute task 1 and register revertible effect
    ctx.effect(lambda: lambda: Path("extracted.json").unlink(missing_ok=True))
    return {"status": "extracted"}

results = await plan.execute_plan(best_plan_text, task_handlers={1: run_task1})
# If any task fails, all intermediate mutations are automatically rolled back in LIFO order
```

---

## Full API Reference

| Function / Primitive | Description |
|---|---|
| `plan.start(obj, plans_dir=None, max_rounds=8)` | Start or resume an active session by objective slug. |
| `plan.assess(session, plan_text, note=None, addressed=None)` | Score plan, run STRIPS + simulation + RoT checks under re-entrant session lock. |
| `plan.assess_candidates(session, drafts, notes=None)` | Best-of-N ranking across candidate plan variations. |
| `await plan.search(session, iterations=4, width=2, mode="ast"\|"mcts"\|"beam", cwd=None)` | Search plan space with AST evolution / MCTS / Beam search; auto-commits winner. |
| `plan.verify(plan_text, cwd=None)` | Deterministic structural & STRIPS causal audit (no LLM). |
| `plan.ground_check(plan_text, cwd=None)` | Verify declared environment inputs exist on disk. |
| `plan.simulate(plan_text, initial_state=None)` | Execute plan against explicit state model (SymPlanner). |
| `plan.record_judge(session, verdict)` | Persist judge verdict into session under lock. |
| `plan.committed(session)` | Return the last successfully released plan (explored best is separate from committed). |
| `plan.validate_execution_contract(plan_text, cwd=None)` | Parse and statically validate the `## Execution Contract` JSON block. |
| `plan.probe_contract(plan_text, cwd=None)` | Run the minimal feasibility spike; failed probes force plan revision. |
| `plan.symbol_audit(plan_text, cwd=None)` | Compare declared functions/variables against actual source files to catch stubs and undeclared helpers. |
| `plan.run_exit_criteria(contract, cwd=None)` | Run structured exit criteria; checks stdout/must_contain/expected_count, not only exit code. |
| `plan.verify_execution_trace(plan_text, evidence)` | Align plan obligations with real execution evidence; rejects stubs and missing symbols. |
| `plan.verify_negative_constraints(plan_text, evidence)` | Check declared falsifiers against execution evidence. |
| `plan.checkpoint(session, note=None)` / `plan.rewind(session, checkpoint_id=None)` | AgentRewind-style aligned session checkpoints and rollback. |
| `plan.release(session, min_score=90.0, require_judge=True)` | Confidence-gated checkpoint release validation; successful release commits `best` -> `committed`. |
| `plan.finish(session, require_release=True, min_score=90.0)` | Complete session after verifying release gate. |
| `plan.log_progress(session, task, status, evidence=None)` | Record execution progress; triggers 4-tier replanning ladder on failure (L1 local -> L4 drift recovery). |
| `plan.fold_history(session, keep_last=2, max_context_tokens=4000)` | HIPIF context token compression for older rounds. |
| `await plan.execute_plan(plan_text, task_handlers=None, timeout_per_task=None)` | Transactional plan execution with Cordis LIFO rollback. |
| `plan.speculative_rollout(plan_text, eval_fn)` | Isolated speculative execution with guaranteed 100% clean recovery. |
| `plan.create_subagent_context(name)` | Derive an isolated child context ($\Gamma_\infty$) for subagent sandboxing. |
| `plan.provide_tool(key, value)` | Ephemeral tool/capability provision with LIFO inverse cleanup. |
| `plan.selfcheck(plans_dir=None, run_pytest=True)` | Mandatory self-evaluation audit of rubric, sessions, and test suites. |

---

## Concurrency & Safety Model

- **Reentrant Process/Thread Lock (`session_lock`)**: All state-mutating entrypoints (`assess`, `record_judge`, `log_progress`, `release`, `finish`, `fold_history`) execute within an in-process thread-aware depth tracker backed by POSIX `fcntl.flock`, eliminating race conditions across parallel subagents without self-deadlock.
