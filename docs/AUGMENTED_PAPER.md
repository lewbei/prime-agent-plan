# Prime Agent Plan — Augmented Paper Draft (2026-08-17)

**Status:** Working draft for the next paper revision.
**Scope:** Augments the current `prime-agent-plan` engine and Cordis/planning literature digest with arXiv papers published 2026-08-13 through 2026-08-14, plus the most relevant late-July/August planning-agent work.
**Current engine commit under discussion:** `4a275fc` / `aaee619` line of work.

---

## 1. Search method

Queried arXiv API on 2026-08-17 with:

- `all:"LLM planning"`
- `all:"agent planning"`
- `all:"plan verification" AND all:"large language models"`
- `all:"tree search" AND all:"large language models"`
- `all:"self-evolving agent"`
- `all:"context management" AND all:"agents"`
- `submittedDate:[202608130000 TO 202608172359] AND (cat:cs.AI OR cat:cs.MA OR cat:cs.CL OR cat:cs.SE) AND (all:"planning" OR all:"plan" OR all:"agent")`

Full texts of the eight most relevant papers were downloaded into the local corpus directory `deep_learning/planning_paper/latest_aug/`.

---

## 2. High-relevance new papers

| Paper | Date | Why it matters for us |
|---|---|---|
| **AgentRewind** `2608.14380` | 2026-08-14 | Aligned checkpoints of LLM context + workspace; agent-selected rewind with rewind memory; MettleBench. Direct comparison/extension target for Cordis rollback. |
| **Agentic Transaction / ACID-Agent** `2608.13900` | 2026-08-14 | Semantic Atomicity, Consistency, Isolation, Durability for agent workflows. Gives a formal language for what our release gate + session lock + journal already do informally. |
| **Graph-based drift diagnosis/recovery** `2608.14109` | 2026-08-14 | Runtime behavioral drift classified by a small recovery-graph model with step-level risk and recovery decisions. Extends our 3-tier replan ladder from heuristic to learned, structured recovery. |
| **Capability Sheaves** `2608.13228` | 2026-08-13 | Formal treatment of locally successful harness components disagreeing on shared state. Complementary to Cordis coeffects; useful for auditing multi-agent state conflicts. |
| **Tree-of-Experience** `2608.09044` | 2026-08-10 | Hierarchical experience tree aligned with reasoning paths; better feedback attribution and transfer than flat trajectory memory. Direct upgrade for our flat RoT rule base. |
| **FlowScout** `2608.10039` | 2026-08-10 | Execution-feedback-guided MCTS over tool/LM dependency graphs. Direct upgrade for our AST search: currently structural only, not execution-guided. |
| **Not Worth Another Token** `2608.08389` | 2026-08-09 | Stage-aware marginal-value pruning; early pruning saves up to 73% tokens. Directly improves `fold_history()`/`ContextBudgeter`. |
| **STAIR** `2608.09524` | 2026-08-10 | Graph-as-State incident planning, stage router, execution feedback, experience reuse. A good template for making our planning state graph-structured and stage-specialized. |
| **RippleMem** `2608.13334` | 2026-08-13 | Associative recollection from cue-rich episodic memory. Improves long-term memory beyond substring/AST RoT matching. |
| **MemoryLake / MemoryArena** `2608.13883` | 2026-08-14 | Matched system-level comparison of memory backends for multi-session tasks. Gives us an evaluation protocol for session memory. |
| **OEO: Rethinking Self-Evolving Agents** `2608.09629` | 2026-08-10 | Open-ended optimization can beat prescribed self-evolution pipelines when a frontier optimizer is available. Forces us to ablate our hardcoded replan/search pipeline. |
| **Second Thought** `2608.13667` | 2026-08-13 | Uses action/observation idle time for parallel reasoning. Relevant to our multi-agent planning pattern and subagent workers. |
| **Demystifying Agent Skills** `2608.14036` | 2026-08-14 | Skills work as procedural anchors; why/when they fail. Directly informs `SKILL.md` and rubric-template design. |

Other relevant August entries: CEDAR `2608.06871`, PACE-Bench `2608.14441`, AgentRewind-adjacent checkpoint work, `2608.04265` strategic planning evaluation, `2608.04661` repository Agent Plans.

---

## 3. Updated abstract (draft)

Planning agents are usually evaluated on whether they produce a plausible plan. We argue the more useful object is a *durable, verifiable planning artifact*: a plan whose causal structure has been mechanically checked, whose exploration effects can be reversed, and whose execution history can be replayed and audited. We present Prime Agent Plan, a planning engine that combines (i) STRIPS-style causal validation over typed action schemas, (ii) evolutionary AST search with flaw-directed mutation, and (iii) a Cordis spatiotemporal-composability runtime with LIFO inverses, provider-stack dependency resolution, and hash-verified rollback. We then extend this engine with two ideas from the newest agent-reliability literature: **agentic transactions** (`2608.13900`) and **aligned context/environment rewind** (`2608.14380`). Specifically, we formalize plan search, judge validation, and execution as semantic transactions with commit-or-retry semantics, and we add aligned checkpoints of the planner context and workspace so that a failed refinement branch can be rewound with accumulated negative evidence. We evaluate on PlanBench, MettleBench, Terminal-Bench 2.0, and a new CausalPlan suite measuring unsatisfied-precondition, clobber-threat, rollback-drift, and multi-agent interference rates. Our goal is to show that treating planning as *ACID-compliant spatiotemporal composition* improves long-horizon success and partial progress more than stronger base models alone.

---

## 4. What the new papers change in the paper

### 4.1 From "rollback runtime" to "agentic transactions" (ACID-Agent)

Our current system already has the raw pieces:

| ACID-Agent property | Our current mechanism | Gaps to close |
|---|---|---|
| Semantic Atomicity | `execute_plan` LIFO rollback; speculative rollouts | No explicit commit point; search improvements are committed implicitly by `assess()` |
| Semantic Consistency | rubric + `verify()` + causal validator + judge + release gate | Validation is plan-level, not transaction-level; no confidence-divergence signal |
| Semantic Isolation | `session_lock`, `Context.isolate`, per-session files | Locking is coarse; no declared isolation levels for subagents sharing plans/artifacts |
| Semantic Durability | `plans/<id>.json`, effect journal, RoT rules, judge log | Journal is not append-only by default; no hash-chained audit trail of commits |

**Augmented contribution:** define a *planning transaction* as

```
T = (objective, initial_state, candidate_plans, validators, judge, commit_policy)
```

with commit only after `verify ∧ simulate ∧ release ∧ judge = go`. Failed exploration branches are rolled back and their negative evidence is retained in RoT, but never committed to the session best plan.

### 4.2 From "effect journal" to "aligned rewind" (AgentRewind)

AgentRewind shows that workspace rollback alone is insufficient: the LLM context must be restored **with the same boundary as the environment**.

Our current Cordis layer journals effects, but does not checkpoint agent context. Proposed mechanism:

1. Before each planning/search expansion, record `(session_state_hash, plan_text_hash, context_summary)`.
2. On failure, restore the session and workspace to the selected checkpoint.
3. Inject **rewind memory**: a distilled summary of falsified hypotheses and the critique IDs that caused rewind.
4. Resume from the restored prefix instead of regenerating.

This maps naturally to `search_backtrack()` and `log_progress()`.

### 4.3 From flat RoT rules to Tree-of-Experience

Our RoT base is a flat `rule_id -> (trigger, forbidden_pattern, remedy)` map. Tree-of-Experience suggests:

```
experience tree = analytical perspective nodes
                 -> reasoning path nodes
                 -> outcome-calibrated leaf rules
```

Concrete upgrade:

- `RoTRuleBase` stores rules under `(flaw_type, predicate, resource)` but also under `strategy/perspective` paths.
- Update reliability by environmental outcome, not only occurrence.
- Retrieval follows the current plan's reasoning path.
- Confidence becomes an empirical success/failure ratio, not `+0.1` on repeated hits.

### 4.4 From static verification to runtime drift diagnosis

`2608.14109` provides a recovery graph:

```
classify drift -> detect operation -> assess risk -> choose recovery action
```

We can instantiate this as a fourth tier of our replan ladder:

- **L1:** local parameter repair
- **L2:** subgraph replan
- **L3:** global redraft
- **L4:** runtime drift recovery node (small model / deterministic policy) using `log_progress` evidence and session world state

This is especially useful for silent failures that pass structural validation but corrupt external state.

### 4.5 From structural AST search to execution-guided search

FlowScout and CEDAR both use execution feedback to guide graph/MCTS search. Our AST search currently uses validator flaws as fitness. Augmented design:

- Rollout value = `rubric + verify + simulate + actual task-handler feedback`.
- Keep a tool/execution graph mined from successful sessions.
- Mutate topology only where execution feedback indicates a dependency or tool mismatch.
- Use CEDAR-style LLM Judge/Editor separation when an API worker is available.

### 4.6 Context budget with marginal-value pruning

`2608.08389` shows stage matters more than scoring rule. Our `ContextBudgeter` currently compresses older rounds uniformly. Proposed:

- Pre-planning: keep only objective, constraints, last best plan, active RoT rules.
- Post-validation: keep critique IDs + localized fixes, not full failed plans.
- Pre-release: keep judge/verification evidence, not raw drafts.
- Fold only when marginal token value is below a learned threshold.

### 4.7 Harness-level consistency via Capability Sheaves

Capability Sheaves formalizes disagreement between locally successful harness components. Our system has the same failure class: planner, verifier, judge, and executor each see different state.

Proposed augmentation:

- Model each component as a stalk with typed behavior signature.
- Model session/plans/artifacts as restriction maps.
- A plan is accepted only if it is a global section of this sheaf.
- Use the paper's relative cohomology diagnostic as an optional search feature for "which component disagrees."

This can be a formal comparison point: Cordis coeffects give runtime dependency resolution; capability sheaves give static/structural disagreement diagnosis.

---

## 5. Revised novelty claim

The strongest defensible claim after the new search is:

> **Planning as ACID-compliant spatiotemporal composition.** A plan is not a text artifact; it is a transaction over a composable context. Plan versions, validators, judges, and subagents are fibers. Exploration is speculative and reversible. A plan is released only when it is a global section of the validation sheaf, and execution is rewindable across both context and environment.

Compared with the newest work:

- **AgentRewind** provides checkpoint/rewind but not causal plan validation or transactional release.
- **ACID-Agent** provides transactional semantics but not STRIPS planning/search or a formal composability calculus.
- **STAIR** provides graph-as-state and stage routing but not reversible execution.
- **Capability Sheaves** provides disagreement diagnosis but not planning or rollback.

The combination is currently missing from the literature and is testable.

---

## 6. Proposed evaluation protocol

### 6.1 Benchmarks

1. **PlanBench / PlanBench-XL** — causal validity, action legality, goal reachability.
2. **MettleBench** (`2608.14380`) — 82 long-horizon engineering tasks, task success + checklist prefix progress.
3. **Terminal-Bench 2.0** — external environment recovery boundary.
4. **CausalPlan-Ours** — generated Blocksworld/Logistics/Tyreworld + software plans with injected unsatisfied preconditions, clobber threats, type mismatches, drift, and parallel artifact conflicts.
5. **MemoryArena** — multi-session memory evaluation for RoT/session memory.

### 6.2 Baselines

- Zero-shot / ReAct
- Self-refine only
- PlanBench-style LLM-Modulo verifier loop
- LATS / Mind Evolution
- AgentRewind (checkpoint-rewind without our validator)
- ACID-Agent-style transaction wrapper without our planner
- Our full system with components ablated

### 6.3 Metrics

- task success rate
- ordered checklist progress (MettleBench)
- causal flaw rate before/after search
- verify/simulate/release pass rate
- rollback drift events
- leaked side effects after failed branches
- token cost and wall-clock time
- context budget violation rate

---

## 7. Proposed paper structure (augmented)

1. Introduction — planning artifacts, not text
2. Related work
   - 2024–2026 planning corpus (existing digest)
   - Agent reliability: AgentRewind, ACID-Agent
   - Self-evolution: ToE, OEO, skills
   - Context/memory: HIPIF, Not Worth Another Token, RippleMem, MemoryLake
   - Composition: Cordis, Capability Sheaves
3. Formal model
   - PlanAST and causal validator
   - Planning transactions and ACID semantics
   - Cordis context, fibers, provider stack
   - Aligned checkpoint/rewind
4. System
   - `/plan` orchestration protocol
   - AST search with execution feedback
   - RoT experience tree
   - release gate as semantic consistency check
5. Experiments
   - RQ1: Does causal validation improve plan legality beyond self-refine?
   - RQ2: Does rewind improve long-horizon checklist progress?
   - RQ3: Do transactional semantics reduce leaked side effects?
   - RQ4: Which components account for token/latency cost?
6. Ablations and failure taxonomy
7. Limitations and negative results
8. Conclusion

---

## 8. Immediate implementation priorities from the new papers

Status (2026-08-17): items 1, 2, 3, and 6 are implemented in engine v0.15.0. Execution-contract anti-stub verification (probe + symbol audit) is also implemented.

1. **Checkpoint records in search** — ✅ `plan.checkpoint()` / `plan.rewind()`; `search(..., checkpoint_before=True)`.
2. **Commit semantics** — ✅ `session["committed_*"]` separate from `best_*`; `plan.committed()`; only `release()` commits.
3. **RoT experience tree** — ✅ `RoTRule.perspective`, `record_outcome()`, `tree_report()`.
4. **Execution-feedback search mode** — 🟡 partial: execution contracts + feasibility probe implemented; full execution-guided AST search still pending.
5. **Stage-aware context budget** — ⏳ not yet implemented.
6. **Drift recovery node** — ✅ `ReplanningLadder` Tier 4 for drift/silent-failure signals.

---

## 9. Risks / honest caveats

- AgentRewind and ACID-Agent are concurrent work; a reviewer may see our contribution as incremental unless the formal planning-transaction model and causal validator show measurable gains.
- Public benchmark results are still required; the local benchmark is not sufficient.
- Capability-sheaf and ACID formalisms are heavy; citing them without implementing the relevant property may invite rejection.
- The strongest empirical claim will likely be on **rollback + planning verification**, not on base task success.
