---
name: plan
description: "Plan mode - iteratively improves a plan for an objective until it converges. Draft, self-critique against a rubric, revise, and score each round; history is persisted under plans/. Use when the user invokes /plan or asks to plan something robustly, refine a plan, or enter plan mode."
---

# Plan Mode (/plan)

Plan mode drafts a plan for an objective and keeps improving it round by
round until the score stops increasing. Every version, score, and critique is
persisted, so "better and better" is an auditable fact, not a claim.

## Protocol (do this every time /plan is invoked)

The parent agent ORCHESTRATES; subagents do the thinking. The Python engine
only scores (rubric + verify + simulation), selects (UCB), and gates
(release). This matches the literature: generator/critic separation
(2505.01479), plan ensembles from multiple proposers (2601.17942, SYMPHONY
2601.22623), and an independent judge (2510.03469).

1. **Start.** Call `plan.start(objective)` (imported as `plan`). If a session
   for the objective exists under `plans/`, resume it.
2. **Spawn a planner subagent** (`await rlm('sub-task', name='planner')`) with
   the objective; it writes a draft plan to `<session_dir>/draft.md` and
   replies "drafted". The parent runs `plan.assess(s, draft)`.
3. **Assess.** Call `plan_mode.assess(session, draft)`. This scores the plan
   against the rubric (see `plan_mode/RUBRIC.md`) and returns structured
   critiques with ids like `success:Give numeric...`.
4. **Revise** the plan to address every remaining critique. Each round must
   beat the previous score by >= 1 point.
5. **Search plan space with subagents.** For each search iteration, spawn
   `width` reviser subagents (parallel), each told: the current plan, the
   critique list, "produce a revised COMPLETE plan addressing these critiques,
   write it to <session_dir>/revision-<i>-<iter>.md". The parent scores all
   with `plan.assess_candidates(s, drafts)` (best-of-N, 2601.17942) and
   continues the loop from the best candidate. This is the subagent form of
   MCTS expansion; the engine's `plan.search()` remains available for
   rule-based/API expansion when subagents are unnecessary.
6. **Repeat** assess -> revise until `continue == False` (status `converged`,
   after 2 non-improving rounds or `max_rounds`). `assess()` now runs the
   plan through the STRIPS-style simulator automatically: a task whose
   dependencies or input artifacts are unsatisfied produces a `mech:sim:*`
   critique and blocks convergence — the plan must be *executable*, not just
   well-worded. For choice points, draft 2-3 variants and use
   `assess_candidates(session, drafts)` (best-of-N).
7. **Release gate (loop before release).** Call `plan_mode.release(session)`.
   It fails until the plan has: converged, score >= 90, clean verify(),
   end-to-end simulation, and a judge verdict of "go" with falsifiable
   criteria. Keep looping assess -> revise -> re-judge until the gate passes;
   only then is the plan released. `finish()` enforces the same gate by
   default.

8. **Verify structure.** Call `plan_mode.verify(best_plan_text)`. After any
   change to the engine or rubric, run `plan_mode.selfcheck()` — re-evaluation
   is mandatory, never optional; only a green selfcheck ships. This checks
   whether the plan would actually execute: dependency graph (no missing
   targets, no forward refs, no cycles), concrete output artifact per task,
   milestone/task consistency, time-vs-deadline arithmetic. `verify` errors
   are injected into `assess()` as `mech:verify:*` critiques, so a plan with
   a broken task graph cannot converge at 100.
9. **Judge feasibility with a judge subagent.** Spawn a judge subagent
   (`await rlm('sub-task', name='plan-judge')`) told to review the plan
   adversarially (contradictions, missing inputs, unstated assumptions,
   unfalsifiable criteria) and reply with JSON {"verdict": "go"|"rework"|
   "reject", "feasibility_0_100", "blockers": [], "contradictions": [],
   "missing": [], "falsifiable_criteria": bool}. Persist it with
   `plan.record_judge(s, verdict)`. If verdict != "go", feed the blockers
   back as critiques and revise. The API judge (`await plan.judge(plan_text,
   objective)`) remains as a fallback when no subagent is needed. The judge
   is non-circular: a different agent than the planner, seeing only the plan
   text.
10. **Report** `plan_mode.status(session)` + judge verdict + verify result.
11. **Persist the learning.** If a critique pattern repeats, add it to
   `plan_mode/RUBRIC.md` (weights/checks JSON block) so future planning is
   stricter. Optionally record the session with `rlm.harness.record_refinement`.

## Engine API (v0.13.0)

- `plan_mode.start(objective, plans_dir=None, max_rounds=8)` -> session dict
- `plan_mode.assess(session_or_id, plan_text, note=None, addressed=None)` -> {version, score, delta, critiques, status, continue}
  - `addressed=[...]` = critique ids this revision claims to fix; unaddressed ids are re-emitted next round
  - mechanical critiques (mech:*: task numbering, dependency refs, past deadlines, duplicates, near-identical revisions) block convergence until resolved
- `plan_mode.execute_plan(plan_text, task_handlers=...)` -> transactional plan execution engine with Cordis Revertible Fibers (Theorem 61): executes tasks in child contexts; automatically rolls back all intermediate mutations in LIFO order on failure (Corollary 62).
- `plan_mode.speculative_rollout(plan_text, eval_fn)` -> isolated speculative MCTS rollouts; evaluates candidate execution in temporary fiber realm with guaranteed 100% clean state recovery.
- `plan_mode.create_subagent_context(name)` -> derives an isolated child context (Gamma_infinity) with a private scratch realm for subagent sandboxing (no parent pollution).
- `plan_mode.provide_tool(key, value)` -> registers an ephemeral tool/verifier fiber into the root harness context with automatic lifecycle disposal.
- `plan_mode.Context` / `Fiber` / `LifecycleState` -> foundational Cordis Spatiotemporal Composability primitives (recursive context, 10-rule lifecycle calculus, reactive coeffects).
- `plan_mode.log_progress(session, task, status, evidence=...)` records execution; a failed/blocked step arms a replan trigger -> next assess demands the smallest-scope repair
- `plan_mode.suggest(session)` emits rubric self-evolution suggestions (too-easy/too-strict checks) and writes `<session>.suggestions.md`
- `plan_mode.verify(plan_text)` -> {"ok", "errors", "warnings", "graph", "tasks"} deterministic structural audit (dependency graph, artifacts, milestones, time arithmetic)
- `plan_mode.plan_dag(plan_text)` -> explicit task graph {nodes, edges, artifacts, inputs} extracted from the plan
- `plan_mode.edit_file(path, content)` -> revertible effect (Cordis): applies a file mutation and journals its inverse automatically; `plan_mode.rollback(n)` applies the n most recent inverses in reverse order (twisted composition). No more hand-rolled backups.
- `plan_mode.deps_check()` -> reactive coeffects (Cordis notify): classifies every feature dependency (llm_expansion, api_judge, pytest, corpus, pdftotext) as satisfied/unsatisfied; search() and judge degrade gracefully per the spec instead of hard-coded key checks.
- `plan_mode.template()` -> the template bank: function inventory + sample plan skeleton + section templates (every planner gets concrete names, not prose)
- `plan_mode.selfcheck(plans_dir=None, run_pytest=True)` -> MANDATORY re-evaluation after ANY engine/rubric change: rubric compiles + corpus IDs verified + every finished session's best plan re-verifies/re-simulates/re-ground-checks + pytest green. Historical sessions (pre-0.11) are reported, not blocking.
- `plan_mode.ground_check(plan_text, cwd=None)` -> grounded feasibility: every declared environment input must resolve to an existing file (zero tolerance); internal handoffs are exempt; returns {ok, missing, verified, internal}
- `plan_mode.fold_history(session)` -> fold superseded round texts into one-line summaries at convergence, keeping the best round and the last two in full (HIPIF 2606.10507); legacy sessions (pre-v0.6.0) are never mutated
- `await plan_mode.judge_ensemble(session, plan_text, objective, n=3)` -> ensemble judgment: collects the API judge plus falsifiable recorded votes and a mechanical verify+simulate baseline, records the lower-median-feasibility verdict (2510.03469, 2601.17942)
- `plan_mode.simulate(plan_text, initial_state=...)` -> STRIPS-style execution against a state model (SymPlanner 2505.01479 / PyPDDLEngine 2603.06064): walks tasks in order, blocks tasks whose deps/inputs are unsatisfied, returns {executable_plan, trace, problems, dead_artifacts, final_state}. Simulation blockers are injected into assess() as mech:sim:* critiques.
- `plan_mode.plan_quality(plan_text, objective, initial_state=...)` -> combined verdict: structural + simulation + coverage closure (2607.12986), returned as executable / structurally-broken / simulation-blocked / incomplete-criteria
- `plan_mode.assess_candidates(session, drafts, notes=None)` -> best-of-N plan selection (2601.17942 ensembles): scores all candidates, records each, keeps the best as this round's version
- `await plan_mode.search(session, iterations=4, width=2, mode="mcts", expansion="llm")` -> FULL PLAN-SPACE SEARCH (v0.7.0: adaptive search effort — width escalates on 2-expansion plateaus, LFS 2506.05213 — plus evaluator-driven plan recombination between close-value siblings, 2509.22613 / Mind Evolution 2501.09891): MCTS with UCB1 selection + backprop (LATS 2310.04406, SYMPHONY 2601.22623), LLM-proposer expansion (DeepSeek generates critique-addressing revisions) with deterministic rule-mutation fallback, rollout = rubric + verify + simulation (SymPlanner 2505.01479), transposition table (GATS 2607.08894), cost tracking (2505.14656), confidence-gated pruning (2602.08948). mode="beam" runs level-wise best-of-N (2601.17942). Returns best plan + tree report.
- `plan_mode.search_expand(session, drafts, parent_node=None)` -> expand the plan search tree with scored candidate versions (SYMPHONY 2601.22623 multi-candidate expansion; rollout = rubric + verify + simulation; backprop = LATS 2310.04406)
- `plan_mode.search_select(session, exploration=1.4)` -> UCB1 selection of the next node to expand (SYMPHONY 2601.22623, cost penalty from 2505.14656)
- `plan_mode.search_backtrack(session, node_id)` -> abandon a plateaued branch and re-expand from an ancestor (LATS 2310.04406)
- `plan_mode.search_report(session)` -> tree audit: nodes, depth, leaves, best node (GATS 2607.08894)
- `plan_mode.release(session, min_score=90, require_judge=True)` -> RELEASE GATE (2602.08948 confidence-gated checkpoints, 2608.10729 acceptance thresholds): a plan is only releasable when ALL hold: converged + best score >= min_score + verify() clean + simulate() executes end-to-end + external judge returned "go" with falsifiable criteria. Until then the loop keeps going.
- `plan_mode.finish(s, require_release=True, min_score=90)` -> raises RuntimeError if the release gate has not passed, so a weak plan cannot be shipped early (loop before release)
- `await plan_mode.judge(plan_text, objective)` -> external LLM feasibility verdict {"verdict", "feasibility_0_100", "blockers", "contradictions", "missing", "falsifiable_criteria"}; returns ok=False with "error" if no API key/network
- `plan_mode.record_judge(session, verdict)` persists the judge verdict into `session["judge_log"]`
- `plan_mode.status(s)` / `plan_mode.history(s)` / `plan_mode.best(s)`
- `plan_mode.finish(s)` marks the session complete
- `plan_mode.list_sessions()` lists all plans
- `plan_mode.rubric()` shows the current rubric

Plans are stored as JSON in `<repo>/plans/<session_id>.json`.

## Long-running goal

If the user wants planning to continue across sessions until a target score,
create a thread goal: `await goal.create("plan <objective> to score >= N", ...)`.
