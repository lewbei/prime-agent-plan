# Plan-Mode Rubric

The plan-mode engine scores every plan version against this rubric and turns
each miss into a structured critique. Scores and critiques are recorded in the
session file under `plans/`, so plan improvement is auditable round by round.

## Mechanical checks (hard gates, not scored)

Independent of the weighted sections, every version is checked mechanically:
contiguous task numbering, dependency references pointing at existing tasks,
deadline dates that parse and lie in the future, duplicate task lines, and a
near-identical revision guard (similarity > 0.97). A `mech:*` critique blocks
convergence until the plan is objectively fixed. Failed execution steps
(`log_progress`) arm a replan trigger that demands the smallest-scope repair.

## How the loop works

1. `/plan <objective>` starts (or resumes) a plan session for the objective.
2. The agent drafts a plan and calls `plan_mode.assess(...)`.
3. Each miss becomes a critique (`section:hint`). The agent revises the plan to
   address every critique.
4. The loop continues while the score beats the best version by >= 1 point.
   Two non-improving rounds, or `max_rounds`, mark the plan "converged".
5. Resuming with `/plan <same objective>` keeps improving the stored plan.

## Sections, weights, and checks

| Section | Weight | Checks |
|---|---|---|
| Objective clarity | 10 | explicit goal statement; in-scope list; non-goals/out-of-scope |
| Measurable success criteria | 15 | numeric/verifiable acceptance criteria; deadline; pass/fail falsifiability |
| Assumptions and unknowns | 10 | assumptions list; unknowns/open questions |
| Task decomposition | 15 | >=3 ordered concrete tasks; dependencies; per-task outputs |
| Milestones | 10 | intermediate checkpoints; go/no-go decision gate |
| Risks and failure modes | 15 | risk list; mitigations/fallbacks; rollback path |
| Resource estimates | 10 | time estimates; cost/compute/token budget |
| Alternatives considered | 5 | >=1 alternative with rejection reason |
| Verification loop | 10 | verification step per milestone; revision/improvement loop |
| Immediate executability | 10 | named first action; no bare "explore" without a criterion |

Total weight: 152, normalized to a 0-100 score.

## Literature upgrades

After the arXiv 2024-2026 planning review, criteria grounded in planning
research (e.g., plan verification, subgoal decomposability, repair-on-failure,
replanning triggers) should be appended here as new sections. Edit the JSON
block below to tune weights or checks without touching code.

Round 2 (2026-08-13) added six checks inside existing sections from the
10-paper round-2 review (batch digests, round 2): replan budget
(2608.01428), event/threshold-driven replan triggers (2511.22354, 2508.09508),
step-wise simulator validation (2603.06064), world-model-probe-conditioned
refinement (2606.17924, 2606.04226), subgoal/dependency graph structure
(2511.20993), dual correction (2602.14551).

Full-corpus pass (2026-08-13) added 11 more checks from the 111-paper
reading (MASTER_PLANNING_DIGEST.md, batches 4-6): failure classification, fallback paths (risks);
dependency DAG, coverage closure (tasks); cost reconciliation (resources);
two-axis validity, adversarial checks (verification_machine); acceptance
thresholds (replan); evidence-traced revisions (memory); abstraction layers
(structure); refusal policy (executability).

Second full-corpus pass (2026-08-13, v4.1) added 9 more checks: primitive
libraries (tasks); distractor rejection (risks); evaluation function + dense
per-step feedback (verification); confidence-gated milestones (milestones);
silent-failure detection (grounding); representation invariance (structure);
multi-agent coordination (constraints); generality/reuse (memory). Also
tightened objective:explicit_goal (labelled goal line required) and activated
executability:no_vague (concrete step criteria). 70 checks, weight 152.

Rubric v8 (2026-08-13, shipped) added one more hardening check:
grounded_inputs — plans must declare the input files they consume from the
environment, each verified to exist (2402.11489). 98 checks, weight 155.

Rubric v6 (2026-08-13, shipped) added 4 more evidence-gated checks to the
low-weight hardening section: feedback_world_update (2606.22488),
opaque_domain_state (2602.03900), complexity_gated_reflection
(2507.14975), and decomposition_cost (2510.17922). 97 checks, weight 155.

Rubric v5 (2026-08-13, shipped) added 10 evidence-gated checks from the
next tier of unused relevance-3 candidates, placed in a new low-weight
"Next-tier hardening" section (weight 3) so pre-v5 plans drift <= 3 points
on re-scoring: hierarchy_validation (2511.18165), solver_first
(2512.00069), error_detection (2512.10342), parallel_opportunities
(2510.11608, 2506.02683), capability_alignment (2608.05999),
evidence_preservation (2606.22388), oscillation_guard (2401.03630),
user_preferences (2512.14138), verify_on_mismatch (2410.00079), and
rule_review_gate (2512.08536). 93 checks, weight 155.

Rubric v4 (2026-08-13, shipped) added 11 evidence-gated checks from the
301-paper digest (MASTER_PLANNING_DIGEST.md): deviation_handling and
preflight_risky (verification_machine); reuse_components, history_budget, and
rules_from_failures (memory); lookahead_backward and limited_commitment
(constraints); planner_upgrade and replan_timing (replan); process_based_eval
(verification); root_cause_isolation (escalation). Every cited paper ID in
the hints now exists in the downloaded corpus (txts/). 83 checks, weight 152.

```json
{
  "objective": {
    "label": "Objective clarity",
    "weight": 10,
    "items": [
      [
        "explicit_goal",
        "^(#+\\s*)?\\s*(goal|objective|aim|purpose)\\b",
        "State the objective explicitly in one opening line."
      ],
      [
        "scope_in",
        "(in\\s*scope|included|we will (do|build|deliver))",
        "List what is in scope."
      ],
      [
        "scope_out",
        "(out\\s*of\\s*scope|excluded|not\\s*(doing|building|in scope)|non-goals?)",
        "List what is explicitly out of scope (non-goals)."
      ]
    ]
  },
  "success": {
    "label": "Measurable success criteria",
    "weight": 15,
    "items": [
      [
        "numeric_criteria",
        "(\\d+(\\.\\d+)?\\s*(%|percent|ms|s|min|hours?|days?|weeks?|points?|items?|papers?|files?|tests?|epochs?|rounds?)|\\bpass\\b|\\bfail\\b)",
        "Give numeric/verifiable acceptance criteria."
      ],
      [
        "deadline",
        "(by\\s+\\d{4}-\\d{2}-\\d{2}|within\\s+\\d+\\s+(hour|day|week|month)s?|deadline)",
        "Give a time bound or deadline."
      ],
      [
        "falsifiable",
        "(pass/fail|pass\\s*/\\s*fail|verif\\w+ (by|with|via)|measur\\w+|acceptance (test|criteria)|reject\\b|falsif\\w+)",
        "Define a pass/fail (falsifiable) check per criterion."
      ]
    ]
  },
  "assumptions": {
    "label": "Assumptions and unknowns",
    "weight": 10,
    "items": [
      [
        "assumptions",
        "(^|\\n)\\s*#{0,6}\\s*assumptions?:?(\\s|$)",
        "List explicit assumptions."
      ],
      [
        "unknowns",
        "(unknowns?|open questions?|to\\s+be\\s+determined|TBD|risks?:)",
        "List unknowns / open questions."
      ]
    ]
  },
  "tasks": {
    "label": "Task decomposition and dependencies",
    "weight": 15,
    "items": [
      [
        "numbered_tasks",
        "(?m)^\\s*(\\d+[.)]|[-*])\\s+[A-Z][^\\n]{10,}$",
        "Provide at least 3 concrete, ordered tasks.",
        3
      ],
      [
        "dependencies",
        "(depends?\\s+on|after\\s+step|before\\s+step|blocked\\s+by|prerequisite|dependency)",
        "State dependencies between tasks."
      ],
      [
        "outputs",
        "(output|deliverable|artifact|produces?)",
        "Say what artifact each task produces."
      ],
      [
        "subgoal_graph",
        "(sub-?goal graph|dependency graph|subgoal (structure|decomposition|hierarch\\w+)|graph-?augment)",
        "Declare a subgoal/dependency graph structure (2403.17246)."
      ],
      [
        "dependency_dag",
        "(DAG|directed acyclic|predecessor|successor|(START|END)\\s*markers?|dependency graph|topological)",
        "Express step ordering as explicit predecessor/successor edges or a DAG."
      ],
      [
        "coverage_closure",
        "(coverage|discharg\\w+|omission|every (criterion|goal|requirement) (maps|is covered)|obligation)",
        "Ensure every success criterion/goal maps to at least one task (no silent omissions)."
      ],
      [
        "primitive_library",
        "(atomic (primitiv\\w+|action\\w+)|primitive library|orthogonal (actions?|operations?|skills?)|small set of (atomic|orthogonal))",
        "Compose steps from a named set of atomic/orthogonal primitives (2506.01475)."
      ]
    ]
  },
  "milestones": {
    "label": "Milestones and checkpoints",
    "weight": 10,
    "items": [
      [
        "milestones",
        "(milestone|checkpoint|gate|phase\\s+\\d)",
        "Define intermediate milestones or checkpoints."
      ],
      [
        "gonogo",
        "(go/no-go|go\\s*/\\s*no-go|decision (point|gate)|abort|stop condition)",
        "Include a go/no-go decision gate."
      ],
      [
        "confidence_gate",
        "((confidence|uncertainty).{0,60}(threshold|gate|signal)|measurable signal|decision rule (halt|revise|change|abort)|halt.?/?.?revise)",
        "Each milestone: measurable signal + decision rule (halt/revise/change) (2604.17821)."
      ]
    ]
  },
  "risks": {
    "label": "Risks and failure modes",
    "weight": 15,
    "items": [
      [
        "risk_list",
        "((^|\\n)\\s*#{0,6}\\s*risks?:?(\\s|$)|failure modes?:|could fail|might fail|worst case)",
        "List risks or failure modes."
      ],
      [
        "mitigations",
        "(mitigat\\w+|fallback|contingency|rollback|revert|backup plan)",
        "Give a mitigation/fallback per major risk."
      ],
      [
        "rollback",
        "(rollback|revert|undo|restore)",
        "Describe how to roll back or undo."
      ],
      [
        "failure_classification",
        "(failure (classes?|types?|levels?)|classif\\w+ (failure|risk)s?|semantic .{0,40}physical|recovery scope|local (repair|fix) .{0,40}global)",
        "Classify each failure mode by type and recovery scope (local fix vs global replan)."
      ],
      [
        "fallback_path",
        "(fallback (path|plan|alternative)|alternative (path|route)|contingenc\\w+|if \\w+ fails?,? (try|use|switch|fall back))",
        "Give a fallback/alternative path for failed steps, not just the happy path."
      ],
      [
        "distractor_rejection",
        "(distractors?|plausible-?but-?wrong|wrong (inputs?|assumptions?)|reject\\w* \\w+ (inputs?|assumptions?))",
        "Name likely distractors (plausible-but-wrong assumptions) and how the plan rejects them (2601.11908)."
      ]
    ]
  },
  "resources": {
    "label": "Resource and cost estimates",
    "weight": 10,
    "items": [
      [
        "time_estimate",
        "(\\d+\\s*(minutes?|hours?|days?|weeks?|months?)\\s*(per|each|total|budget)|\\bestimated?\\b)",
        "Estimate time per task or total."
      ],
      [
        "budget",
        "(cost|budget|\\$|USD|tokens?|compute|GPU)",
        "Estimate cost/compute/token budget."
      ],
      [
        "cost_reconciliation",
        "(reconcile|reconciliation|cumulative (cost|time|budget)|totals? vs\\.?|budget (check|tracking)|cost (check|tracking))",
        "Reconcile cumulative cost/time estimates against the stated budget."
      ]
    ]
  },
  "alternatives": {
    "label": "Alternatives considered",
    "weight": 5,
    "items": [
      [
        "alternatives",
        "(alternative|instead|rather than|option [A-Z]|vs\\.? )",
        "Consider at least one alternative and say why it was rejected."
      ]
    ]
  },
  "verification": {
    "label": "Verification and self-improvement loop",
    "weight": 10,
    "items": [
      [
        "verify",
        "(verify|validate|test|audit|check that|review)",
        "Include a verification step for each milestone."
      ],
      [
        "feedback_loop",
        "(revis\\w+|iterate|improve|refine|feedback|next round|loop)",
        "Include a revision/improvement loop."
      ],
      [
        "evaluation_function",
        "(evaluation function|objective function|scoring (rule|function|criteri\\w+)|rank\\w* (intermediate|partial) (states?|plans?)|heuristic for (states?|steps?|actions?)|acceptance predicate)",
        "Define an explicit evaluation function to score intermediate states/steps (2501.02486)."
      ],
      [
        "dense_feedback",
        "(per-?step (feedback|verif\\w+|scor\\w+|check\\w+)|dense (feedback|reward|signal)|immediately? (score|check|verify) (each|every) step|step-?level (feedback|scor\\w+))",
        "Give per-step immediate quality feedback, not only an end-of-plan review (2512.23167)."
      ],
      [
        "process_based_eval",
        "(process-?based (evaluat\\w+|reward|feedback)|evaluate (reasoning|grounding|recovery)|score (the )?(reasoning|recovery|grounding) steps?)",
        "Evaluate the process (reasoning, grounding, recovery steps), not only the final outcome (2603.14248)."
      ]
    ]
  },
  "structure": {
    "label": "Explicit plan structure",
    "weight": 6,
    "items": [
      [
        "declared_structure",
        "(sections?|phases?|##|numbered|structure of this plan)",
        "Declare the plan's structure (sections/phases)."
      ],
      [
        "pseudocode_steps",
        "(?m)^\\s*(step\\s+\\d+|\\d+[.)])\\s*[A-Za-z]",
        "Use numbered pseudocode-style steps.",
        3
      ],
      [
        "abstraction_layers",
        "(abstraction (level|layer)s?|top-?level (plan|summary) .{0,40}(detail|sub-?plan)|layered plan|hierarch\\w+ (plan|layer|structure))",
        "Layer the plan: top-level named sub-plans plus detail blocks."
      ],
      [
        "representation_invariance",
        "(renam\\w+|representation-?invarian\\w*|invariant (to|under) (renam\\w+|format|wording))",
        "Plan should be robust to renaming/reformatting (representation invariance, 2409.13373)."
      ],
      [
        "detail_calibration",
        "(detail (level|calibrat\\w+)|level of detail|concise|concise(ly)? (describe|state)|no over-?thinking|proportional (detail|to (task )?complexity))",
        "Calibrate detail to task complexity; no overthinking on simple steps (2505.10543)."
      ]
    ]
  },
  "constraints": {
    "label": "Step constraints (preconditions/effects)",
    "weight": 6,
    "items": [
      [
        "preconditions",
        "(precondition|requires?|inputs?|before starting|prerequisite)",
        "State preconditions/inputs per step."
      ],
      [
        "effects",
        "(expected (output|result|effect)|outputs?|deliverables?|produces?|postcondition)",
        "State expected outputs/effects per step."
      ],
      [
        "per_step_constraints",
        "(applies? at (each|every|the) step|re-?anchor|constraint(s)? (per|at (each|every)) step|step-level (constraint|check))",
        "Re-anchor each constraint at the step where it applies."
      ],
      [
        "dual_correction",
        "(dual-?correction|semantic\\w* .{0,30}physical\\w*|physical (feasib\\w+|infeasib\\w+)|logical (consistenc\\w+|inconsistenc\\w+))",
        "Address dual correction: logical consistency AND physical feasibility (2502.12435)."
      ],
      [
        "multiagent_coordination",
        "(multi-?agent|inter-?agent|coordination (constraints?|protocols?|requirements?)|who (communicates?|coordinates?)|perspective-?taking|agent (roles?|handoff))",
        "State inter-agent/role coordination constraints explicitly (2408.17379, 2411.17636)."
      ],
      [
        "lookahead_backward",
        "(lookahead|look-?ahead|backward (reasoning|value|propagation)|regress(ion)? plan|bidirectional|forward-?backward)",
        "Verify each step against future steps (lookahead) and propagate value backward from the goal (2601.22311, 2411.01790)."
      ],
      [
        "limited_commitment",
        "(commit(s)? (only|at most) (\\d+|a few) (next )?steps?|limited (commitment|horizon)|re-?evaluat\\w+ (after|every) (each )?\\d+ steps?)",
        "Limit commitment to a small horizon; re-decide after every few steps (2601.22311)."
      ]
    ]
  },
  "verification_machine": {
    "label": "Machine-checkable verification",
    "weight": 6,
    "items": [
      [
        "invariant",
        "(invariant|checkable|solver|validator|automated check|script|test that)",
        "Include at least one mechanically checkable invariant."
      ],
      [
        "how_checked",
        "(verify (by|with|via|using)|checked by|validated by|assert|hash|checksum)",
        "Say how the invariant is checked (tool/script/assert)."
      ],
      [
        "external_checker",
        "(solver|validator|script|tool|external (check|verif)|non-LLM|deterministic (check|verif)|assert)",
        "Name a non-LLM checker (tool/script/solver) for the plan."
      ],
      [
        "simulator_check",
        "(simulat\\w+|step-?wise (valid\\w+|check\\w+)|PDDL|symbolic (simulat\\w+|valid\\w+))",
        "Validate each step with an external simulator/symbolic checker (2505.01479)."
      ],
      [
        "two_axes",
        "(local(ly)? (step )?valid\\w* .{0,60}global|global (reachab\\w*|goal) .{0,60}local|two (axes|dimensions)|separat\\w+ (scores?|axes?|checks?))",
        "Check local step validity and global goal reachability separately (two axes)."
      ],
      [
        "adversarial_check",
        "(adversar\\w+|stress-?test|falsif\\w+|try to (break|falsify)|edge cases?)",
        "Include an adversarial check that tries to falsify success (edge cases)."
      ],
      [
        "deviation_handling",
        "(corrective action|on (failure|deviation)|if (the )?check fails|when verification fails)",
        "Say what corrective action follows a failed verification (2512.09629)."
      ],
      [
        "preflight_risky",
        "(irreversible|risky steps?|pre-?execution (check|critique|review)|preflight)",
        "Pre-execution critique pass over risky/irreversible steps (2604.19558)."
      ]
    ]
  },
  "replan": {
    "label": "Replanning policy",
    "weight": 6,
    "items": [
      [
        "triggers",
        "(replan|revise the plan|if .* (fail|change|wrong)|trigger)",
        "Define replan triggers (what failure/observation causes revision)."
      ],
      [
        "scoped_repair",
        "(smallest|local (fix|repair|patch)|patch (the|only|that)|reuse (the|prior|existing) plan|prefix)",
        "Prefer smallest-scope repair and reuse of the valid plan prefix."
      ],
      [
        "preemptive_failure_enumeration",
        "(likely (failure|pitfall|risk)s? (before|up ?front|in advance)|enumerate (failure|pitfall)|negative constraints?|proactive (pitfall|failure) avoidance)",
        "Enumerate likely failure modes as negative constraints before drafting."
      ],
      [
        "replan_budget",
        "(replan\\w* budget|budget\\w* replan|cap (the |on )?(replan|revision)\\w*|max\\w* replan\\w* rounds?)",
        "Budget or cap replanning/revision rounds (BRACE 2608.01428)."
      ],
      [
        "event_driven",
        "(event-?driven|threshold|risk crosses|when \\w+ (crosses|exceeds)|named events?)",
        "Replan triggers must be event- or threshold-driven (2511.22354, 2508.09508)."
      ],
      [
        "acceptance_threshold",
        "(accept(ance)? (threshold|criteri\\w+)|stop when|halting (condition|rule)|convergence (criterion|condition)|termination (condition|rule))",
        "State the acceptance threshold / stopping condition for the refinement loop."
      ],
      [
        "planner_upgrade",
        "(upgrade|replace|retrain|swap) (the )?(planner|strategy|model)|improve the planner itself|SERP|strategy (upgrade|replacement)",
        "After repeated failure, upgrade the planner/strategy itself, not just the plan (2603.02772)."
      ],
      [
        "replan_timing",
        "(replan (immediately )?(before|prior to) (time-?sensitive|critical|deadline) (steps?|phases?)|timing-?sensitive|fresh replan)",
        "Schedule a fresh replan immediately before timing-sensitive steps (2608.03483)."
      ]
    ]
  },
  "grounding": {
    "label": "Step grounding and progress detection",
    "weight": 6,
    "items": [
      [
        "step_outcome",
        "(how (to|we) know|detect|success (of|for) (a |each |the )?step|step (succeeds|fails|works))",
        "Say how each step's success/failure is detected."
      ],
      [
        "subgoal_checkpoints",
        "(sub-?goal|intermediate (goal|checkpoint)|checkpoint after|incremental progress)",
        "Include intermediate sub-goal checkpoints for long plans."
      ],
      [
        "state_restatement",
        "(restat(e|ing) (the )?(state|world|goal)|memoiz\\w+ state|state (after|at) (each|every) step|update (the )?state (after|per|each))",
        "Restate the world/plan state after each step to prevent goal drift."
      ],
      [
        "world_probe",
        "(world model|state model|scene (twin|reconstruct\\w+)|world (probe|query)|future state)",
        "Condition refinement rounds on a world/state model probe (2402.11489)."
      ],
      [
        "silent_failure",
        "(silent(ly)? fail\\w*|fails? silently|no output|empty (result|response)|implicit failure|tool\\w* (return|fail).{0,30}(nothing|empty|silently))",
        "Define failure detection for steps that fail silently (2606.22388)."
      ],
      [
        "incremental_presentation",
        "(incremental|stream(ed|ing)? (updates?|presentation)|inspectable (steps?|sequence)|interruptible|one step at a time|per-?step (presentation|output|delivery))",
        "Present the plan as an inspectable, incremental step sequence (2510.08992)."
      ]
    ]
  },
  "memory": {
    "label": "History and revision strategy",
    "weight": 4,
    "items": [
      [
        "lessons",
        "(lessons? learned|past (failure|attempt|mistake)|previous (round|version)|history)",
        "Reference past failures/lessons or plan history."
      ],
      [
        "revision_strategy",
        "(revision (strategy|loop|policy)|improve (by|via|through)|next round|iterate)",
        "Name the revision strategy (how the next round will differ)."
      ],
      [
        "diverse_alternatives",
        "(best-?of-?N|candidate (plans?|alternatives?|options?)|sample (multiple|several|N) (plans?|alternatives?)|evaluate (multiple|candidate))",
        "Consider diverse candidate plans and pick the best via an evaluator (best-of-N)."
      ],
      [
        "evidence_traced",
        "(cit(e|es|ing|ed) .{0,40}(evidence|failure|violation)|traceab\\w+ (to|from) .{0,30}(failure|evidence)|evidence-?traced)",
        "Each revision must cite the specific evidence/failure it addresses."
      ],
      [
        "generality_reuse",
        "(reusab\\w+|transfer(able)? (to|across)|template-?able|generaliz\\w+ (to|across)|core (intent|structure) .{0,40}context|parameterized)",
        "Separate reusable core structure from task-specific detail (2605.06957, 2606.03951)."
      ],
      [
        "reuse_components",
        "(reus\\w+|prior (successful )?demonstration|existing (component|module|pattern))",
        "Reuse validated components or prior demonstrations (2605.06957)."
      ],
      [
        "history_budget",
        "(fold(ed)?|prun\\w+|summariz\\w+|compress\\w+|cap (the )?(history|context)|history budget|context budget)",
        "Budget, fold, or prune plan history to bound context growth (2606.10507)."
      ],
      [
        "rules_from_failures",
        "(rule(s)? (from|distill\\w+ from)|distill\\w+ (rule|lesson)|append\\w+ (to )?(rule|constraint) list|failure-?derived (rule|constraint))",
        "Distill explicit reusable rules from past failures and append them (RoT 2404.05449)."
      ]
    ]
  },
  "executability": {
    "label": "Immediate executability",
    "weight": 10,
    "items": [
      [
        "first_action",
        "(first|step\\s*1|start by|today|immediately|now)",
        "Name the very first action to take."
      ],
      [
        "no_vague",
        "(criterion|criteri\\w+|until|threshold|stop when|accept\\w+|measur\\w+ signal|exit (condition|criteria))",
        "Steps carry concrete criteria/stopping conditions (no bare 'explore')."
      ],
      [
        "refusal_policy",
        "(refus(e|al)|not worth (attempting|doing)|infeasib\\w+|when (the )?(goal|task) (is )?unachiev|declin\\w+ (the )?task)",
        "State when the plan should refuse/abort an infeasible goal."
      ]
    ]
  },
  "escalation": {
    "label": "Adaptive deliberation and escalation",
    "weight": 4,
    "items": [
      [
        "cheapest_first",
        "(cheapest|cheap|low-?cost).{0,40}(first|action|option)",
        "Attempt the cheapest viable action first."
      ],
      [
        "escalate_when_needed",
        "(escalat\\w+|only if (needed|necessary)|fall ?back to)",
        "Escalate to expensive/deliberate steps only when needed."
      ],
      [
        "root_cause_isolation",
        "(root-?cause (isolat\\w+|analys\\w+)|isolate (the )?root cause|root cause (of|behind))",
        "Isolate the root cause of failures instead of step-by-step symptom reflection (2509.25370)."
      ]
    ]
  },
  "uncertainty": {
    "label": "Uncertainty awareness",
    "weight": 4,
    "items": [
      [
        "uncertainty_estimate",
        "(uncertain\\w+|confidence (estimate|level|score)|risk of (failure|error))",
        "Estimate uncertainty per step."
      ],
      [
        "conservative_switch",
        "(conservative|switch (planning )?mode|fall ?back plan|when (uncertain|confidence is low))",
        "Switch to a conservative mode when uncertainty is high."
      ]
    ]
  },
  "hardening": {
    "label": "Next-tier hardening (v5)",
    "weight": 3,
    "items": [
      [
        "user_preferences",
        "(user preferences?|preference capture|elicit\\w+ (preferences?|constraints)|requirements (from|of) (the )?user|user-?specified (constraints?|preferences?))",
        "Capture user preferences/constraints before drafting (2512.14138)."
      ],
      [
        "parallel_opportunities",
        "(paralleliz\\w+|concurrent\\w+|independent tasks? (that|which|can) (run|execute)|(tasks?|steps?) [^.]{0,40}(run|execute) (in )?parallel|run (tasks?|steps?) (in )?parallel)",
        "Name which tasks parallelize; explicit parallelism/coordination (2510.11608, 2506.02683)."
      ],
      [
        "verify_on_mismatch",
        "(verif\\w+ (only )?on (mismatch|deviation)|check (only )?on (mismatch|deviation)|speculative steps?|verify only (when|if) (changed|mismatch))",
        "Verify on mismatch: speculative steps, cheap checks until deviation (2410.00079)."
      ],
      [
        "hierarchy_validation",
        "(hierarch\\w+ (plan|decompos\\w+).{0,60}(valid|parse|check)|validat\\w+ (the )?hierarch\\w+|syntactic (valid|check)\\w* .{0,40}hierarch\\w+)",
        "Validate hierarchical decomposition syntactically, not just prose (2511.18165)."
      ],
      [
        "capability_alignment",
        "(align\\w+ (subgoals?|plans?|steps?) (to|with) (executor|agent|tool)|match\\w+ (subgoals?|steps?) (to|with) (executor|agent)|subgoals? (aligned|matched) (to|with) (executor|agent|tool) capabilit\\w+|executor capabilit\\w+)",
        "Align subgoals with executor capabilities (2608.05999)."
      ],
      [
        "solver_first",
        "(solver-?first|flaw cache|plan cache|solver (checks|verif\\w+) .{0,40}(LLM|review|repair)|LLM (review|repair) .{0,40}solver)",
        "Solver-first design with LLM review/repair and flaw caches (2512.00069)."
      ],
      [
        "oscillation_guard",
        "(oscillat\\w+|revisit\\w+ (the )?same (state|step|failure)|repeat\\w+ (the )?same (failure|step)|avoid (repeating|revisiting) (the )?same)",
        "Guard against oscillation: never revisit an identical failed state (2401.03630)."
      ],
      [
        "error_detection",
        "(detect (the )?error (immediately|at (the|each) step|early)|error detection (at|per) step|stop at (the )?first (error|failure)|catch (the )?first (error|failure))",
        "Detect errors immediately at each step; corrections are near-random once errors compound (2512.10342)."
      ],
      [
        "evidence_preservation",
        "(preserv\\w+ (intermediate )?evidence|keep (intermediate )?evidence|evidence across steps|unreliable feedback|distrust\\w+ (feedback|outputs?))",
        "Preserve intermediate evidence across steps; recognize unreliable feedback (2606.22388)."
      ],
      [
        "rule_review_gate",
        "(review\\w* (generated )?rules?|human review|rule review (gate|step)|prioriti[sz]\\w+ rules?)",
        "Human/verifier review of generated rules before they bind (2512.08536)."
      ],
      [
        "decomposition_cost",
        "(decomposition (granularity|depth|cost)|decompos\\w+ .{0,40}(cost|difficulty) (trade-?off|dictates|matched)|match\\w+ decomposition (to|with) (task )?(difficulty|complexity|cost)|performance-?cost (dilemma|trade-?off))",
        "Match decomposition granularity to task difficulty and cost (2510.17922)."
      ],
      [
        "feedback_world_update",
        "(feedback (updat|mutat|revis)\\w+ (the )?(world|state) (model|state)|world-?model (updat\\w+|revis\\w+) from (execution )?feedback|loop closure|close the loop (between|of) (execution|plan).{0,40}(world|state|model))",
        "Execution feedback must update the world/state model, closing the loop (2606.22488)."
      ],
      [
        "opaque_domain_state",
        "(opaque (domain|environment|state)|hidden (state|rules?|transition)|task mental knowledge|explicit\\w+ (state|rules?) (for|of) (the )?(opaque|hidden|black-?box))",
        "State opaque domains' hidden rules/state explicitly before planning (2602.03900)."
      ],
      [
        "complexity_gated_reflection",
        "(complexity (assessment|class|level) (decides|chooses|sets|gates) (reflection|deliberation|reasoning|effort)|reflection intensity .{0,40}(task )?(complexity|difficulty)|match\\w+ (reflection|deliberation|reasoning) (effort|intensity) (to|with) (task )?(complexity|difficulty))",
        "Choose reflection/deliberation intensity from a task-complexity assessment (2507.14975)."
      ],
      [
        "grounded_inputs",
        "(requires?|inputs?|reads?|consumes?)\\s*[:(]\\s*[\\w./-]+\\.\\w{1,8}",
        "Declare the input files this plan consumes from the environment; each is verified to exist (2402.11489)."
      ]
    ]
  }
}
```

The JSON block above is optional. When it contains sections, it fully replaces
the default rubric. When empty (`{}`), the code default is used.
