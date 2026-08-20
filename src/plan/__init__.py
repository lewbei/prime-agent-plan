"""plan - plan mode skill entrypoint (global prime-agents install).

Re-exports the plan_mode engine and provides the run() convention:
    await plan.run("objective", "draft plan")   # start + assess round 1
or drive the loop round by round with start/assess/status/best.

The engine is resolved in order: (1) this skill's bundled engine, so plan
mode works in any project; (2) a repo-local plan_mode if present.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # .../skills/plan/src/plan
_SKILL_SRC = _HERE.parent  # .../skills/plan/src
if str(_SKILL_SRC) not in sys.path:
    sys.path.insert(0, str(_SKILL_SRC))

# prefer a repo-local engine if one is importable from cwd (dev override)
import importlib.util as _ilu
_repo = Path.cwd()
while _repo != _repo.parent:
    if (_repo / "plan_mode" / "__init__.py").exists():
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        break
    _repo = _repo.parent

import plan_mode
from plan_mode import (  # noqa: E402
    __version__,
    start,
    assess,
    assess_candidates,
    run,
    status,
    history,
    best,
    committed,
    checkpoint,
    rewind,
    finish,
    log_progress,
    suggest,
    list_sessions,
    rubric,
    verify,
    judge,
    record_judge,
    release,
    plan_dag,
    simulate,
    plan_quality,
    edit_file,
    rollback,
    deps_check,
    ground_check,
    constraint_check,
    fold_history,
    judge_ensemble,
    template,
    selfcheck,
    search_expand,
    search_select,
    search_backtrack,
    search_report,
    search,
    Context,
    Fiber,
    LifecycleState,
    TwistedMonoid,
    get_root_context,
    reset_root_context,
    create_subagent_context,
    provide_tool,
    execute_plan,
    execute_plan_sync,
    speculative_rollout,
    speculative_rollout_async,
    session_lock,
    ExecutionContract,
    parse_execution_contract,
    validate_execution_contract,
    parse_exit_criteria,
    validate_exit_criteria,
    run_exit_criteria,
    probe_contract,
    symbol_audit,
    scan_symbols,
    parity_audit,
    run_verification_commands,
    ExecutionEvidence,
    TaskExecution,
    CommandResult,
    parse_execution_evidence,
    AgentIsolation,
    OperationIsolation,
    ArtifactVersion,
    ConflictReport,
    IsolationManager,
    acquire_artifact,
    release_artifact,
    detect_conflicts,
    DriftEvidence,
    RecoveryDecision,
    RecoveryGraph,
    classify_drift,
    recovery_decision,
    PredicateSignature,
    validate_typed_atom,
    feedback_penalty,
    extract_declared_obligations,
    align_task_evidence,
    verify_execution_trace,
    verify_negative_constraints,
    RoTRuleBase,
    RoTRule,
    ReplanningLadder,
    ContextBudgeter,
    mutate_flaw_directed,
    mutate_exploratory,
    crossover_ast,
    ast_distance,
    PopulationMember,
    ASTSearchEngine,
    Proposition,
    PlanParser,
    PlanAST,
    CausalValidator,
    CausalLink,
    CausalFlaw,
    ActionSchema,
    ActionIR,
    FactTruth,
    ProjectedTruth,
    HardConstraint,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WitnessabilityStatus,
    WorldFact,
    render_markdown_view,
    CapabilityEntry,
    CapabilityRegistry,
    CompensationAction,
    ObservationVerifier,
    EpistemicCausalValidator,
    PlanValidationResult,
    ValidationStatus,
    merge_fact_truth,
    AuthorizationCertificate,
    PlanningSession,
    PlanVersion,
    SessionState,
    DiagnosticProbe,
    VOIProbingEngine,
    RecoveryStatus,
    SagaRecoveryManager,
    SagaRecoveryReport,
    BlindJudge,
    DualJudgeComparison,
    DualJudgeEvaluator,
    GroundedEpistemicJudge,
    JudgeVerdict,
    JudgeAdapter,
    OpenAIJudge,
    AnthropicJudge,
    GeminiJudge,
    DeepSeekJudge,
    EnsembleJudge,
    EpistemicPlanSearch,
    SearchResult,
    causal_crossover,
    insert_disambiguation_action,
    mutate_action_parameters,
    DEFAULT_PLANS_DIR,
    RUBRIC_PATH,
    REPO_ROOT,
    DEFAULT_MAX_ROUNDS,
    MAX_PLATEAU_ROUNDS,
    MIN_DELTA_TO_CONTINUE,
    JOURNAL_PATH,
    EphemeralWorkspace,
    ExecutionSandbox,
    IsolationPolicy,
    TransactionalExecutionManager,
    TransactionOutcome,
    ExecutionPlanManager,
    ExecutionBackend,
    DEFAULT_RUBRIC,
)
from plan_mode.self_verification import (  # noqa: E402
    DEFAULT_SELF_VERIFICATION_MODEL,
    DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    DEFAULT_SELF_VERIFICATION_PIVOTS,
    ProbabilisticSelfVerifier as _ProbabilisticSelfVerifier,
)


# Preserve the PR #1 deterministic selector as the fail-safe path. The public
# Prime entrypoint below adds inherited same-model verification automatically;
# callers continue to use plan.assess_candidates(...) without selecting a new
# mode or helper.
_deterministic_assess_candidates = assess_candidates


def assess_candidates(
    session,
    drafts,
    *,
    notes=None,
    plans_dir=None,
    verifier=None,
    generator_model=DEFAULT_SELF_VERIFICATION_MODEL,
    verifier_model=DEFAULT_SELF_VERIFICATION_MODEL,
    n_evaluations=DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    pivots=DEFAULT_SELF_VERIFICATION_PIVOTS,
):
    """Inherited Best-of-N selection: deterministic gate + Gemini self-verifier.

    The existing PR #1 deterministic checks remain the hard prefilter and
    fallback. When at least two eligible drafts exist and the probabilistic
    backend is available, Gemini 3.7 Flash ranks the candidates by default.
    The selected draft is then passed through the normal ``assess`` pipeline.

    Missing provider/backend credentials never disable deterministic Prime;
    selection falls back to the original PR #1 selector and reports the
    fallback in the returned metadata.
    """
    if not drafts:
        raise ValueError("drafts must be non-empty")
    if len(drafts) == 1:
        result = _deterministic_assess_candidates(
            session, drafts, notes=notes, plans_dir=plans_dir
        )
        result["selection_method"] = "deterministic-single-candidate"
        return result

    # Hard deterministic prefilter inherited from PR #1. Broken candidates do
    # not compete with clean candidates merely because the LLM prefers them.
    checked = []
    pass_indices = []
    for i, draft in enumerate(drafts):
        v = plan_mode.verify(draft)
        gc = plan_mode.ground_check(draft)
        sim = plan_mode.simulate(draft, initial_state=set(gc.get("verified", [])))
        hard_pass = bool(v.get("ok") and gc.get("ok") and sim.get("executable_plan"))
        checked.append({
            "candidate": i,
            "verify_ok": bool(v.get("ok")),
            "feasibility_ok": bool(gc.get("ok")),
            "sim_ok": bool(sim.get("executable_plan")),
            "hard_pass": hard_pass,
        })
        if hard_pass:
            pass_indices.append(i)

    # If no draft is fully clean, all drafts remain eligible for selecting the
    # best rework target. This does not certify them; assess/release gates still
    # apply afterward.
    eligible = pass_indices if pass_indices else list(range(len(drafts)))
    if len(eligible) == 1:
        chosen = eligible[0]
        result = plan_mode.assess(
            session,
            drafts[chosen],
            note=(notes or [None] * len(drafts))[chosen],
            plans_dir=plans_dir,
        )
        result.update({
            "selection_method": "deterministic-prefilter-single",
            "selected_candidate": chosen,
            "candidate_checks": checked,
            "candidates_scored": len(drafts),
        })
        return result

    if isinstance(session, dict):
        objective = str(session.get("objective") or "Select the best candidate plan")
    else:
        try:
            objective = str(plan_mode.status(session, plans_dir=plans_dir).get("objective") or "Select the best candidate plan")
        except Exception:
            objective = "Select the best candidate plan"

    soft_verifier = verifier or _ProbabilisticSelfVerifier()
    try:
        soft = soft_verifier.select(
            problem=objective,
            candidates=[drafts[i] for i in eligible],
            model=verifier_model or DEFAULT_SELF_VERIFICATION_MODEL,
            n_evaluations=n_evaluations,
            pivots=pivots,
        )
        chosen = eligible[soft.selected_index]
        result = plan_mode.assess(
            session,
            drafts[chosen],
            note=(notes or [None] * len(drafts))[chosen],
            plans_dir=plans_dir,
        )
        result.update({
            "selection_method": "inherited-same-model-self-verification",
            "selected_candidate": chosen,
            "generator_model": generator_model or DEFAULT_SELF_VERIFICATION_MODEL,
            "verifier_model": verifier_model or DEFAULT_SELF_VERIFICATION_MODEL,
            "is_self_verification": (
                (generator_model or DEFAULT_SELF_VERIFICATION_MODEL)
                == (verifier_model or DEFAULT_SELF_VERIFICATION_MODEL)
            ),
            "n_evaluations": n_evaluations,
            "pivots": min(pivots, len(eligible)),
            "eligible_candidates": eligible,
            "candidate_checks": checked,
            "probabilistic_scores": soft.scores,
            "probabilistic_ranking": [eligible[i] for i in soft.ranking],
            "candidates_scored": len(drafts),
        })
        return result
    except Exception as exc:
        # Provider/API/logprob availability is a soft dependency. A verifier
        # outage must not disable the deterministic PR #1 runtime.
        result = _deterministic_assess_candidates(
            session, drafts, notes=notes, plans_dir=plans_dir
        )
        result.update({
            "selection_method": "deterministic-fallback",
            "self_verification_available": False,
            "self_verification_error": f"{type(exc).__name__}: {exc}",
            "candidate_checks": checked,
        })
        return result


# Make the inherited behavior visible through both public imports once the
# global `plan` entrypoint is loaded. This preserves all PR #1 functionality
# while removing any need to select a separate verification mode.
plan_mode.assess_candidates = assess_candidates

__all__ = list(plan_mode.__all__)
