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

_HERE = Path(__file__).resolve().parent
_SKILL_SRC = _HERE.parent
if str(_SKILL_SRC) not in sys.path:
    sys.path.insert(0, str(_SKILL_SRC))

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
    __version__, start, assess, assess_candidates, run, status, history, best,
    committed, checkpoint, rewind, finish, log_progress, suggest, list_sessions,
    rubric, verify, judge, record_judge, release, plan_dag, simulate, plan_quality,
    edit_file, rollback, deps_check, ground_check, constraint_check, fold_history,
    judge_ensemble, template, selfcheck, search_expand, search_select,
    search_backtrack, search_report, search, Context, Fiber, LifecycleState,
    TwistedMonoid, get_root_context, reset_root_context, create_subagent_context,
    provide_tool, execute_plan, execute_plan_sync, speculative_rollout,
    speculative_rollout_async, session_lock, ExecutionContract,
    parse_execution_contract, validate_execution_contract, parse_exit_criteria,
    validate_exit_criteria, run_exit_criteria, probe_contract, symbol_audit,
    scan_symbols, parity_audit, run_verification_commands, ExecutionEvidence,
    TaskExecution, CommandResult, parse_execution_evidence, AgentIsolation,
    OperationIsolation, ArtifactVersion, ConflictReport, IsolationManager,
    acquire_artifact, release_artifact, detect_conflicts, DriftEvidence,
    RecoveryDecision, RecoveryGraph, classify_drift, recovery_decision,
    PredicateSignature, validate_typed_atom, feedback_penalty,
    extract_declared_obligations, align_task_evidence, verify_execution_trace,
    verify_negative_constraints, RoTRuleBase, RoTRule, ReplanningLadder,
    ContextBudgeter, mutate_flaw_directed, mutate_exploratory, crossover_ast,
    ast_distance, PopulationMember, ASTSearchEngine, Proposition, PlanParser,
    PlanAST, CausalValidator, CausalLink, CausalFlaw, ActionSchema, ActionIR,
    FactTruth, ProjectedTruth, HardConstraint, PlanIR, PredicateCondition,
    Provenance, SourceType, SuccessCriterion, WitnessabilityStatus, WorldFact,
    render_markdown_view, CapabilityEntry, CapabilityRegistry,
    CompensationAction, ObservationVerifier, EpistemicCausalValidator,
    PlanValidationResult, ValidationStatus, merge_fact_truth,
    AuthorizationCertificate, PlanningSession, PlanVersion, SessionState,
    DiagnosticProbe, VOIProbingEngine, RecoveryStatus, SagaRecoveryManager,
    SagaRecoveryReport, BlindJudge, DualJudgeComparison, DualJudgeEvaluator,
    GroundedEpistemicJudge, JudgeVerdict, JudgeAdapter, OpenAIJudge,
    AnthropicJudge, GeminiJudge, DeepSeekJudge, EnsembleJudge,
    EpistemicPlanSearch, SearchResult, causal_crossover,
    insert_disambiguation_action, mutate_action_parameters, DEFAULT_PLANS_DIR,
    RUBRIC_PATH, REPO_ROOT, DEFAULT_MAX_ROUNDS, MAX_PLATEAU_ROUNDS,
    MIN_DELTA_TO_CONTINUE, JOURNAL_PATH, EphemeralWorkspace, ExecutionSandbox,
    IsolationPolicy, TransactionalExecutionManager, TransactionOutcome,
    ExecutionPlanManager, ExecutionBackend, DEFAULT_RUBRIC,
)
from plan_mode.self_verification import (  # noqa: E402
    DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    DEFAULT_SELF_VERIFICATION_PIVOTS,
    ProbabilisticSelfVerifier as _ProbabilisticSelfVerifier,
    resolve_implementation_model,
)

_deterministic_assess_candidates = assess_candidates


def _compat_ranking(checks, preferred_order=None):
    """Preserve the PR #1 ranking return contract for inherited selection."""
    by_index = {item["candidate"]: item for item in checks}
    preferred = list(preferred_order or [])
    remaining = [
        item["candidate"] for item in sorted(
            checks,
            key=lambda x: (
                -int(x["hard_pass"]),
                -int(x["sim_ok"]),
                -int(x["verify_ok"]),
                -int(x["feasibility_ok"]),
                x["candidate"],
            ),
        )
        if item["candidate"] not in preferred
    ]
    order = preferred + remaining
    return [
        {
            "candidate": i,
            "score": None,
            "effective_score": 100.0 if by_index[i]["hard_pass"] else 0.0,
            "sim_ok": by_index[i]["sim_ok"],
            "verify_ok": by_index[i]["verify_ok"],
            "feasibility_ok": by_index[i]["feasibility_ok"],
        }
        for i in order
    ]


def _session_state_for_model(session, plans_dir=None):
    if isinstance(session, dict):
        return session
    try:
        pdir = Path(plans_dir) if plans_dir else DEFAULT_PLANS_DIR
        return plan_mode._load_session(pdir, session)
    except Exception:
        return None


def assess_candidates(
    session,
    drafts,
    *,
    notes=None,
    plans_dir=None,
    verifier=None,
    implementation_model=None,
    n_evaluations=DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    pivots=DEFAULT_SELF_VERIFICATION_PIVOTS,
):
    """Inherited Best-of-N with model-agnostic same-model self-verification.

    The model is inherited from the active implementation runtime/session. If
    model M generated the candidates, model M verifies/ranks them. Prime never
    silently substitutes a hard-coded Gemini, DeepSeek, or other model.
    """
    if not drafts:
        raise ValueError("drafts must be non-empty")
    if len(drafts) == 1:
        result = _deterministic_assess_candidates(
            session, drafts, notes=notes, plans_dir=plans_dir
        )
        result["selection_method"] = "deterministic-single-candidate"
        return result

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
            "ranking": _compat_ranking(checked, [chosen]),
            "candidates_scored": len(drafts),
        })
        return result

    session_state = _session_state_for_model(session, plans_dir)
    active_model = resolve_implementation_model(
        implementation_model,
        session=session_state,
    )
    if not active_model:
        result = _deterministic_assess_candidates(
            session, drafts, notes=notes, plans_dir=plans_dir
        )
        result.update({
            "selection_method": "deterministic-fallback-no-model",
            "self_verification_available": False,
            "self_verification_error": (
                "Active implementation-model identity unavailable; no verifier model was substituted"
            ),
            "candidate_checks": checked,
        })
        return result

    if isinstance(session_state, dict):
        objective = str(session_state.get("objective") or "Select the best candidate plan")
    else:
        objective = "Select the best candidate plan"

    soft_verifier = verifier or _ProbabilisticSelfVerifier()
    try:
        soft = soft_verifier.select(
            problem=objective,
            candidates=[drafts[i] for i in eligible],
            model=active_model,
            n_evaluations=n_evaluations,
            pivots=pivots,
        )
        chosen = eligible[soft.selected_index]
        soft_order = [eligible[i] for i in soft.ranking]
        result = plan_mode.assess(
            session,
            drafts[chosen],
            note=(notes or [None] * len(drafts))[chosen],
            plans_dir=plans_dir,
        )
        result.update({
            "selection_method": "inherited-same-model-self-verification",
            "selected_candidate": chosen,
            "implementation_model": active_model,
            "generator_model": active_model,
            "verifier_model": active_model,
            "is_self_verification": True,
            "n_evaluations": n_evaluations,
            "pivots": min(pivots, len(eligible)),
            "eligible_candidates": eligible,
            "candidate_checks": checked,
            "probabilistic_scores": soft.scores,
            "probabilistic_ranking": soft_order,
            "ranking": _compat_ranking(checked, soft_order),
            "candidates_scored": len(drafts),
        })
        return result
    except Exception as exc:
        result = _deterministic_assess_candidates(
            session, drafts, notes=notes, plans_dir=plans_dir
        )
        result.update({
            "selection_method": "deterministic-fallback",
            "implementation_model": active_model,
            "self_verification_available": False,
            "self_verification_error": f"{type(exc).__name__}: {exc}",
            "candidate_checks": checked,
        })
        return result


plan_mode.assess_candidates = assess_candidates

__all__ = list(plan_mode.__all__)
