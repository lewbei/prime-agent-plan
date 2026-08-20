"""plan - hardened public Plan Mode entrypoint.

The bundled ``plan_mode`` engine remains the implementation core.  This module
adds user-facing fail-closed wrappers for convergence, release and legacy
Cordis execution, then installs those wrappers back onto ``plan_mode`` so the
normal ``import plan`` workflow cannot accidentally bypass them.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SKILL_SRC = _HERE.parent
if str(_SKILL_SRC) not in sys.path:
    sys.path.insert(0, str(_SKILL_SRC))

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
    DEFAULT_VERIFIER_MAX_CALLS,
    DEFAULT_VERIFIER_MAX_WORKERS,
    DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
    ProbabilisticSelfVerifier as _ProbabilisticSelfVerifier,
    resolve_implementation_model,
    resolve_implementation_thinking,
)

_raw_assess = assess
_raw_release = release
_raw_finish = finish
_raw_execute_plan = execute_plan
_deterministic_assess_candidates = assess_candidates


def _plans_dir_for(session, plans_dir=None) -> Path:
    if plans_dir is not None:
        return Path(plans_dir)
    if isinstance(session, dict) and session.get("plans_dir"):
        return Path(session["plans_dir"])
    return Path(DEFAULT_PLANS_DIR)


def _load_state(session, plans_dir=None):
    if isinstance(session, dict):
        return session
    try:
        return plan_mode._load_session(_plans_dir_for(session, plans_dir), session)
    except Exception:
        return None


def _best_plan_text(session, plans_dir=None) -> str:
    state = _load_state(session, plans_dir)
    if not isinstance(state, dict):
        return ""
    rounds = state.get("rounds") or []
    version = state.get("best_version")
    if isinstance(version, int) and 1 <= version <= len(rounds):
        return str(rounds[version - 1].get("plan_text") or "")
    if rounds:
        return str(rounds[-1].get("plan_text") or "")
    return ""


def _persist_status(session, status_value: str, plans_dir=None, **metadata) -> None:
    state = _load_state(session, plans_dir)
    if not isinstance(state, dict):
        return
    state["status"] = status_value
    state.update(metadata)
    try:
        plan_mode._save_session(_plans_dir_for(state, plans_dir), state)
    except Exception:
        pass


def assess(session, plan_text, *, note=None, addressed=None, plans_dir=None,
           require_execution_contract=False, run_probe=False, probe_cwd=None,
           execution_evidence=None, require_execution_evidence=False,
           conflicts=None, require_conflict_free=False):
    """Assess a plan, but reserve ``converged`` for a clean deterministic state.

    The underlying optimizer may stop because it plateaued or exhausted its
    round budget.  That is not equivalent to correctness.  If it returns
    ``converged`` while critiques or hard checks remain, the public status is
    changed to ``plateaued`` and the persisted session is not release-eligible.
    """
    result = _raw_assess(
        session,
        plan_text,
        note=note,
        addressed=addressed,
        plans_dir=plans_dir,
        require_execution_contract=require_execution_contract,
        run_probe=run_probe,
        probe_cwd=probe_cwd,
        execution_evidence=execution_evidence,
        require_execution_evidence=require_execution_evidence,
        conflicts=conflicts,
        require_conflict_free=require_conflict_free,
    )
    if result.get("status") != "converged":
        return result

    cwd = Path(probe_cwd or Path.cwd())
    checked_verify = plan_mode.verify(plan_text)
    checked_ground = plan_mode.ground_check(plan_text, cwd=cwd)
    checked_sim = plan_mode.simulate(plan_text, initial_state=set(checked_ground.get("verified", [])))
    clean = bool(
        not result.get("critiques")
        and checked_verify.get("ok")
        and checked_ground.get("ok")
        and checked_sim.get("executable_plan")
    )
    if require_execution_contract:
        contract = result.get("execution_contract") or {}
        clean = clean and bool(contract.get("ok"))
        if run_probe and (result.get("probe") or {}).get("configured"):
            clean = clean and bool((result.get("probe") or {}).get("ok"))
    if require_execution_evidence:
        clean = clean and bool((result.get("execution_trace") or {}).get("ok"))

    result["clean_convergence"] = clean
    result["convergence_checks"] = {
        "zero_critiques": not bool(result.get("critiques")),
        "verify_ok": bool(checked_verify.get("ok")),
        "ground_ok": bool(checked_ground.get("ok")),
        "sim_ok": bool(checked_sim.get("executable_plan")),
    }
    if clean:
        _persist_status(session, "converged", plans_dir, convergence_quality="clean")
        return result

    result["status"] = "plateaued"
    result["continue"] = False
    result["requires_revision"] = True
    result["convergence_quality"] = "stopped-with-open-issues"
    _persist_status(
        session,
        "plateaued",
        plans_dir,
        convergence_quality="stopped-with-open-issues",
    )
    return result


def release(session, *, min_score=90.0, require_judge=True,
            require_external_judge=False, require_execution_contract=False,
            execution_cwd=None, execution_evidence=None,
            require_execution_evidence=False, conflicts=None,
            require_conflict_free=False, plans_dir=None):
    """Release with an additional canonical-CWD feasibility/simulation guard."""
    gate = _raw_release(
        session,
        min_score=min_score,
        require_judge=require_judge,
        require_external_judge=require_external_judge,
        require_execution_contract=require_execution_contract,
        execution_cwd=execution_cwd,
        execution_evidence=execution_evidence,
        require_execution_evidence=require_execution_evidence,
        conflicts=conflicts,
        require_conflict_free=require_conflict_free,
        plans_dir=plans_dir,
    )
    if execution_cwd is None:
        return gate

    text = _best_plan_text(session, plans_dir)
    strict_gc = plan_mode.ground_check(text, cwd=Path(execution_cwd)) if text else {"ok": False, "missing": ["no best plan"]}
    strict_sim = plan_mode.simulate(text, initial_state=set(strict_gc.get("verified", []))) if text else {"executable_plan": False, "problems": ["no best plan"]}
    gate["execution_cwd_checks"] = {
        "cwd": str(Path(execution_cwd).resolve()),
        "ground_ok": bool(strict_gc.get("ok")),
        "sim_ok": bool(strict_sim.get("executable_plan")),
        "missing": strict_gc.get("missing", []),
        "simulation_problems": strict_sim.get("problems", []),
    }
    if not strict_gc.get("ok"):
        gate["ok"] = False
        gate.setdefault("problems", []).append(
            f"execution_cwd grounding failed: {strict_gc.get('missing', [])[:5]}"
        )
    if not strict_sim.get("executable_plan"):
        gate["ok"] = False
        gate.setdefault("problems", []).append(
            f"execution_cwd simulation failed: {strict_sim.get('problems', [])[:5]}"
        )
    return gate


def finish(session, *, verdict="converged", plans_dir=None, require_release=True,
           min_score=90.0, require_judge=True, require_external_judge=False,
           require_execution_contract=False, execution_cwd=None,
           execution_evidence=None, require_execution_evidence=False,
           conflicts=None, require_conflict_free=False):
    """Finish only after the hardened public release gate passes."""
    if require_release:
        gate = release(
            session,
            min_score=min_score,
            require_judge=require_judge,
            require_external_judge=require_external_judge,
            require_execution_contract=require_execution_contract,
            execution_cwd=execution_cwd,
            execution_evidence=execution_evidence,
            require_execution_evidence=require_execution_evidence,
            conflicts=conflicts,
            require_conflict_free=require_conflict_free,
            plans_dir=plans_dir,
        )
        if not gate.get("ok"):
            return {
                "ok": False,
                "status": (_load_state(session, plans_dir) or {}).get("status"),
                "error": "release gate failed",
                "release_gate": gate,
            }
        # Gate already ran with canonical cwd/evidence; avoid the legacy
        # internal release pass using a different cwd.
        result = _raw_finish(
            session,
            verdict=verdict,
            plans_dir=plans_dir,
            require_release=False,
        )
        if isinstance(result, dict):
            result["release_gate"] = gate
        return result
    return _raw_finish(
        session,
        verdict=verdict,
        plans_dir=plans_dir,
        require_release=False,
    )


def _is_async_handler(handler) -> bool:
    return bool(
        inspect.iscoroutinefunction(handler)
        or inspect.iscoroutinefunction(getattr(handler, "__call__", None))
    )


async def execute_plan(plan_text, task_handlers=None, *, dry_run=False,
                       continue_on_error=False, timeout_per_task=None,
                       context=None):
    """Fail-closed compatibility wrapper around the legacy Cordis executor.

    Production capability execution should use ``TransactionalExecutionManager``.
    This compatibility API now requires one async handler for every task and a
    finite timeout, so a missing handler can never become synthetic success and
    a blocking synchronous callback cannot bypass the timeout.
    """
    handlers = dict(task_handlers or {})
    nodes = list((plan_mode.plan_dag(plan_text) or {}).get("nodes", []))
    if not dry_run:
        missing = [task_id for task_id in nodes if task_id not in handlers]
        if missing:
            return {
                "ok": False,
                "error": f"missing task handlers for tasks {missing}; refusing synthetic success",
                "failed_task": missing[0],
                "executed_tasks": [],
                "recovered": False,
            }
        sync_handlers = [task_id for task_id, handler in handlers.items() if not _is_async_handler(handler)]
        if sync_handlers:
            return {
                "ok": False,
                "error": f"synchronous task handlers are not allowed: {sync_handlers}",
                "failed_task": sync_handlers[0],
                "executed_tasks": [],
                "recovered": False,
            }
    effective_timeout = 60.0 if timeout_per_task is None else float(timeout_per_task)
    if effective_timeout <= 0:
        raise ValueError("timeout_per_task must be > 0")
    return await _raw_execute_plan(
        plan_text,
        task_handlers=handlers,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
        timeout_per_task=effective_timeout,
        context=context,
    )


def _compat_ranking(checks, preferred_order=None):
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
    return [
        {
            "candidate": i,
            "score": None,
            "effective_score": 100.0 if by_index[i]["hard_pass"] else 0.0,
            "sim_ok": by_index[i]["sim_ok"],
            "verify_ok": by_index[i]["verify_ok"],
            "feasibility_ok": by_index[i]["feasibility_ok"],
        }
        for i in preferred + remaining
    ]


def _session_state_for_model(session, plans_dir=None):
    return _load_state(session, plans_dir)


def assess_candidates(
    session,
    drafts,
    *,
    notes=None,
    plans_dir=None,
    verifier=None,
    implementation_model=None,
    implementation_thinking=None,
    n_evaluations=DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
    pivots=DEFAULT_SELF_VERIFICATION_PIVOTS,
    request_timeout_seconds=DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
    max_workers=DEFAULT_VERIFIER_MAX_WORKERS,
    max_verifier_calls=DEFAULT_VERIFIER_MAX_CALLS,
):
    """Inherited Best-of-N with bounded same-model, same-thinking verification."""
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
        verified = plan_mode.verify(draft)
        grounded = plan_mode.ground_check(draft)
        simulated = plan_mode.simulate(draft, initial_state=set(grounded.get("verified", [])))
        hard_pass = bool(verified.get("ok") and grounded.get("ok") and simulated.get("executable_plan"))
        checked.append({
            "candidate": i,
            "verify_ok": bool(verified.get("ok")),
            "feasibility_ok": bool(grounded.get("ok")),
            "sim_ok": bool(simulated.get("executable_plan")),
            "hard_pass": hard_pass,
        })
        if hard_pass:
            pass_indices.append(i)

    eligible = pass_indices if pass_indices else list(range(len(drafts)))
    if len(eligible) == 1:
        chosen = eligible[0]
        result = assess(
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
    active_model = resolve_implementation_model(implementation_model, session=session_state)
    active_thinking = resolve_implementation_thinking(implementation_thinking, session=session_state)
    if not active_model:
        result = _deterministic_assess_candidates(session, drafts, notes=notes, plans_dir=plans_dir)
        result.update({
            "selection_method": "deterministic-fallback-no-model",
            "self_verification_available": False,
            "self_verification_error": "Active implementation-model identity unavailable; no verifier model was substituted",
            "candidate_checks": checked,
        })
        return result

    objective = str((session_state or {}).get("objective") or "Select the best candidate plan")
    soft_verifier = verifier or _ProbabilisticSelfVerifier()
    try:
        soft = soft_verifier.select(
            problem=objective,
            candidates=[drafts[i] for i in eligible],
            model=active_model,
            thinking_profile=active_thinking,
            n_evaluations=n_evaluations,
            pivots=pivots,
            request_timeout_seconds=request_timeout_seconds,
            max_workers=max_workers,
            max_verifier_calls=max_verifier_calls,
        )
        chosen = eligible[soft.selected_index]
        soft_order = [eligible[i] for i in soft.ranking]
        result = assess(
            session,
            drafts[chosen],
            note=(notes or [None] * len(drafts))[chosen],
            plans_dir=plans_dir,
        )
        result.update({
            "selection_method": "inherited-same-model-same-thinking-self-verification",
            "selected_candidate": chosen,
            "implementation_model": active_model,
            "generator_model": active_model,
            "verifier_model": active_model,
            "implementation_thinking": dict(active_thinking),
            "generator_thinking": dict(active_thinking),
            "verifier_thinking": dict(active_thinking),
            "is_self_verification": True,
            "is_same_thinking": True,
            "n_evaluations": n_evaluations,
            "pivots": min(pivots, len(eligible)),
            "eligible_candidates": eligible,
            "candidate_checks": checked,
            "probabilistic_scores": soft.scores,
            "probabilistic_ranking": soft_order,
            "ranking": _compat_ranking(checked, soft_order),
            "candidates_scored": len(drafts),
            "estimated_verifier_calls": getattr(soft, "estimated_calls", None),
        })
        return result
    except Exception as exc:
        result = _deterministic_assess_candidates(session, drafts, notes=notes, plans_dir=plans_dir)
        result.update({
            "selection_method": "deterministic-fallback",
            "implementation_model": active_model,
            "implementation_thinking": dict(active_thinking),
            "self_verification_available": False,
            "self_verification_error": f"{type(exc).__name__}: {exc}",
            "candidate_checks": checked,
        })
        return result


# Install hardened public wrappers back onto plan_mode for code that imports
# ``plan`` first and subsequently reaches through the implementation module.
plan_mode.assess = assess
plan_mode.release = release
plan_mode.finish = finish
plan_mode.execute_plan = execute_plan
plan_mode.assess_candidates = assess_candidates

__all__ = list(plan_mode.__all__)
