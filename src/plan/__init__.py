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
    probe_contract,
    symbol_audit,
    scan_symbols,
    parity_audit,
    run_verification_commands,
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
    DEFAULT_PLANS_DIR,
    RUBRIC_PATH,
    REPO_ROOT,
    DEFAULT_MAX_ROUNDS,
    MAX_PLATEAU_ROUNDS,
    MIN_DELTA_TO_CONTINUE,
    JOURNAL_PATH,
    DEFAULT_RUBRIC,
)

__all__ = [
    "__version__", "start", "assess", "assess_candidates", "run", "status", "history", "best", "committed", "checkpoint", "rewind", "finish",
    "log_progress", "suggest", "list_sessions", "rubric", "verify", "judge", "record_judge",
    "release", "plan_dag", "simulate", "plan_quality", "edit_file", "rollback", "deps_check",
    "ground_check", "constraint_check", "fold_history", "judge_ensemble", "template", "selfcheck",
    "search_expand", "search_select", "search_backtrack", "search_report", "search",
    "Context", "Fiber", "LifecycleState", "TwistedMonoid", "get_root_context", "reset_root_context",
    "create_subagent_context", "provide_tool", "execute_plan", "execute_plan_sync",
    "speculative_rollout", "speculative_rollout_async", "session_lock", "ExecutionContract",
    "parse_execution_contract", "validate_execution_contract", "probe_contract",
    "symbol_audit", "scan_symbols", "parity_audit", "run_verification_commands", "RoTRuleBase", "RoTRule", "ReplanningLadder", "ContextBudgeter",
    "mutate_flaw_directed", "mutate_exploratory", "crossover_ast", "ast_distance",
    "PopulationMember", "ASTSearchEngine", "Proposition", "PlanParser", "PlanAST",
    "CausalValidator", "CausalLink", "CausalFlaw", "ActionSchema",
    "DEFAULT_PLANS_DIR", "RUBRIC_PATH", "REPO_ROOT", "DEFAULT_MAX_ROUNDS",
    "MAX_PLATEAU_ROUNDS", "MIN_DELTA_TO_CONTINUE", "JOURNAL_PATH", "DEFAULT_RUBRIC",
]
