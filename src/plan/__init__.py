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
    assess,
    assess_candidates,
    best,
    finish,
    history,
    judge,
    list_sessions,
    log_progress,
    plan_dag,
    plan_quality,
    record_judge,
    release,
    rubric,
    run,
    search,
    search_backtrack,
    search_expand,
    search_report,
    search_select,
    simulate,
    start,
    status,
    suggest,
    verify,
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
)

__all__ = ["run", "start", "assess", "assess_candidates", "status", "history",
           "best", "finish", "list_sessions", "rubric", "verify", "judge",
           "record_judge", "release", "plan_dag", "simulate", "plan_quality",
           "search_expand", "search_select", "search_backtrack", "search_report",
           "search", "log_progress", "suggest",
           "Context", "Fiber", "LifecycleState", "TwistedMonoid", "get_root_context",
           "reset_root_context", "create_subagent_context", "provide_tool",
           "execute_plan", "execute_plan_sync", "speculative_rollout"]
