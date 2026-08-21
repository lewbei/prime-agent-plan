"""Atomic public release/finish closure installed after compatibility wrappers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, MutableMapping


def install_public_release_closure(ns: MutableMapping[str, Any]) -> None:
    def plans_dir_for(session, plans_dir=None) -> Path:
        if plans_dir is not None:
            return Path(plans_dir)
        if isinstance(session, dict) and session.get("plans_dir"):
            return Path(session["plans_dir"])
        return Path(ns["DEFAULT_PLANS_DIR"])

    def load_state(session, plans_dir=None):
        if isinstance(session, dict):
            return session
        return ns["_load_session"](plans_dir_for(session, plans_dir), session)

    def best_plan_text(state: dict[str, Any]) -> str:
        rounds = state.get("rounds") or []
        version = state.get("best_version")
        if isinstance(version, int) and 1 <= version <= len(rounds):
            return str(rounds[version - 1].get("plan_text") or "")
        if rounds:
            return str(rounds[-1].get("plan_text") or "")
        return ""

    def release(
        session,
        *,
        min_score=90.0,
        require_judge=True,
        require_external_judge=False,
        require_execution_contract=False,
        execution_cwd=None,
        execution_evidence=None,
        require_execution_evidence=False,
        conflicts=None,
        require_conflict_free=False,
        plans_dir=None,
    ):
        pdir = plans_dir_for(session, plans_dir)
        sid = session if isinstance(session, str) else session.get("session_id", "default")
        with ns["session_lock"](pdir, sid):
            state = load_state(session, pdir)
            commit_keys = (
                "committed_version",
                "committed_score",
                "committed_at",
                "committed_plan_hash",
                "release_gate",
            )
            snapshot = {
                key: (key in state, copy.deepcopy(state.get(key)))
                for key in commit_keys
            }

            cwd_checks = None
            if execution_cwd is not None:
                text = best_plan_text(state)
                cwd = Path(execution_cwd).resolve()
                grounded = ns["ground_check"](text, cwd=cwd) if text else {
                    "ok": False,
                    "missing": ["no best plan"],
                    "verified": [],
                }
                simulated = ns["simulate"](
                    text,
                    initial_state=set(grounded.get("verified", [])),
                ) if text else {
                    "executable_plan": False,
                    "problems": ["no best plan"],
                }
                cwd_checks = {
                    "cwd": str(cwd),
                    "ground_ok": bool(grounded.get("ok")),
                    "sim_ok": bool(simulated.get("executable_plan")),
                    "missing": grounded.get("missing", []),
                    "simulation_problems": simulated.get("problems", []),
                }
                if not cwd_checks["ground_ok"] or not cwd_checks["sim_ok"]:
                    problems = []
                    if not cwd_checks["ground_ok"]:
                        problems.append(
                            f"execution_cwd grounding failed: {cwd_checks['missing'][:5]}"
                        )
                    if not cwd_checks["sim_ok"]:
                        problems.append(
                            "execution_cwd simulation failed: "
                            f"{cwd_checks['simulation_problems'][:5]}"
                        )
                    gate = {
                        "ok": False,
                        "checks": [{
                            "name": "execution_cwd",
                            "ok": False,
                            "detail": str(problems)[:240],
                        }],
                        "problems": problems,
                        "execution_cwd_checks": cwd_checks,
                    }
                    state["release_gate"] = gate
                    if state.get("session_id"):
                        ns["_save_session"](pdir, state)
                    return gate

            gate = ns["_raw_release"](
                state,
                min_score=min_score,
                require_judge=require_judge,
                require_external_judge=require_external_judge,
                require_execution_contract=require_execution_contract,
                execution_cwd=execution_cwd,
                execution_evidence=execution_evidence,
                require_execution_evidence=require_execution_evidence,
                conflicts=conflicts,
                require_conflict_free=require_conflict_free,
                plans_dir=pdir,
            )
            if cwd_checks is not None:
                gate["execution_cwd_checks"] = cwd_checks

            if not gate.get("ok"):
                for key, (was_present, value) in snapshot.items():
                    if was_present:
                        state[key] = value
                    else:
                        state.pop(key, None)
                state["release_gate"] = gate
                if state.get("session_id"):
                    ns["_save_session"](pdir, state)
            return gate

    def finish(
        session,
        *,
        verdict="converged",
        plans_dir=None,
        require_release=True,
        min_score=90.0,
        require_judge=True,
        require_external_judge=False,
        require_execution_contract=False,
        execution_cwd=None,
        execution_evidence=None,
        require_execution_evidence=False,
        conflicts=None,
        require_conflict_free=False,
    ):
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
                state = load_state(session, plans_dir)
                return {
                    "ok": False,
                    "status": state.get("status"),
                    "error": "release gate failed",
                    "release_gate": gate,
                }
            result = ns["_raw_finish"](
                session,
                verdict=verdict,
                plans_dir=plans_dir,
                require_release=False,
            )
            if isinstance(result, dict):
                result["release_gate"] = gate
            return result
        return ns["_raw_finish"](
            session,
            verdict=verdict,
            plans_dir=plans_dir,
            require_release=False,
        )

    ns["release"] = release
    ns["finish"] = finish
