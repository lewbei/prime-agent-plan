"""Final API unification: make release helpers dynamically patchable and atomic."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def patch(ns: dict[str, Any]) -> None:
    def plans_dir_for(session, plans_dir=None):
        if plans_dir is not None:
            return Path(plans_dir)
        if isinstance(session, dict) and session.get("plans_dir"):
            return Path(session["plans_dir"])
        return Path(ns["DEFAULT_PLANS_DIR"])

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
            state = ns["_load_session"](pdir, session) if isinstance(session, str) else session
            snapshot = {
                key: copy.deepcopy(state.get(key))
                for key in (
                    "committed_version",
                    "committed_score",
                    "committed_at",
                    "committed_plan_hash",
                    "release_gate",
                )
            }
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
            if execution_cwd is not None:
                text = ns["_best_plan_text"](state, pdir)
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
                gate["execution_cwd_checks"] = {
                    "cwd": str(cwd),
                    "ground_ok": bool(grounded.get("ok")),
                    "sim_ok": bool(simulated.get("executable_plan")),
                    "missing": grounded.get("missing", []),
                    "simulation_problems": simulated.get("problems", []),
                }
                if not grounded.get("ok"):
                    gate["ok"] = False
                    gate.setdefault("problems", []).append(
                        f"execution_cwd grounding failed: {grounded.get('missing', [])[:5]}"
                    )
                if not simulated.get("executable_plan"):
                    gate["ok"] = False
                    gate.setdefault("problems", []).append(
                        f"execution_cwd simulation failed: {simulated.get('problems', [])[:5]}"
                    )
            if not gate.get("ok"):
                for key, value in snapshot.items():
                    state[key] = value
                state["release_gate"] = gate
                # The observable release result and durable commit state are one
                # atomic truth: failed final gates cannot leave a commit behind.
                if isinstance(state, dict) and state.get("session_id"):
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
                state = session if isinstance(session, dict) else ns["_load_session"](
                    plans_dir_for(session, plans_dir), session
                )
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
