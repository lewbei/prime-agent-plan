"""Atomic public release/finish closure installed after compatibility wrappers."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, MutableMapping

from .runtime_closure_context import workspace_identity


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

    def certificate_id_for(state: dict[str, Any]) -> str | None:
        certificate = state.get("authorization_certificate")
        if isinstance(certificate, dict):
            value = certificate.get("certificate_id")
            return str(value) if value else None
        value = getattr(certificate, "certificate_id", None)
        return str(value) if value else None

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

            text = ""
            prechecks: list[dict[str, Any]] = []
            problems: list[str] = []
            cwd_checks = None
            trace_checks = None

            if execution_cwd is not None or require_execution_evidence:
                text = ns["_best_plan_text"](state, pdir)

            if execution_cwd is not None:
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
                cwd_ok = cwd_checks["ground_ok"] and cwd_checks["sim_ok"]
                prechecks.append({
                    "name": "execution_cwd",
                    "ok": cwd_ok,
                    "detail": str(cwd_checks)[:240],
                })
                if not cwd_checks["ground_ok"]:
                    problems.append(
                        f"execution_cwd grounding failed: {cwd_checks['missing'][:5]}"
                    )
                if not cwd_checks["sim_ok"]:
                    problems.append(
                        "execution_cwd simulation failed: "
                        f"{cwd_checks['simulation_problems'][:5]}"
                    )

            if require_execution_evidence:
                evidence_cwd = Path(execution_cwd or Path.cwd()).resolve()
                expected_session_id = str(state.get("session_id") or sid)
                expected_workspace_id = workspace_identity(evidence_cwd)
                expected_certificate_id = certificate_id_for(state)
                trace = ns["verify_execution_trace"](
                    text,
                    execution_evidence,
                    cwd=evidence_cwd,
                    require_independent_verifier=True,
                    expected_session_id=expected_session_id,
                    expected_certificate_id=expected_certificate_id,
                    expected_workspace_identity=expected_workspace_id,
                )
                plan_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                attested = bool(trace.get("ok")) and bool(
                    ns["_verify_execution_trace_runtime_attestation"](
                        trace,
                        plan_hash=plan_hash,
                        session_id=expected_session_id,
                        workspace_id=expected_workspace_id,
                        certificate_id=expected_certificate_id,
                    )
                )
                trace_checks = {
                    "ok": attested,
                    "cwd": str(evidence_cwd),
                    "workspace_identity": expected_workspace_id,
                    "errors": list(trace.get("errors") or []),
                    "runtime_attested": attested,
                }
                prechecks.append({
                    "name": "execution_trace_runtime_attestation",
                    "ok": attested,
                    "detail": str(trace_checks)[:240],
                })
                if not attested:
                    problems.append(
                        "execution evidence was not independently re-observed and "
                        "runtime-attested for this plan/session/workspace"
                    )
                    problems.extend(trace_checks["errors"][:5])

            if any(not bool(check.get("ok")) for check in prechecks):
                gate = {
                    "ok": False,
                    "checks": prechecks,
                    "problems": list(dict.fromkeys(problems)),
                }
                if cwd_checks is not None:
                    gate["execution_cwd_checks"] = cwd_checks
                if trace_checks is not None:
                    gate["execution_trace_checks"] = trace_checks
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
            if trace_checks is not None:
                gate["execution_trace_checks"] = trace_checks

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
