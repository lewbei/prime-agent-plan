"""Late-bound hardening for the public ``plan_mode`` API.

This module contains correctness gates that must operate on the public module
namespace so test/runtime substitutions of verifier and grounding functions
remain observable. ``install_api_hardening`` is called only after the legacy API
implementation has finished loading.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, MutableMapping

from .search_stable import stable_rng_for_text

_ATTESTATION_KEY = secrets.token_bytes(32)


def _verdict_payload(verdict: dict[str, Any]) -> bytes:
    data = {
        "ok": bool(verdict.get("ok")),
        "verdict": verdict.get("verdict"),
        "falsifiable_criteria": bool(verdict.get("falsifiable_criteria")),
        "provider": verdict.get("provider"),
        "model": verdict.get("model"),
        "source": verdict.get("source"),
        "external": bool(verdict.get("external")),
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _attest_external(verdict: dict[str, Any]) -> dict[str, Any]:
    out = dict(verdict)
    if (
        out.get("ok")
        and out.get("external") is True
        and out.get("source") == "external_llm"
        and out.get("provider")
        and out.get("model")
    ):
        out["_runtime_attestation"] = hmac.new(
            _ATTESTATION_KEY, _verdict_payload(out), hashlib.sha256
        ).hexdigest()
    return out


def _verify_attestation(verdict: dict[str, Any]) -> bool:
    token = verdict.get("_runtime_attestation")
    if not isinstance(token, str) or not token:
        return False
    expected = hmac.new(_ATTESTATION_KEY, _verdict_payload(verdict), hashlib.sha256).hexdigest()
    return hmac.compare_digest(token, expected)


def install_api_hardening(ns: MutableMapping[str, Any]) -> None:
    """Install hardened public API functions after the main implementation loads."""
    legacy_judge = ns["judge"]

    async def judge(plan_text: str, objective: str = "", *,
                    model: str | None = None, timeout: int | None = None) -> dict[str, Any]:
        result = await legacy_judge(plan_text, objective, model=model, timeout=timeout)
        return _attest_external(result) if isinstance(result, dict) else result

    def record_judge(session: dict[str, Any] | str, verdict: dict[str, Any], *,
                     round_version: int | None = None,
                     plans_dir: str | Path | None = None) -> dict[str, Any]:
        plans_dir_path = Path(plans_dir) if plans_dir else (
            Path(session.get("plans_dir"))
            if isinstance(session, dict) and session.get("plans_dir")
            else ns["DEFAULT_PLANS_DIR"]
        )
        sid = session if isinstance(session, str) else session.get("session_id", "default")
        with ns["session_lock"](plans_dir_path, sid):
            s = ns["_load_session"](plans_dir_path, session) if isinstance(session, str) else session
            ver = round_version if round_version is not None else s.get("best_version") or len(s.get("rounds", []))
            plan_hash = None
            rounds = s.get("rounds") or []
            if isinstance(ver, int) and 1 <= ver <= len(rounds):
                plan_hash = hashlib.sha256(
                    str(rounds[ver - 1].get("plan_text", "")).encode("utf-8")
                ).hexdigest()
            incoming = dict(verdict)
            provenance_verified = _verify_attestation(incoming)
            incoming.pop("_runtime_attestation", None)
            entry = {
                "ts": ns["_now"](),
                "round_version": ver,
                "plan_hash": plan_hash,
                **incoming,
                "provenance_verified": provenance_verified,
            }
            s.setdefault("judge_log", []).append(entry)
            ns["_save_session"](plans_dir_path, s)
            return entry

    async def judge_ensemble(session: dict[str, Any] | str, plan_text: str, objective: str,
                            n: int = 3, *, plans_dir: str | Path | None = None) -> dict[str, Any]:
        if isinstance(session, str):
            plans_dir_path = Path(plans_dir) if plans_dir else ns["DEFAULT_PLANS_DIR"]
            s = ns["_load_session"](plans_dir_path, session)
        else:
            s = session
            plans_dir_path = Path(plans_dir) if plans_dir else Path(s.get("plans_dir") or ns["DEFAULT_PLANS_DIR"])

        v = ns["verify"](plan_text)
        si = ns["simulate"](plan_text)
        current_version = s.get("best_version")
        current_hash = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
        baseline = {
            "ok": True,
            "verdict": "go" if (v["ok"] and si["executable_plan"]) else "rework",
            "feasibility_0_100": 100 if (v["ok"] and si["executable_plan"]) else 40,
            "falsifiable_criteria": True,
            "judge_path": "local-deterministic-fallback",
            "source": "mechanical_baseline",
            "external": False,
            "round_version": current_version,
            "plan_hash": current_hash,
        }
        votes: list[dict[str, Any]] = [baseline]
        try:
            live = await ns["judge"](plan_text, objective)
            if isinstance(live, dict) and live.get("ok") and live.get("falsifiable_criteria"):
                votes.append({**live, "round_version": current_version, "plan_hash": current_hash})
        except Exception:
            pass

        for prior in reversed(s.get("judge_log", [])):
            if len(votes) >= max(1, n):
                break
            if prior.get("round_version") != current_version:
                continue
            prior_hash = prior.get("plan_hash")
            if prior_hash is not None and prior_hash != current_hash:
                continue
            if prior.get("ok") and prior.get("falsifiable_criteria"):
                votes.append(prior)

        votes = votes[:max(1, min(n, len(votes)))]
        feases = sorted(vt.get("feasibility_0_100", 0) for vt in votes)
        median = feases[(len(feases) - 1) // 2]
        med_vote = min(votes, key=lambda vt: abs(vt.get("feasibility_0_100", 0) - median))
        entry = {
            **med_vote,
            "ensemble": True,
            "votes": votes,
            "median_feasibility": median,
            "verdict": med_vote.get("verdict", "go"),
            "ok": True,
        }
        ns["record_judge"](
            s,
            entry,
            round_version=current_version,
            plans_dir=plans_dir_path,
        )
        return entry

    def release(session: dict[str, Any] | str, *, min_score: float = 90.0,
                require_judge: bool = True,
                require_external_judge: bool = False,
                require_execution_contract: bool = False,
                execution_cwd: str | Path | None = None,
                execution_evidence: str | dict[str, Any] | None = None,
                require_execution_evidence: bool = False,
                conflicts: Any = None,
                require_conflict_free: bool = False,
                plans_dir: str | Path | None = None) -> dict[str, Any]:
        plans_dir_path = Path(plans_dir) if plans_dir else (
            Path(session.get("plans_dir"))
            if isinstance(session, dict) and session.get("plans_dir")
            else ns["DEFAULT_PLANS_DIR"]
        )
        sid = session if isinstance(session, str) else session.get("session_id", "default")
        canonical_cwd = Path(execution_cwd or Path.cwd()).resolve()

        with ns["session_lock"](plans_dir_path, sid):
            s = ns["_load_session"](plans_dir_path, session) if isinstance(session, str) else session
            checks: list[dict[str, Any]] = []
            problems: list[str] = []

            converged = s.get("status") == "converged"
            checks.append({"name": "converged", "ok": converged, "detail": f"status={s.get('status')}"})
            if not converged:
                problems.append("plan has not converged; keep looping assess->revise")

            best_score = s.get("best_score") or 0
            score_ok = best_score >= min_score
            checks.append({"name": "score", "ok": score_ok, "detail": f"best={best_score} >= {min_score}"})
            if not score_ok:
                problems.append(f"best score {best_score} < {min_score}; keep revising")

            best_ver = s.get("best_version")
            rounds = s.get("rounds") or []
            best_text = ""
            if isinstance(best_ver, int) and 1 <= best_ver <= len(rounds):
                best_text = str(rounds[best_ver - 1].get("plan_text", ""))

            mech = ns["_mechanical_checks"](best_text) if best_text else [{"id": "mech:empty", "hint": "no best plan yet"}]
            checks.append({"name": "mechanical", "ok": not mech, "detail": str([c["hint"] for c in mech])[:120]})
            problems.extend(c["hint"] for c in mech)

            v = ns["verify"](best_text, cwd=canonical_cwd) if best_text else {"ok": False, "errors": ["no best plan yet"]}
            checks.append({"name": "verify", "ok": bool(v.get("ok")), "detail": str(v.get("errors", []))[:120]})
            if not v.get("ok"):
                problems.extend(v.get("errors", []))

            gc = ns["ground_check"](best_text, cwd=canonical_cwd) if best_text else {"ok": False, "missing": ["no best plan yet"], "verified": []}
            checks.append({"name": "feasibility", "ok": bool(gc.get("ok")), "detail": f"missing inputs: {gc.get('missing', [])[:3]}"})
            if not gc.get("ok"):
                problems.append(f"declared inputs do not exist: {gc.get('missing', [])[:5]}")

            sim = ns["simulate"](best_text, initial_state=set(gc.get("verified", []))) if best_text else {"executable_plan": False, "problems": ["no best plan yet"]}
            checks.append({"name": "simulation", "ok": bool(sim.get("executable_plan")), "detail": str(sim.get("problems", []))[:120]})
            if not sim.get("executable_plan"):
                problems.extend(sim.get("problems", []))

            ec = ns["validate_execution_contract"](best_text, cwd=canonical_cwd) if best_text else {"ok": False, "errors": ["no best plan yet"], "contract": None}
            contract_ok = bool(ec.get("ok")) if require_execution_contract else not (
                ec.get("contract") is not None and ec.get("errors")
            )
            probe_ok = True
            if require_execution_contract and ec.get("contract") is not None:
                probe_cfg = ec["contract"].probe
                probe_last = s.get("probe_last")
                if probe_cfg and probe_cfg.get("command"):
                    probe_ok = bool(probe_last and probe_last.get("configured") and probe_last.get("ok"))
                    if not probe_ok:
                        problems.append("feasibility probe not passed; run assess(..., run_probe=True) and revise until the spike succeeds")
            checks.append({"name": "execution_contract", "ok": contract_ok and probe_ok, "detail": str(ec.get("errors", []))[:120] + ("; probe passed" if probe_ok else "; probe pending/failed")})
            if not contract_ok:
                problems.extend(ec.get("errors", []))

            trace = ns["verify_execution_trace"](best_text, execution_evidence, cwd=canonical_cwd) if (execution_evidence is not None or require_execution_evidence) and best_text else {"ok": True, "errors": [], "warnings": []}
            trace_ok = bool(trace.get("ok")) if require_execution_evidence else not bool(trace.get("errors"))
            checks.append({"name": "execution_trace", "ok": trace_ok, "detail": str(trace.get("errors", []))[:120]})
            if not trace_ok:
                problems.extend(trace.get("errors", []))

            if conflicts is not None:
                conflict_ok = bool(getattr(conflicts, "ok", True))
                checks.append({"name": "isolation_conflicts", "ok": conflict_ok, "detail": str(getattr(conflicts, "conflicts", []))[:120]})
                if not conflict_ok:
                    problems.extend(str(c) for c in getattr(conflicts, "conflicts", []))
            elif require_conflict_free:
                checks.append({"name": "isolation_conflicts", "ok": False, "detail": "missing conflict report"})
                problems.append("conflict report not provided; run plan.detect_conflicts() before release")
            else:
                checks.append({"name": "isolation_conflicts", "ok": True, "detail": "not required"})

            judge_ok = False
            judge_detail = "no judge verdict recorded"
            matching = None
            for entry in reversed(s.get("judge_log", [])):
                if entry.get("round_version") == best_ver:
                    matching = entry
                    break
            if matching:
                is_go = bool(matching.get("ok") and matching.get("verdict") == "go" and matching.get("falsifiable_criteria"))
                if require_external_judge:
                    is_external = bool(
                        matching.get("external") is True
                        and matching.get("source") == "external_llm"
                        and matching.get("provenance_verified") is True
                    )
                    judge_ok = is_go and is_external
                    judge_detail = f"round={matching.get('round_version')} verdict={matching.get('verdict')} source={matching.get('source', 'unknown')} verified_external={is_external}"
                else:
                    judge_ok = is_go
                    judge_detail = f"round={matching.get('round_version')} verdict={matching.get('verdict')} source={matching.get('source', 'unknown')} feasibility={matching.get('feasibility_0_100')}"
            checks.append({"name": "judge", "ok": judge_ok or not require_judge, "detail": judge_detail})
            if require_judge and not judge_ok:
                problems.append("judge gate not passed; run plan.judge + record_judge, fix blockers, re-assess")

            ok = all(bool(c["ok"]) for c in checks)
            report = {"ok": ok, "checks": checks, "problems": problems}

            # Atomic publication: commit fields are mutated only after every
            # check, including canonical-CWD checks, has passed.
            if ok and isinstance(best_ver, int):
                s["committed_version"] = best_ver
                s["committed_score"] = s.get("best_score")
                s["committed_at"] = ns["_now"]()
                s["committed_plan_hash"] = hashlib.sha256(best_text.encode("utf-8")).hexdigest()
                report["committed"] = {"version": best_ver, "score": s.get("committed_score")}
            s["release_gate"] = report
            ns["_save_session"](plans_dir_path, s)
            return report

    async def speculative_rollout_async(plan_text: str, eval_fn: Any, *,
                                        context: Any = None,
                                        timeout_seconds: float = 60.0) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        ctx = (context or ns["get_root_context"]()).derive(name="speculative_rollout_async")
        try:
            score = await asyncio.wait_for(eval_fn(ctx), timeout=float(timeout_seconds))
            return {"ok": True, "score": score, "error": None, "timed_out": False}
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "score": 0.0,
                "error": f"speculative rollout timeout after {float(timeout_seconds):.3f}s",
                "timed_out": True,
            }
        except Exception as exc:
            return {"ok": False, "score": 0.0, "error": str(exc), "timed_out": False}
        finally:
            await ctx.async_dispose()

    # Patch direct search-engine imports as well as plan.search() internals.
    search_engine = __import__("plan_mode.search_engine", fromlist=["_mutations"])
    original_mutations = search_engine._mutations

    def stable_mutations(plan_text: str, width: int, critiques: list[dict[str, str]] | None = None):
        # Preserve the authoritative mutation implementation while temporarily
        # replacing its local RNG seed source. Reimplement the tiny selection
        # portion here to avoid Python's process-randomized hash().
        rng = stable_rng_for_text(search_engine._norm(plan_text))
        targeted: list[tuple[str, Any]] = []
        if critiques:
            seen: set[str] = set()
            for critique in critiques:
                sec = critique.get("id", "").split(":", 1)[0] if isinstance(critique, dict) else str(critique).split(":", 1)[0]
                for key, tmpl in search_engine._SECTION_TEMPLATES.items():
                    if key in sec and key not in seen:
                        seen.add(key)
                        targeted.append((f"target-{key}", lambda text, tmpl=tmpl: text + tmpl))
        if targeted:
            chosen = rng.sample(targeted, min(width, len(targeted)))
            return [{"text": fn(plan_text), "note": name} for name, fn in chosen]
        chosen = rng.sample(search_engine._MUTATIONS, min(width, len(search_engine._MUTATIONS)))
        return [{"text": fn(plan_text), "note": name} for name, fn in chosen]

    search_engine._mutations = stable_mutations
    ns["judge"] = judge
    ns["record_judge"] = record_judge
    ns["judge_ensemble"] = judge_ensemble
    ns["release"] = release
    ns["speculative_rollout_async"] = speculative_rollout_async
