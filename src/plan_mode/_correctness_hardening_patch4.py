"""Restore assess_candidates response compatibility on the unified runtime."""
from __future__ import annotations

from typing import Any


def patch(ns: dict[str, Any]) -> None:
    from .self_verification import (
        DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
        DEFAULT_SELF_VERIFICATION_PIVOTS,
        DEFAULT_VERIFIER_MAX_CALLS,
        DEFAULT_VERIFIER_MAX_WORKERS,
        DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
        ProbabilisticSelfVerifier,
        resolve_implementation_model,
        resolve_implementation_thinking,
    )

    raw_candidates = ns.get("_deterministic_assess_candidates") or ns.get("assess_candidates")
    # If the first hardening layer did not expose the legacy callable, capture
    # the implementation saved by that layer through its closure-compatible
    # module global where available.
    if "_deterministic_assess_candidates" not in ns:
        ns["_deterministic_assess_candidates"] = raw_candidates

    def compat_ranking(checks, preferred_order=None):
        by_index = {item["candidate"]: item for item in checks}
        preferred = list(preferred_order or [])
        remaining = [
            item["candidate"]
            for item in sorted(
                checks,
                key=lambda item: (
                    -int(item["hard_pass"]),
                    -int(item["sim_ok"]),
                    -int(item["verify_ok"]),
                    -int(item["feasibility_ok"]),
                    item["candidate"],
                ),
            )
            if item["candidate"] not in preferred
        ]
        order = preferred + remaining
        return [
            {
                "candidate": index,
                "score": None,
                "effective_score": 100.0 if by_index[index]["hard_pass"] else 0.0,
                "sim_ok": by_index[index]["sim_ok"],
                "verify_ok": by_index[index]["verify_ok"],
                "feasibility_ok": by_index[index]["feasibility_ok"],
            }
            for index in order
        ]

    def deterministic(session, drafts, *, notes=None, plans_dir=None):
        legacy = ns.get("_deterministic_assess_candidates")
        if legacy is None:
            raise RuntimeError("deterministic assess_candidates implementation unavailable")
        return legacy(session, drafts, notes=notes, plans_dir=plans_dir)

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
        if not drafts:
            raise ValueError("drafts must be non-empty")
        if len(drafts) == 1:
            result = deterministic(session, drafts, notes=notes, plans_dir=plans_dir)
            result["selection_method"] = "deterministic-single-candidate"
            return result

        checks = []
        pass_indices = []
        for index, draft in enumerate(drafts):
            verified = ns["verify"](draft)
            grounded = ns["ground_check"](draft)
            simulated = ns["simulate"](
                draft,
                initial_state=set(grounded.get("verified", [])),
            )
            hard_pass = bool(
                verified.get("ok")
                and grounded.get("ok")
                and simulated.get("executable_plan")
            )
            checks.append({
                "candidate": index,
                "verify_ok": bool(verified.get("ok")),
                "feasibility_ok": bool(grounded.get("ok")),
                "sim_ok": bool(simulated.get("executable_plan")),
                "hard_pass": hard_pass,
            })
            if hard_pass:
                pass_indices.append(index)

        eligible = pass_indices if pass_indices else list(range(len(drafts)))
        note_list = notes or [None] * len(drafts)
        if len(eligible) == 1:
            chosen = eligible[0]
            result = ns["assess"](
                session,
                drafts[chosen],
                note=note_list[chosen],
                plans_dir=plans_dir,
            )
            result.update({
                "selection_method": "deterministic-prefilter-single",
                "selected_candidate": chosen,
                "candidate_checks": checks,
                "ranking": compat_ranking(checks, [chosen]),
                "candidates_scored": len(drafts),
            })
            return result

        state = session if isinstance(session, dict) else None
        if state is None:
            try:
                pdir = plans_dir or ns["DEFAULT_PLANS_DIR"]
                state = ns["_load_session"](pdir, session)
            except Exception:
                state = None
        active_model = resolve_implementation_model(
            implementation_model,
            session=state,
        )
        active_thinking = resolve_implementation_thinking(
            implementation_thinking,
            session=state,
        )
        if not active_model:
            result = deterministic(session, drafts, notes=notes, plans_dir=plans_dir)
            result.update({
                "selection_method": "deterministic-fallback-no-model",
                "self_verification_available": False,
                "self_verification_error": (
                    "Active implementation-model identity unavailable; "
                    "no verifier model was substituted"
                ),
                "candidate_checks": checks,
            })
            return result

        selector = verifier or ProbabilisticSelfVerifier()
        try:
            soft = selector.select(
                problem=str((state or {}).get("objective") or "Select the best candidate plan"),
                candidates=[drafts[index] for index in eligible],
                model=active_model,
                thinking_profile=active_thinking,
                n_evaluations=n_evaluations,
                pivots=pivots,
                request_timeout_seconds=request_timeout_seconds,
                max_workers=max_workers,
                max_verifier_calls=max_verifier_calls,
            )
            chosen = eligible[soft.selected_index]
            soft_order = [eligible[index] for index in soft.ranking]
            result = ns["assess"](
                session,
                drafts[chosen],
                note=note_list[chosen],
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
                "candidate_checks": checks,
                "probabilistic_scores": soft.scores,
                "probabilistic_ranking": soft_order,
                "ranking": compat_ranking(checks, soft_order),
                "candidates_scored": len(drafts),
                "estimated_verifier_calls": getattr(soft, "estimated_calls", None),
            })
            return result
        except Exception as exc:
            result = deterministic(session, drafts, notes=notes, plans_dir=plans_dir)
            result.update({
                "selection_method": "deterministic-fallback",
                "implementation_model": active_model,
                "implementation_thinking": dict(active_thinking),
                "self_verification_available": False,
                "self_verification_error": f"{type(exc).__name__}: {exc}",
                "candidate_checks": checks,
            })
            return result

    ns["assess_candidates"] = assess_candidates
