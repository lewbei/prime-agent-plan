"""Self-contained deterministic candidate selection for the unified API."""
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

    def inspect_candidates(session, drafts):
        state = session if isinstance(session, dict) else None
        rubric = (
            (state or {}).get("rubric_snapshot")
            or ns["_load_rubric"]()
        )
        checks = []
        for index, draft in enumerate(drafts):
            scored = ns["_score"](draft, rubric)
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
                "score": float(scored.get("score", 0) or 0),
                "verify_ok": bool(verified.get("ok")),
                "feasibility_ok": bool(grounded.get("ok")),
                "sim_ok": bool(simulated.get("executable_plan")),
                "hard_pass": hard_pass,
            })
        return checks

    def ranking(checks, preferred_order=None):
        by_index = {item["candidate"]: item for item in checks}
        preferred = list(preferred_order or [])
        default_order = [
            item["candidate"]
            for item in sorted(
                checks,
                key=lambda item: (
                    -int(item["hard_pass"]),
                    -int(item["sim_ok"]),
                    -int(item["verify_ok"]),
                    -int(item["feasibility_ok"]),
                    -item["score"],
                    item["candidate"],
                ),
            )
        ]
        order = preferred + [i for i in default_order if i not in preferred]
        return [
            {
                "candidate": index,
                "score": by_index[index]["score"],
                "effective_score": (
                    by_index[index]["score"]
                    if by_index[index]["hard_pass"]
                    else max(0.0, by_index[index]["score"] - 100.0)
                ),
                "sim_ok": by_index[index]["sim_ok"],
                "verify_ok": by_index[index]["verify_ok"],
                "feasibility_ok": by_index[index]["feasibility_ok"],
            }
            for index in order
        ]

    def assess_selected(session, drafts, checks, chosen, *, notes=None, plans_dir=None):
        note_list = notes or [None] * len(drafts)
        result = ns["assess"](
            session,
            drafts[chosen],
            note=note_list[chosen],
            plans_dir=plans_dir,
        )
        result.update({
            "selected_candidate": chosen,
            "candidate_checks": checks,
            "candidates_scored": len(drafts),
        })
        return result

    def deterministic_result(
        session,
        drafts,
        checks,
        *,
        notes=None,
        plans_dir=None,
        selection_method="deterministic-fallback",
    ):
        order = [row["candidate"] for row in ranking(checks)]
        chosen = order[0]
        result = assess_selected(
            session,
            drafts,
            checks,
            chosen,
            notes=notes,
            plans_dir=plans_dir,
        )
        result.update({
            "selection_method": selection_method,
            "ranking": ranking(checks, [chosen]),
        })
        return result

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
        checks = inspect_candidates(session, drafts)
        if len(drafts) == 1:
            return deterministic_result(
                session,
                drafts,
                checks,
                notes=notes,
                plans_dir=plans_dir,
                selection_method="deterministic-single-candidate",
            )

        hard_pass_indices = [item["candidate"] for item in checks if item["hard_pass"]]
        if len(hard_pass_indices) == 1:
            chosen = hard_pass_indices[0]
            result = assess_selected(
                session,
                drafts,
                checks,
                chosen,
                notes=notes,
                plans_dir=plans_dir,
            )
            result.update({
                "selection_method": "deterministic-prefilter-single",
                "ranking": ranking(checks, [chosen]),
            })
            return result

        eligible = hard_pass_indices if hard_pass_indices else list(range(len(drafts)))
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
            result = deterministic_result(
                session,
                drafts,
                checks,
                notes=notes,
                plans_dir=plans_dir,
                selection_method="deterministic-fallback-no-model",
            )
            result.update({
                "self_verification_available": False,
                "self_verification_error": (
                    "Active implementation-model identity unavailable; "
                    "no verifier model was substituted"
                ),
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
            result = assess_selected(
                session,
                drafts,
                checks,
                chosen,
                notes=notes,
                plans_dir=plans_dir,
            )
            result.update({
                "selection_method": "inherited-same-model-same-thinking-self-verification",
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
                "probabilistic_scores": soft.scores,
                "probabilistic_ranking": soft_order,
                "ranking": ranking(checks, soft_order),
                "estimated_verifier_calls": getattr(soft, "estimated_calls", None),
            })
            return result
        except Exception as exc:
            result = deterministic_result(
                session,
                drafts,
                checks,
                notes=notes,
                plans_dir=plans_dir,
                selection_method="deterministic-fallback",
            )
            result.update({
                "implementation_model": active_model,
                "implementation_thinking": dict(active_thinking),
                "self_verification_available": False,
                "self_verification_error": f"{type(exc).__name__}: {exc}",
            })
            return result

    ns["assess_candidates"] = assess_candidates
