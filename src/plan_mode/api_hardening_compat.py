"""Compatibility layer for metadata-poor single-round judge histories.

Older one-round sessions recorded judge votes before round/version hashes were
stored. Those votes are safe to reuse only while there is exactly one plan
round; once a session has multiple versions, only explicitly version/hash-bound
votes may participate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, MutableMapping


def install_api_hardening_compat(ns: MutableMapping[str, Any]) -> None:
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
        rounds = s.get("rounds") or []
        single_round_legacy = len(rounds) == 1

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
            prior_version = prior.get("round_version")
            if prior_version is None:
                if not single_round_legacy:
                    continue
            elif prior_version != current_version:
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

    ns["judge_ensemble"] = judge_ensemble
