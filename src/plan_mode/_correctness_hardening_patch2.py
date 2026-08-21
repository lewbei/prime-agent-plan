"""Second-pass corrections discovered by the first GREEN CI run."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def patch(ns: dict[str, Any]) -> None:
    from . import ast_search as ast_search_mod
    from . import judges as judges_mod

    # ActionSchema intentionally has a smaller field surface than ActionIR.
    # Bind the transposition identity to every semantic field it actually owns
    # without assuming fields from another representation.
    def ast_state_hash(self, ast):
        def atom(value):
            if hasattr(value, "model_dump"):
                return atom(value.model_dump(mode="json"))
            if isinstance(value, dict):
                return {
                    str(k): atom(v)
                    for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                }
            if isinstance(value, (list, tuple)):
                return [atom(v) for v in value]
            if isinstance(value, set):
                rendered = [atom(v) for v in value]
                return sorted(rendered, key=_stable)
            if hasattr(value, "__dict__"):
                return {
                    str(k): atom(v)
                    for k, v in sorted(value.__dict__.items(), key=lambda item: str(item[0]))
                }
            return value

        actions = []
        for action in ast.actions:
            actions.append({
                "id": getattr(action, "id", None),
                "name": getattr(action, "name", ""),
                "preconditions": atom(getattr(action, "preconditions", [])),
                "add_effects": atom(getattr(action, "add_effects", [])),
                "del_effects": atom(getattr(action, "del_effects", [])),
                "inputs": atom(getattr(action, "inputs", [])),
                "outputs": atom(getattr(action, "outputs", [])),
                "depends_on": atom(getattr(action, "depends_on", [])),
                "duration": getattr(action, "duration", None),
                "resources": atom(getattr(action, "resources", {})),
            })
        payload = {
            "goal": ast.goal,
            "initial_state": atom(ast.initial_state),
            "target_propositions": atom(ast.target_propositions),
            "constraints": atom(ast.constraints),
            "predicate_signatures": atom(getattr(ast, "predicate_signatures", [])),
            "actions": actions,
        }
        return hashlib.sha256(_stable(payload).encode("utf-8")).hexdigest()

    ast_search_mod.ASTSearchEngine._state_hash = ast_state_hash

    # Pydantic compiles field defaults at class creation time; changing the
    # field metadata after creation is insufficient.  Enforce fail-closed
    # falsifiability at construction and provider parsing boundaries.
    raw_judge_init = judges_mod.JudgeVerdict.__init__

    def judge_init(self, **data):
        data.setdefault("falsifiable_criteria", False)
        raw_judge_init(self, **data)

    judges_mod.JudgeVerdict.__init__ = judge_init

    raw_payload_parser = judges_mod.BaseLLMJudge._verdict_from_payload

    def verdict_from_payload(self, payload, latency_ms, *, prompt_tokens=0, completion_tokens=0):
        if isinstance(payload, dict) and "falsifiable_criteria" not in payload:
            payload = dict(payload)
            payload["falsifiable_criteria"] = False
        return raw_payload_parser(
            self,
            payload,
            latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    judges_mod.BaseLLMJudge._verdict_from_payload = verdict_from_payload

    # Rebuild legacy judge_ensemble from its invariants.  Legacy unbound votes
    # remain usable only when they do not claim another explicit round; an
    # explicitly versioned stale vote can never cross into the current plan.
    async def judge_ensemble(session, plan_text, objective, *, n=3, plans_dir=None, **judge_kwargs):
        pdir = (
            ns["Path"](plans_dir)
            if plans_dir is not None
            else (
                ns["Path"](session.get("plans_dir"))
                if isinstance(session, dict) and session.get("plans_dir")
                else ns["DEFAULT_PLANS_DIR"]
            )
        )
        state = ns["_load_session"](pdir, session) if isinstance(session, str) else session
        current_version = state.get("best_version") or len(state.get("rounds", []))
        rounds = list(state.get("rounds") or [])
        current_text = (
            str(rounds[current_version - 1].get("plan_text") or "")
            if isinstance(current_version, int) and 1 <= current_version <= len(rounds)
            else plan_text
        )
        current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()

        votes: list[dict[str, Any]] = []
        verify_res = ns["verify"](current_text)
        ground_res = ns["ground_check"](current_text)
        sim_res = ns["simulate"](
            current_text,
            initial_state=set(ground_res.get("verified", [])),
        )
        if verify_res.get("ok") and ground_res.get("ok") and sim_res.get("executable_plan"):
            votes.append({
                "ok": True,
                "verdict": "go",
                "feasibility_0_100": 100,
                "falsifiable_criteria": True,
                "source": "mechanical",
                "external": False,
                "round_version": current_version,
                "plan_hash": current_hash,
            })

        try:
            current_vote = await ns["judge"](
                current_text,
                objective,
                **judge_kwargs,
            )
            if (
                isinstance(current_vote, dict)
                and current_vote.get("ok")
                and current_vote.get("falsifiable_criteria")
            ):
                vote = dict(current_vote)
                vote.setdefault("round_version", current_version)
                vote.setdefault("plan_hash", current_hash)
                votes.append(vote)
        except Exception:
            pass

        for prior in reversed(list(state.get("judge_log", []))):
            if len(votes) >= max(1, n):
                break
            if not prior.get("ok") or not prior.get("falsifiable_criteria"):
                continue
            prior_version = prior.get("round_version")
            prior_hash = prior.get("plan_hash")
            if prior_version is not None and prior_version != current_version:
                continue
            if prior_hash is not None and prior_hash != current_hash:
                continue
            if prior not in votes:
                votes.append(dict(prior))

        votes = votes[: max(1, min(n, len(votes)))]
        if not votes:
            entry = {
                "ok": False,
                "verdict": "rework",
                "feasibility_0_100": 0,
                "falsifiable_criteria": False,
                "ensemble": True,
                "votes": [],
                "median_feasibility": 0,
                "source": "ensemble",
                "external": False,
            }
            ns["record_judge"](state, entry, round_version=current_version, plans_dir=pdir)
            return entry

        feasibilities = sorted(float(v.get("feasibility_0_100", 0) or 0) for v in votes)
        median = feasibilities[(len(feasibilities) - 1) // 2]
        representative = min(
            votes,
            key=lambda v: abs(float(v.get("feasibility_0_100", 0) or 0) - median),
        )
        entry = {
            **representative,
            "ensemble": True,
            "votes": votes,
            "median_feasibility": median,
            "verdict": representative.get("verdict", "rework"),
            "ok": True,
            # An ensemble is advisory aggregation, not itself an independently
            # attested provider response. Do not inherit external provenance.
            "source": "ensemble",
            "external": False,
            "round_version": current_version,
            "plan_hash": current_hash,
        }
        entry.pop("_judge_attestation", None)
        ns["record_judge"](state, entry, round_version=current_version, plans_dir=pdir)
        return entry

    ns["judge_ensemble"] = judge_ensemble
