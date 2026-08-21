"""Evidence, memory, and symbolic-validation closure."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, MutableMapping


def install_execution_trace_closure(ns: MutableMapping[str, Any]) -> None:
    from . import execution_trace as trace

    raw_verify = trace.verify_execution_trace
    if getattr(raw_verify, "_runtime_closure", False):
        return

    def verify_execution_trace(
        plan_text: str,
        evidence: Any,
        *,
        cwd: str | Path | None = None,
        require_independent_verifier: bool | None = None,
        require_bound_identity: bool = False,
        expected_session_id: str | None = None,
        expected_certificate_id: str | None = None,
        expected_workspace_identity: str | None = None,
    ) -> dict[str, Any]:
        result = raw_verify(
            plan_text,
            evidence,
            cwd=cwd,
            require_independent_verifier=require_independent_verifier,
        )
        errors = list(result.get("errors", []))
        parsed = result.get("evidence")
        raw = parsed.raw if parsed is not None and isinstance(parsed.raw, dict) else {}
        raw_tasks = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
        seen: set[int] = set()
        for index, item in enumerate(raw_tasks):
            if not isinstance(item, dict):
                errors.append(f"execution evidence task entry {index} is not an object")
                continue
            if "status" not in item:
                errors.append(f"execution evidence task entry {index} has no explicit status")
            try:
                task_id = int(item.get("task_id"))
            except (TypeError, ValueError):
                errors.append(f"execution evidence task entry {index} has invalid task_id")
                continue
            if task_id <= 0:
                errors.append(f"execution evidence task entry {index} has non-positive task_id")
            if task_id in seen:
                errors.append(f"execution evidence contains duplicate task_id {task_id}")
            seen.add(task_id)

        expected_plan_hash = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
        supplied_plan_hash = raw.get("plan_hash")
        if supplied_plan_hash is not None and supplied_plan_hash != expected_plan_hash:
            errors.append("execution evidence plan_hash does not match the verified plan")

        if require_bound_identity:
            for key in ("plan_hash", "session_id", "workspace_identity"):
                if not raw.get(key):
                    errors.append(f"execution evidence is missing bound identity field {key!r}")
            if raw.get("plan_hash") and raw.get("plan_hash") != expected_plan_hash:
                errors.append("execution evidence is bound to another plan")
            if expected_session_id and raw.get("session_id") != expected_session_id:
                errors.append("execution evidence is bound to another session")
            if expected_certificate_id and raw.get("certificate_id") != expected_certificate_id:
                errors.append("execution evidence is bound to another authorization certificate")
            if expected_workspace_identity and raw.get("workspace_identity") != expected_workspace_identity:
                errors.append("execution evidence is bound to another workspace")

        result["errors"] = list(dict.fromkeys(errors))
        result["ok"] = not result["errors"]
        return result

    verify_execution_trace._runtime_closure = True  # type: ignore[attr-defined]
    trace.verify_execution_trace = verify_execution_trace
    ns["verify_execution_trace"] = verify_execution_trace


def install_memory_closure() -> None:
    from .memory_distiller import ContextBudgeter

    @classmethod
    def compress_history(
        cls: Any,
        session: dict[str, Any],
        max_context_tokens: int = 4000,
        keep_last: int = 2,
    ) -> dict[str, Any]:
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        rounds = session.get("rounds", [])
        if not isinstance(rounds, list):
            raise ValueError("session rounds must be a list")
        if len(rounds) <= 2:
            session["context_budget_tokens"] = cls.session_token_count(session)
            session["context_budget_exceeded"] = (
                session["context_budget_tokens"] > max_context_tokens
            )
            return session

        protected = {
            session.get("best_version"),
            session.get("committed_version"),
            rounds[-1].get("version") if rounds else None,
        }
        recent_indices = set(range(max(0, len(rounds) - keep_last), len(rounds)))

        def fold(index: int, round_data: dict[str, Any]) -> None:
            version = round_data.get("version", index + 1)
            critiques = round_data.get("critiques", [])
            critique_ids = ", ".join(
                (
                    item.get("id", str(item))
                    if isinstance(item, dict)
                    else str(item)
                )
                for item in critiques[:3]
            ) or "none"
            summary = (
                f"[folded: version {version}, score {round_data.get('score')}, "
                f"delta {round_data.get('delta')}, critiques {critique_ids}]"
            )
            round_data["folded"] = True
            round_data["summary"] = summary
            round_data["plan_text"] = summary

        for index, round_data in enumerate(rounds):
            version = round_data.get("version", index + 1)
            if version in protected or index in recent_indices or round_data.get("folded"):
                continue
            fold(index, round_data)

        for index, round_data in enumerate(rounds):
            if cls.session_token_count(session) <= max_context_tokens:
                break
            version = round_data.get("version", index + 1)
            if version in protected or round_data.get("folded"):
                continue
            fold(index, round_data)

        remaining = cls.session_token_count(session)
        session["context_budget_tokens"] = remaining
        session["context_budget_exceeded"] = remaining > max_context_tokens
        return session

    ContextBudgeter.compress_history = compress_history


def install_causal_closure() -> None:
    from .causal_validator import CausalValidator, Proposition

    raw_validate = CausalValidator.validate
    if getattr(raw_validate, "_runtime_closure", False):
        return

    @classmethod
    def validate(cls: Any, ast: Any, initial_state: set[str] | None = None) -> dict[str, Any]:
        source = set(initial_state if initial_state is not None else ast.initial_state)
        sanitized: set[str] = set()
        for raw in source:
            text = str(raw).strip()
            if Proposition.parse(text).negated:
                continue
            sanitized.add(text)

        result = raw_validate(ast, initial_state=sanitized)
        flaws = list(result.get("flaws", []))
        for unit, constraints in ast.constraints.items():
            for constraint in constraints:
                kind = str(constraint.get("type") or "").lower()
                if "at least" not in kind and "minimum of" not in kind:
                    continue
                limit = float(constraint.get("value", 0))
                if "task" in unit or "step" in unit:
                    actual = float(result.get("num_actions", 0))
                elif "min" in unit:
                    actual = float(result.get("total_duration_minutes", 0.0))
                elif "hour" in unit:
                    actual = float(result.get("total_duration_minutes", 0.0)) / 60.0
                elif "token" in unit or "$" in unit or "usd" in unit:
                    actual = float(result.get("total_cost", 0.0))
                else:
                    continue
                if actual < limit:
                    flaws.append({
                        "type": "resource_minimum_not_met",
                        "task_id": 0,
                        "detail": (
                            f"Plan violates constraint: {kind} {limit:g} {unit} "
                            f"(actual: {actual:g} {unit})"
                        ),
                        "remedy": f"Increase the relevant resource above {limit:g} {unit}",
                        "involved_tasks": [],
                    })
        result = dict(result)
        result["flaws"] = flaws
        result["ok"] = not flaws
        return result

    validate._runtime_closure = True  # type: ignore[attr-defined]
    CausalValidator.validate = validate
