"""Legacy DeepSeek external judge with explicit provider selection.

This module remains for backward compatibility.  It no longer silently chooses
DeepSeek or disables thinking when the caller omits a model.  Multi-provider
workflows should prefer ``plan_mode.judges`` adapters; this legacy entry point
requires an explicit DeepSeek model (or ``PLAN_JUDGE_MODEL``) and enforces one
total wall-clock budget across the initial request and optional JSON retry.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

_BASE = "https://api.deepseek.com/anthropic"
_DEFAULT_TIMEOUT = 30.0

JUDGE_SYSTEM = (
    "You are an adversarial plan reviewer. You evaluate whether a written plan "
    "will actually work if an autonomous agent executes it step by step. "
    "Be skeptical: prose claims are cheap, execution is hard. Check for "
    "internal contradictions, missing dependencies, unstated assumptions, "
    "impossible steps, unfalsifiable success criteria, and missing rollback. "
    "Do not grade style. Respond ONLY with a JSON object (no markdown fences) "
    "with exactly these keys:\n"
    '{"verdict": "go" | "rework" | "reject", '
    '"feasibility_0_100": int, '
    '"blockers": [list of concrete show-stopper flaws, empty if verdict is go], '
    '"contradictions": [list of mutually inconsistent claims], '
    '"missing": [list of things the plan needs but does not state], '
    '"falsifiable_criteria": true | false, '
    '"summary": "1-2 sentence verdict"}'
)


def _resolve_api_key() -> str:
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        auth = json.loads((Path.home() / ".prime" / "agent" / "auth.json").read_text())
        cred = auth.get("deepseek") if isinstance(auth, dict) else None
        if isinstance(cred, dict) and cred.get("type") == "api_key":
            value = str(cred.get("key") or "").strip()
            if value and not value.startswith("!"):
                return os.environ.get(value) or value
    except (OSError, ValueError):
        pass
    return ""


def _salvage_json(text: str) -> dict[str, Any] | None:
    for cut in range(len(text), 0, -1):
        candidate = text[:cut]
        if candidate.count("{") != candidate.count("}"):
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _anchor_verdict(parsed: dict[str, Any], mech: dict[str, Any]) -> dict[str, Any]:
    if (mech.get("verify_ok") is False or mech.get("ground_ok") is False or mech.get("sim_ok") is False) \
            and parsed.get("verdict") == "go":
        parsed["verdict"] = "rework"
        parsed.setdefault("blockers", []).append(
            "[grounding][mechanical] deterministic verification/grounding/simulation failed; verdict capped at rework"
        )
        try:
            parsed["feasibility_0_100"] = min(int(parsed.get("feasibility_0_100", 100)), 60)
        except (TypeError, ValueError):
            parsed["feasibility_0_100"] = 0
    return parsed


def _mechanical_summary(plan_text: str) -> tuple[str, dict[str, Any]]:
    try:
        from . import ground_check, simulate, verify
        verified = verify(plan_text)
        grounded = ground_check(plan_text)
        simulated = simulate(plan_text, initial_state=set(grounded.get("verified", [])))
        summary = (
            "MECHANICAL CHECKS (ground truth, do not contradict):\n"
            f"- verify(): ok={verified.get('ok')} errors={verified.get('errors', [])[:5]}\n"
            f"- ground_check(): ok={grounded.get('ok')} missing={grounded.get('missing', [])[:5]}\n"
            f"- simulate(): executable={simulated.get('executable_plan')} "
            f"problems={simulated.get('problems', [])[:5]}"
        )
        return summary, {
            "verify_ok": bool(verified.get("ok")),
            "ground_ok": bool(grounded.get("ok")),
            "sim_ok": bool(simulated.get("executable_plan")),
        }
    except Exception as exc:
        return f"MECHANICAL CHECKS unavailable: {exc}", {}


def _extract_text(data: dict[str, Any]) -> str:
    text = ""
    for block in data.get("content", []) if isinstance(data, dict) else []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += str(block.get("text") or "")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    start, end = text.find("{") , text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return text


async def judge(
    plan_text: str,
    objective: str = "",
    *,
    model: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return an explicitly requested legacy DeepSeek feasibility verdict.

    A bare call does not choose a provider.  Set ``model='deepseek-...'`` or
    ``PLAN_JUDGE_MODEL`` deliberately.  Provider/model failures return
    ``ok=False`` and never become a gate-passing vote.
    """
    selected_model = (model or os.environ.get("PLAN_JUDGE_MODEL", "")).strip()
    if not selected_model:
        return {
            "ok": False,
            "error": (
                "no external judge model selected; pass model= explicitly or set PLAN_JUDGE_MODEL. "
                "Prime will not silently choose DeepSeek"
            ),
        }
    if not selected_model.lower().startswith("deepseek"):
        return {
            "ok": False,
            "error": (
                f"legacy judge_client only supports explicit DeepSeek models, got {selected_model!r}; "
                "use plan_mode.judges adapters for other providers"
            ),
        }
    if timeout <= 0:
        return {"ok": False, "error": "judge timeout must be > 0"}

    key = api_key or _resolve_api_key()
    if not key:
        return {"ok": False, "error": "no DeepSeek API key configured"}

    mech_text, mech = _mechanical_summary(plan_text)
    user_msg = (
        f"Objective: {objective}\n\n{mech_text}\n\nPlan:\n{plan_text}"
        if objective else f"{mech_text}\n\nPlan:\n{plan_text}"
    )
    body = {
        "model": selected_model,
        "max_tokens": 4096,
        # No verifier-specific thinking override: use the explicitly selected
        # model/provider default unless a future provider adapter says otherwise.
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async def _run() -> dict[str, Any]:
        per_request = max(1.0, float(timeout) / 2.0)
        async with httpx.AsyncClient(timeout=per_request) as client:
            response = await client.post(f"{_BASE}/v1/messages", json=body, headers=headers)
            if response.status_code != 200:
                return {"ok": False, "error": f"judge HTTP {response.status_code}: {response.text[:200]}"}
            data = response.json()
            text = _extract_text(data)
            if not text:
                return {"ok": False, "error": f"judge returned no text (stop_reason={data.get('stop_reason')})"}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = _salvage_json(text)

            if parsed is None:
                retry_body = {
                    **body,
                    "messages": body["messages"] + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "Return ONLY the JSON object, no prose."},
                    ],
                }
                retry = await client.post(f"{_BASE}/v1/messages", json=retry_body, headers=headers)
                if retry.status_code == 200:
                    retry_text = _extract_text(retry.json())
                    try:
                        parsed = json.loads(retry_text) if retry_text else None
                    except json.JSONDecodeError:
                        parsed = _salvage_json(retry_text)

            if not isinstance(parsed, dict):
                return {"ok": False, "error": f"judge JSON unparseable: {text[:200]}", "raw": text}
            parsed = _anchor_verdict(parsed, mech)
            parsed["ok"] = True
            parsed["mechanical"] = mech
            parsed["source"] = "external_llm"
            parsed["external"] = True
            parsed["provider"] = "deepseek"
            parsed["model"] = selected_model
            return parsed

    try:
        return await asyncio.wait_for(_run(), timeout=float(timeout))
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"judge total timeout exceeded after {float(timeout):.1f}s"}
    except Exception as exc:
        return {"ok": False, "error": f"judge failed: {exc}"}
