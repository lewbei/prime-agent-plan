"""LLM judge for plan_mode: independent feasibility verdict via DeepSeek.

The regex rubric scores whether a plan *says* the right things. This judge
asks an external model whether the plan would *work*: internal
contradictions, unstated blockers, missing evidence, and an overall
go/rework/reject verdict. This is the non-circular external checker the
literature demands (2510.03469, 2603.06064: self-assessed progress is
unreliable; use an independent evaluator).

Key resolution mirrors the deepseek-search skill: DEEPSEEK_API_KEY env var or
~/.prime/agent/auth.json (deepseek provider, saved via /login).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

_BASE = "https://api.deepseek.com/anthropic"
_DEFAULT_MODEL = "deepseek-v4-flash"
_DEFAULT_TIMEOUT = 120

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
            v = str(cred.get("key") or "").strip()
            if v and not v.startswith("!"):
                return os.environ.get(v) or v
    except (OSError, ValueError):
        pass
    return ""


def _salvage_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a truncated JSON object."""
    for cut in range(len(text), 0, -1):
        t = text[:cut]
        depth = t.count("{") - t.count("}")
        if depth != 0:
            continue
        try:
            v = json.loads(t)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


def _anchor_verdict(parsed: dict[str, Any], mech: dict[str, Any]) -> dict[str, Any]:
    """Mechanical anchoring (2406.04520): a structurally broken plan can never
    be judged "go", whatever the model says."""
    if mech.get("verify_ok") is False and parsed.get("verdict") == "go":
        parsed["verdict"] = "rework"
        parsed.setdefault("blockers", []).append(
            "[grounding] verify() failed; verdict capped at rework")
        parsed["feasibility_0_100"] = min(int(parsed.get("feasibility_0_100", 100)), 60)
    return parsed


def _mechanical_summary(plan_text: str) -> tuple[str, dict[str, Any]]:
    """Ground-truth check summary (2505.01479): verify() + simulate() results,
    computed lazily to avoid a circular import at module load time."""
    try:
        from . import verify, simulate
        v = verify(plan_text)
        s = simulate(plan_text)
        summary = ("MECHANICAL CHECKS (ground truth, do not contradict):\n"
                   f"- verify(): ok={v.get('ok')} errors={v.get('errors', [])[:5]}\n"
                   f"- simulate(): executable={s.get('executable_plan')} "
                   f"problems={s.get('problems', [])[:5]}")
        return summary, {"verify_ok": v.get("ok"),
                         "sim_ok": s.get("executable_plan")}
    except Exception as e:
        return (f"MECHANICAL CHECKS unavailable: {e}"), {}


async def judge(plan_text: str, objective: str = "", *,
                model: str | None = None,
                timeout: int = _DEFAULT_TIMEOUT,
                api_key: str | None = None) -> dict[str, Any]:
    """Return an external feasibility verdict for a plan.

    The judge is grounded (2406.04520, 2606.27757): it receives the
    verify()+simulate() ground truth and its verdict is mechanically
    anchored — a plan whose verify() fails cannot be judged "go".

    Returns {"ok": bool, "verdict", "feasibility_0_100", "blockers",
    "contradictions", "missing", "falsifiable_criteria", "summary", "error",
    "mechanical": {...}}.
    ok=False with an "error" key means the judge itself could not run.
    """
    key = api_key or _resolve_api_key()
    if not key:
        return {"ok": False, "error": "no DeepSeek API key configured (run /login, provider deepseek)"}
    model = model or os.environ.get("PLAN_JUDGE_MODEL", "").strip() or _DEFAULT_MODEL
    mech_text, mech = _mechanical_summary(plan_text)
    user_msg = (f"Objective: {objective}\n\n{mech_text}\n\nPlan:\n{plan_text}"
                if objective else f"{mech_text}\n\nPlan:\n{plan_text}")
    body = {
        "model": model,
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{_BASE}/v1/messages", json=body, headers=headers)
        if resp.status_code != 200:
            return {"ok": False, "error": f"judge HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        text = text.strip()
        if not text:
            return {"ok": False, "error": f"judge returned no text (stop_reason={data.get('stop_reason')})"}
        # tolerate markdown fences and leading chatter
        if text.startswith("```"):
            text = text.strip("`")
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # truncated JSON: salvage the valid prefix of the object
            parsed = _salvage_json(text)
        if parsed is None:
            # one JSON-only retry before giving up (robustness)
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp2 = await client.post(
                        f"{_BASE}/v1/messages",
                        json={**body, "messages": body["messages"] + [
                            {"role": "assistant", "content": text},
                            {"role": "user",
                             "content": "Return ONLY the JSON object, no prose."}]},
                        headers=headers)
                text2 = ""
                for block in resp2.json().get("content", []):
                    if block.get("type") == "text":
                        text2 += block.get("text", "")
                start2 = text2.find("{"); end2 = text2.rfind("}")
                parsed = json.loads(text2[start2:end2 + 1]) if end2 > start2 else None
            except Exception:
                parsed = None
        if parsed is None:
            return {"ok": False, "error": f"judge JSON unparseable: {text[:200]}", "raw": text}
        # mechanical anchoring (2406.04520): a structurally broken plan can
        # never be "go", whatever the model says
        parsed = _anchor_verdict(parsed, mech)
        parsed["ok"] = True
        parsed["mechanical"] = mech
        parsed["source"] = "external_llm"
        parsed["external"] = True
        return parsed
    except Exception as e:  # network/parse errors must not crash plan mode
        return {"ok": False, "error": f"judge failed: {e}"}
