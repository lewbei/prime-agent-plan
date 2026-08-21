"""Low-level integrity regressions salvaged from superseded PR #6."""
from __future__ import annotations

import os

import pytest

from plan_mode.cordis import Context
from plan_mode.fact_identity import canonical_typed_arg_token
from plan_mode.memory_distiller import RoTRule, RoTRuleBase
from plan_mode.runtime.secret_scrubber import SecretScrubber


def test_rot_memory_save_is_atomic_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "rot.json"
    path.write_text("{}", encoding="utf-8")
    rb = RoTRuleBase(storage_path=path)
    rb.rules["r"] = RoTRule(
        rule_id="r",
        trigger_condition="x",
        forbidden_pattern="y",
        remedy="z",
        source_flaw_type="test",
    )

    def fail_replace(src, dst):
        raise OSError("simulated crash before atomic replace")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        rb._save()
    assert path.read_text(encoding="utf-8") == "{}"


def test_rot_memory_corruption_is_not_silently_ignored(tmp_path):
    path = tmp_path / "rot.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt|invalid"):
        RoTRuleBase(storage_path=path)


def test_secret_scrubber_redacts_bare_provider_tokens():
    scrubber = SecretScrubber()
    text = (
        "openai=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789 "
        "google=AIzaSyDUMMYDUMMYDUMMYDUMMYDUMMYDUMMY12"
    )
    scrubbed = scrubber.scrub_text(text)
    assert "sk-proj-" not in scrubbed
    assert "AIza" not in scrubbed
    assert "REDACTED_SECRET" in scrubbed


def test_fact_identity_rejects_nondeterministic_arbitrary_object_repr():
    class Unstable:
        pass

    with pytest.raises(TypeError, match="deterministic|unsupported"):
        canonical_typed_arg_token(Unstable())


@pytest.mark.asyncio
async def test_sync_cordis_dispose_refuses_to_fake_completion_in_running_loop():
    ctx = Context(name="sync-dispose-async-inverse")
    trace = []

    async def undo():
        trace.append("undone")

    ctx.effect(lambda: undo)
    with pytest.raises(RuntimeError, match="async_dispose"):
        ctx.dispose()
    assert trace == []

    await ctx.async_dispose()
    assert trace == ["undone"]
