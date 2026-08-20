"""Regression tests for hardened public ``plan`` behavior."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import plan
import plan_mode


def test_public_run_uses_hardened_assess_status(tmp_path):
    session = plan.run(
        "dirty run wrapper",
        "# Goal\nGoal: x.\n\n## Tasks\n1. Maybe do it. Output: out.txt.\n",
        plans_dir=tmp_path,
        max_rounds=1,
    )
    assert session["status"] == "plateaued"


def test_public_execute_plan_sync_rejects_missing_handlers():
    text = "1. Build. Output: a.txt.\n"
    result = plan.execute_plan_sync(text, task_handlers={})
    assert result["ok"] is False
    assert "missing task handlers" in result["error"]


def test_public_release_rechecks_exact_execution_cwd(monkeypatch, tmp_path):
    seen = []

    monkeypatch.setattr(
        plan,
        "_raw_release",
        lambda *args, **kwargs: {"ok": True, "checks": [], "problems": []},
    )
    monkeypatch.setattr(
        plan,
        "_best_plan_text",
        lambda *args, **kwargs: "1. Do. Inputs: input.txt. Output: out.txt.\n",
    )

    def fake_ground_check(text, *, cwd=None):
        seen.append(PathLike(cwd))
        return {"ok": False, "missing": ["input.txt (task 1)"], "verified": []}

    monkeypatch.setattr(plan_mode, "ground_check", fake_ground_check)
    monkeypatch.setattr(
        plan_mode,
        "simulate",
        lambda *args, **kwargs: {"executable_plan": False, "problems": ["missing input"]},
    )

    gate = plan.release({}, execution_cwd=tmp_path, require_judge=False)
    assert gate["ok"] is False
    assert gate["execution_cwd_checks"]["cwd"] == str(tmp_path.resolve())
    assert seen == [str(tmp_path.resolve())]


def PathLike(value):
    return str(value.resolve()) if hasattr(value, "resolve") else str(value)


def test_local_search_mutations_are_stable_across_hash_seeds(tmp_path):
    code = (
        "import json, plan, plan_mode.search_engine as se; "
        "p='# Goal\\nGoal: stable.\\n## Tasks\\n1. A. Output: a.txt.'; "
        "print(json.dumps([x['note'] for x in se._mutations(p, 3)]))"
    )
    outputs = []
    for seed in ("1", "987654"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout.strip()))
    assert outputs[0] == outputs[1]


def test_importing_plan_does_not_replace_unrelated_plan_mode_entrypoints():
    # assess_candidates is the one intentional alias. Other core functions stay
    # owned by plan_mode, avoiding import-order-dependent test/runtime behavior.
    assert plan_mode.assess is not plan.assess
    assert plan_mode.release is not plan.release
    assert plan_mode.finish is not plan.finish
    assert plan_mode.execute_plan is not plan.execute_plan
    assert plan_mode.assess_candidates is plan.assess_candidates
