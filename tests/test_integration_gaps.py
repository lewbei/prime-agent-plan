"""Pin the integration gaps closed after the formal-core commit.

The formal modules existed but were not all wired into the main planning loop.
These tests guard the wiring itself.
"""
from __future__ import annotations

import plan_mode


def test_verify_seeds_relative_grounded_inputs_for_causal_validation(tmp_path):
    (tmp_path / "config.yaml").write_text("x")
    plan = (
        "# Goal\nGoal: x.\n\n"
        "## Tasks\n"
        "1. Read. Inputs: config.yaml. Output: a.md.\n"
        "2. Use. Depends on 1. Inputs: a.md. Output: b.md.\n"
    )
    result = plan_mode.verify(plan, cwd=tmp_path)
    assert result["ok"], result["errors"]
    assert result["causal_validation"]["ok"]


def test_assess_distills_and_reports_rot_rules(tmp_path):
    plan = (
        "# Goal\nGoal: x.\n\n"
        "## Tasks\n"
        "1. Broken. Inputs: missing_input_zz.md. Output: a.md.\n"
    )
    session = plan_mode.start("rot wiring", plans_dir=tmp_path)
    result = plan_mode.assess(session, plan, plans_dir=tmp_path)
    assert result["rot_rules"]["learned"] >= 1
    assert (tmp_path / f"{session['session_id']}.rot.json").exists()


def test_log_progress_uses_replanning_ladder(tmp_path):
    plan = (
        "# Goal\nGoal: x.\n\n"
        "## Tasks\n"
        "1. A. Output: a.md.\n"
        "2. B. Depends on 1. Output: b.md.\n"
        "3. C. Depends on 2. Output: c.md.\n"
    )
    session = plan_mode.start("replan wiring", plans_dir=tmp_path)
    plan_mode.assess(session, plan, plans_dir=tmp_path)
    plan_mode.log_progress(session, "task-2", status="failed", evidence="HTTP Timeout 504")
    assert session["replan_scope"]["tier"] == 1
    result = plan_mode.assess(session, plan, plans_dir=tmp_path)
    assert any("Tier 1 Replan" in c["hint"] for c in result["critiques"])
