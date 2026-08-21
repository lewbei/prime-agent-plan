"""Judge-log tamper regression salvaged from superseded PR #6."""
from __future__ import annotations

import plan_mode


def test_strict_release_revalidates_judge_attestation_after_direct_log_tamper(tmp_path):
    session = plan_mode.start("judge-log-tamper", plans_dir=tmp_path, max_rounds=1)
    plan_mode.assess(session, "1. Setup. Output: a.txt.\n", plans_dir=tmp_path)
    session["status"] = "converged"
    session["best_score"] = 95.0
    session.setdefault("judge_log", []).append({
        "ok": True,
        "verdict": "go",
        "falsifiable_criteria": True,
        "source": "external_llm",
        "external": True,
        "external_attested": True,
        "round_version": session.get("best_version"),
        "plan_hash": "forged",
        "_judge_attestation": {
            "plan_hash": "forged",
            "provider": "fake",
            "model": "fake",
            "response_digest": "fake",
            "issued_at": 0.0,
            "signature": "fake",
        },
    })

    gate = plan_mode.release(
        session,
        min_score=0,
        require_judge=True,
        require_external_judge=True,
        plans_dir=tmp_path,
    )
    assert gate["ok"] is False
    assert session.get("committed_version") is None
