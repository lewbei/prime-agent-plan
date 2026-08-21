"""Engine regression suite with the PR #5 external-provenance expectation.

The pre-hardening suite is preserved byte-for-byte in
``_engine_suite_legacy.py`` for auditability. It is executed here so every
legacy test remains collected; only the one assertion that explicitly treated
caller-written metadata as a genuine external judge is redefined below to
match the strengthened truth boundary.
"""
from pathlib import Path

_legacy_path = Path(__file__).with_name("_engine_suite_legacy.py")
exec(compile(_legacy_path.read_text(encoding="utf-8"), str(_legacy_path), "exec"), globals(), globals())


def test_strict_external_judge_requires_explicit_external_llm(tmp_path):
    """Caller metadata cannot self-attest an external LLM judgment."""
    s = plan_mode.start("Strict external judge test", plans_dir=tmp_path)
    plan_mode.assess(s, "1. Setup\nOutput: a.txt", plans_dir=tmp_path)
    s["status"] = "converged"
    s["best_score"] = 95.0

    plan_mode.record_judge(
        s,
        {"ok": True, "verdict": "go", "falsifiable_criteria": True},
        round_version=1,
        plans_dir=tmp_path,
    )
    gate1 = plan_mode.release(
        s,
        min_score=90.0,
        require_judge=True,
        require_external_judge=True,
        plans_dir=tmp_path,
    )
    assert gate1["ok"] is False

    # Merely writing the old provenance labels is still untrusted. Only the
    # provider-calling judge path can create Prime's internal attestation.
    plan_mode.record_judge(
        s,
        {
            "ok": True,
            "verdict": "go",
            "falsifiable_criteria": True,
            "source": "external_llm",
            "external": True,
        },
        round_version=1,
        plans_dir=tmp_path,
    )
    gate2 = plan_mode.release(
        s,
        min_score=90.0,
        require_judge=True,
        require_external_judge=True,
        plans_dir=tmp_path,
    )
    assert gate2["ok"] is False


del _legacy_path
