"""Engine test facade with the strict-external provenance regression updated.

The historical suite is preserved exactly in ``_test_engine_impl.py`` and
executed into this module. The one superseded test is then replaced so a
"genuine external" verdict must carry the same runtime-issued attestation that
production release now requires; caller-written source/external booleans are no
longer sufficient evidence.
"""
from pathlib import Path as _BootstrapPath

_impl_path = _BootstrapPath(__file__).with_name("_test_engine_impl.py")
_impl_source = _impl_path.read_text(encoding="utf-8")
exec(compile(_impl_source, str(_impl_path), "exec"), globals(), globals())
del _impl_source


def test_strict_external_judge_requires_explicit_external_llm(tmp_path):
    """Strict external release accepts only a runtime-attested LLM verdict."""
    from plan_mode.api_hardening import _attest_external

    s = plan_mode.start("Strict external judge test", plans_dir=tmp_path)
    plan_mode.assess(s, "1. Setup\nOutput: a.txt", plans_dir=tmp_path)
    s["status"] = "converged"
    s["best_score"] = 95.0

    # Untagged / caller-authored verdict cannot establish external provenance.
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

    # Simulate the trusted return path of plan_mode.judge(): the provider/model
    # identity is present and the process-local HMAC attestation is attached
    # before record_judge persists the verdict.
    attested = _attest_external({
        "ok": True,
        "verdict": "go",
        "falsifiable_criteria": True,
        "source": "external_llm",
        "external": True,
        "provider": "test-provider",
        "model": "test-model",
    })
    plan_mode.record_judge(s, attested, round_version=1, plans_dir=tmp_path)
    gate2 = plan_mode.release(
        s,
        min_score=90.0,
        require_judge=True,
        require_external_judge=True,
        plans_dir=tmp_path,
    )
    assert gate2["ok"] is True
