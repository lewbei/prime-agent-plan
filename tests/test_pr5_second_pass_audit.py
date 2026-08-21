"""Second-pass regressions found after the first PR #5 GREEN matrix."""
from __future__ import annotations

import asyncio
import time

import pytest

import plan_mode
from plan_mode.execution_contract import ExecutionContract, parity_audit, symbol_audit
from plan_mode.ir import PlanIR
from plan_mode.judges import OpenAIJudge
from plan_mode.registry import CapabilityRegistry
from plan_mode.self_verification import ProbabilisticSelfVerifier, SelfVerificationUnavailableError
from plan_mode.session import AuthorizationCertificate


def _contract_plan_with_symbol(path: str) -> str:
    return f'''# Goal
Goal: audit symbols.

## Tasks
1. Audit. Output: out.txt.

## Execution Contract
```json
{{
  "symbols": {{
    {path!r}: {{"functions": ["secret_function"]}}
  }}
}}
```
'''.replace("'", '"')


def test_symbol_audit_rejects_absolute_path_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("def secret_function():\n    return 1\n")

    result = symbol_audit(_contract_plan_with_symbol(str(outside)), cwd=workspace)
    assert result["ok"] is False
    assert any("workspace" in error.lower() or "outside" in error.lower() for error in result["errors"])


def test_parity_audit_rejects_paths_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_a = tmp_path / "a.bin"
    outside_b = tmp_path / "b.bin"
    outside_a.write_bytes(b"same")
    outside_b.write_bytes(b"same")
    contract = ExecutionContract(parity_checks=[{
        "left": str(outside_a),
        "right": str(outside_b),
        "algorithm": "sha256",
    }])

    result = parity_audit(contract, cwd=workspace)
    assert result["ok"] is False
    assert any("workspace" in error.lower() or "outside" in error.lower() for error in result["errors"])


def test_authorization_signature_binds_certificate_and_version_identity():
    plan = PlanIR(plan_id="auth-plan", goal_description="bind cert", version=1)
    cert = AuthorizationCertificate.create(
        plan,
        [],
        CapabilityRegistry(),
        policy_hash="policy",
        isolation_policy_hash="isolation",
        secret_key=b"k" * 32,
        ttl_seconds=60,
    )
    assert cert.verify_signature(b"k" * 32) is True
    for update in (
        {"certificate_id": "tampered"},
        {"plan_id": "another-plan"},
        {"plan_version": 999},
        {"issued_at": cert.issued_at - 100},
    ):
        assert cert.model_copy(update=update).verify_signature(b"k" * 32) is False


@pytest.mark.asyncio
async def test_speculative_rollout_async_has_outer_timeout():
    async def hangs(ctx):
        await asyncio.sleep(0.5)
        return 1.0

    started = time.monotonic()
    result = await plan_mode.speculative_rollout_async(
        "1. Evaluate. Output: x.txt.\n",
        hangs,
        timeout_seconds=0.02,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert result["ok"] is False
    assert "timeout" in result["error"].lower()


class _SlowHTTPClient:
    async def post(self, *args, **kwargs):
        await asyncio.sleep(0.5)
        raise AssertionError("outer timeout should cancel this request")


@pytest.mark.asyncio
async def test_direct_llm_judge_has_outer_timeout_even_with_custom_client():
    judge = OpenAIJudge(model="test", http_client=_SlowHTTPClient())
    started = time.monotonic()
    verdict = await judge.evaluate(
        PlanIR(plan_id="judge-timeout", goal_description="timeout"),
        timeout=0.02,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.2
    assert verdict.verdict == "UNKNOWN"
    assert "timeout" in " ".join(verdict.blockers).lower()


def test_custom_self_verifier_select_fn_is_wall_clock_bounded():
    def slow_select(**kwargs):
        time.sleep(0.5)
        raise AssertionError("caller must already have timed out")

    verifier = ProbabilisticSelfVerifier(select_fn=slow_select)
    started = time.monotonic()
    with pytest.raises(SelfVerificationUnavailableError, match="timeout"):
        verifier.select(
            problem="choose",
            candidates=["a", "b"],
            model="model-x",
            thinking_profile="default",
            request_timeout_seconds=0.02,
            max_verifier_calls=128,
        )
    assert time.monotonic() - started < 0.2
