"""Second-pass hardening for path containment, certificate identity and callback bounds."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Any


def patch(ns: dict[str, Any]) -> None:
    from . import execution_contract as contract_mod
    from . import execution_trace as trace_mod
    from . import judges as judges_mod
    from . import self_verification as sv_mod
    from . import session as session_mod
    from .session import AuthorizationCertificate

    # ------------------------------------------------------------------
    # All plan-declared read paths must remain under the requested workspace.
    # ------------------------------------------------------------------
    def safe_path(raw: str, cwd: str | Path | None) -> Path:
        root = Path(cwd or Path.cwd()).resolve()
        candidate = Path(raw)
        candidate = candidate if candidate.is_absolute() else root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"path '{raw}' resolves outside workspace '{root}'"
            ) from exc
        return resolved

    raw_scan_symbols = contract_mod.scan_symbols
    raw_symbol_audit = contract_mod.symbol_audit
    raw_parity_audit = contract_mod.parity_audit

    def scan_symbols(paths, *, cwd=None):
        root = Path(cwd or Path.cwd()).resolve()
        out: dict[str, dict[str, Any]] = {}
        safe: list[str] = []
        for raw in paths:
            try:
                safe_path(str(raw), root)
            except ValueError as exc:
                out[str(raw)] = {
                    "functions": [],
                    "classes": [],
                    "variables": [],
                    "outside_workspace": True,
                    "error": str(exc),
                }
            else:
                safe.append(str(raw))
        if safe:
            out.update(raw_scan_symbols(safe, cwd=root))
        return out

    def symbol_audit(plan_text: str, *, cwd=None):
        contract, parse_errors = contract_mod.parse_execution_contract(plan_text)
        if contract is None:
            return {
                "ok": False,
                "errors": parse_errors or ["execution contract missing"],
                "files": {},
            }
        root = Path(cwd or Path.cwd()).resolve()
        path_errors: list[str] = []
        for raw in contract.symbols:
            try:
                safe_path(str(raw), root)
            except ValueError as exc:
                path_errors.append(str(exc))
        if path_errors:
            return {
                "ok": False,
                "errors": path_errors,
                "files": {
                    str(raw): {"outside_workspace": True}
                    for raw in contract.symbols
                    if any(str(raw) in error for error in path_errors)
                },
                "contract": contract,
            }
        return raw_symbol_audit(plan_text, cwd=root)

    def parity_audit(contract, *, cwd=None):
        root = Path(cwd or Path.cwd()).resolve()
        errors: list[str] = []
        for check in contract.parity_checks:
            for side in ("left", "right"):
                raw = str(check.get(side, ""))
                try:
                    safe_path(raw, root)
                except ValueError as exc:
                    errors.append(str(exc))
        if errors:
            return {"ok": False, "errors": errors, "results": []}
        return raw_parity_audit(contract, cwd=root)

    contract_mod.scan_symbols = scan_symbols
    contract_mod.symbol_audit = symbol_audit
    contract_mod.parity_audit = parity_audit
    trace_mod.symbol_audit = symbol_audit
    trace_mod.parity_audit = parity_audit
    ns["scan_symbols"] = scan_symbols
    ns["symbol_audit"] = symbol_audit
    ns["parity_audit"] = parity_audit

    # ------------------------------------------------------------------
    # Authorization signature binds the full certificate identity, not only
    # selected hashes. Older certificates fail closed and require reauth.
    # ------------------------------------------------------------------
    def cert_payload(
        certificate_id: str,
        plan_id: str,
        plan_version: int,
        plan_hash: str,
        world_state_hash: str,
        registry_hash: str,
        policy_hash: str,
        isolation_policy_hash: str,
        issued_at: float,
        expires_at: float,
    ) -> bytes:
        payload = {
            "certificate_id": certificate_id,
            "plan_id": plan_id,
            "plan_version": int(plan_version),
            "plan_hash": plan_hash,
            "world_state_hash": world_state_hash,
            "registry_hash": registry_hash,
            "policy_hash": policy_hash,
            "isolation_policy_hash": isolation_policy_hash,
            "issued_at": f"{float(issued_at):.6f}",
            "expires_at": f"{float(expires_at):.6f}",
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def create_certificate(
        cls,
        plan_ir,
        world_facts,
        registry,
        policy_hash,
        isolation_policy_hash,
        secret_key,
        ttl_seconds=60.0,
    ):
        now = time.time()
        expires_at = now + float(ttl_seconds)
        cert_id = f"cert_{secrets.token_hex(8)}"
        plan_hash = plan_ir.compute_hash()
        ws_hash = session_mod.compute_world_state_hash(world_facts)
        reg_hash = registry.compute_registry_hash()
        signature = hmac.new(
            secret_key,
            cert_payload(
                cert_id,
                plan_ir.plan_id,
                plan_ir.version,
                plan_hash,
                ws_hash,
                reg_hash,
                policy_hash,
                isolation_policy_hash,
                now,
                expires_at,
            ),
            hashlib.sha256,
        ).hexdigest()
        return cls(
            certificate_id=cert_id,
            plan_id=plan_ir.plan_id,
            plan_version=plan_ir.version,
            plan_hash=plan_hash,
            world_state_hash=ws_hash,
            registry_hash=reg_hash,
            policy_hash=policy_hash,
            isolation_policy_hash=isolation_policy_hash,
            issued_at=now,
            expires_at=expires_at,
            signature_hmac=signature,
        )

    def verify_certificate(self, secret_key: bytes) -> bool:
        expected = hmac.new(
            secret_key,
            cert_payload(
                self.certificate_id,
                self.plan_id,
                self.plan_version,
                self.plan_hash,
                self.world_state_hash,
                self.registry_hash,
                self.policy_hash,
                self.isolation_policy_hash,
                self.issued_at,
                self.expires_at,
            ),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected)

    AuthorizationCertificate.create = create_certificate
    AuthorizationCertificate.verify_signature = verify_certificate

    # ------------------------------------------------------------------
    # Speculative async callback: finite wall-clock budget including cleanup.
    # ------------------------------------------------------------------
    async def speculative_rollout_async(
        plan_text,
        eval_fn,
        *,
        context=None,
        timeout_seconds=60.0,
        cleanup_timeout_seconds=5.0,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        ctx = (context or ns["get_root_context"]()).derive(
            name="speculative_rollout_async"
        )
        result: dict[str, Any]
        try:
            score = await asyncio.wait_for(
                eval_fn(ctx),
                timeout=float(timeout_seconds),
            )
            result = {"ok": True, "score": score, "error": None}
        except asyncio.TimeoutError:
            result = {
                "ok": False,
                "score": 0.0,
                "error": f"speculative rollout timeout after {float(timeout_seconds):.3f}s",
            }
        except Exception as exc:
            result = {"ok": False, "score": 0.0, "error": str(exc)}
        try:
            await asyncio.wait_for(
                ctx.async_dispose(),
                timeout=max(0.001, float(cleanup_timeout_seconds)),
            )
        except asyncio.TimeoutError:
            result = {
                "ok": False,
                "score": 0.0,
                "error": (
                    result.get("error")
                    or "speculative rollout cleanup timeout"
                ),
            }
        return result

    ns["speculative_rollout_async"] = speculative_rollout_async

    # ------------------------------------------------------------------
    # Direct LLM judge calls receive an outer timeout even when a caller
    # injects an http client that ignores provider/client timeout settings.
    # ------------------------------------------------------------------
    raw_llm_evaluate = judges_mod.BaseLLMJudge.evaluate

    async def llm_evaluate(
        self,
        plan_ir,
        goal_description="",
        registry=None,
        observed_world_state=None,
        timeout=30.0,
    ):
        if timeout <= 0:
            return self._unknown("judge timeout must be > 0", 0.0)
        started = time.monotonic()
        try:
            return await asyncio.wait_for(
                raw_llm_evaluate(
                    self,
                    plan_ir,
                    goal_description=goal_description,
                    registry=registry,
                    observed_world_state=observed_world_state,
                    timeout=timeout,
                ),
                timeout=float(timeout),
            )
        except asyncio.TimeoutError:
            return self._unknown(
                f"{self.PROVIDER_NAME} total timeout exceeded after {float(timeout):.3f}s",
                (time.monotonic() - started) * 1000.0,
            )

    judges_mod.BaseLLMJudge.evaluate = llm_evaluate

    # ------------------------------------------------------------------
    # Probabilistic verifier is synchronous upstream. Run the whole selection
    # in a daemon worker and bound caller wait time. The built-in transport
    # also keeps per-request timeouts and retry=0; custom sync callbacks cannot
    # block the planning thread indefinitely even if they ignore kwargs.
    # ------------------------------------------------------------------
    raw_select = sv_mod.ProbabilisticSelfVerifier.select

    def bounded_select(self, *args, **kwargs):
        request_timeout = float(
            kwargs.get(
                "request_timeout_seconds",
                sv_mod.DEFAULT_VERIFIER_REQUEST_TIMEOUT_SECONDS,
            )
        )
        if request_timeout <= 0:
            raise ValueError("request_timeout_seconds must be > 0")

        candidates = kwargs.get("candidates") or []
        criteria = kwargs.get("criteria") or sv_mod.DEFAULT_VERIFICATION_CRITERIA
        n_evaluations = int(
            kwargs.get(
                "n_evaluations",
                sv_mod.DEFAULT_SELF_VERIFICATION_N_EVALUATIONS,
            )
        )
        pivots = int(kwargs.get("pivots", sv_mod.DEFAULT_SELF_VERIFICATION_PIVOTS))
        max_workers = int(kwargs.get("max_workers", sv_mod.DEFAULT_VERIFIER_MAX_WORKERS))
        estimated = sv_mod._estimated_verifier_calls(
            len(candidates),
            len(criteria),
            n_evaluations,
            pivots,
        ) if candidates else 1
        waves = max(1, math.ceil(max(1, estimated) / max(1, max_workers)))
        total_timeout = max(request_timeout, request_timeout * waves)

        results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                results.put(("ok", raw_select(self, *args, **kwargs)))
            except BaseException as exc:  # propagate provider/custom failures
                results.put(("error", exc))

        worker = threading.Thread(
            target=invoke,
            name="prime-self-verifier",
            daemon=True,
        )
        worker.start()
        try:
            kind, value = results.get(timeout=total_timeout)
        except queue.Empty as exc:
            raise sv_mod.SelfVerificationUnavailableError(
                f"self-verification total timeout exceeded after {total_timeout:.3f}s"
            ) from exc
        if kind == "error":
            raise value
        return value

    sv_mod.ProbabilisticSelfVerifier.select = bounded_select
