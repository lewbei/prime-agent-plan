"""Compatibility correction for PR5 authorization hardening.

Authorization must reject post-validation PlanIR mutation and capability-contract
incompatibility, but it must not silently replace the validator chosen by the
caller or advance a synthetic validation clock. Runtime TTL freshness remains a
``start_execution`` responsibility.
"""
from __future__ import annotations

import inspect
from typing import Any


def install_authorization_compat() -> None:
    from .session import PlanningSession, StateDriftError

    current = PlanningSession.authorize_selected
    if getattr(current, "_pr5_auth_compat", False):
        return

    # The prior PR5 wrapper closes over the original method. Recover that exact
    # implementation so this correction does not stack the unwanted default
    # causal revalidation a second time.
    try:
        original = inspect.getclosurevars(current).nonlocals.get("original")
    except Exception:
        original = None
    if original is None:
        original = current

    def authorize_selected(self: PlanningSession, registry: Any, policy_hash: str,
                           ttl_seconds: float = 60.0,
                           isolation_policy_hash: str | None = None):
        version_number = self.best_candidate_version
        if version_number is not None and version_number in self.versions:
            version = self.versions[version_number]
            if version.plan_ir.compute_hash() != version.plan_hash:
                raise StateDriftError(
                    "Plan semantics changed after validation; revalidate before authorization."
                )
            try:
                for action in version.plan_ir.actions:
                    registry.validate_action(action)
            except Exception as exc:
                raise StateDriftError(
                    f"Capability registry/contract changed after validation: {exc}"
                ) from exc

        return original(
            self,
            registry,
            policy_hash,
            ttl_seconds=ttl_seconds,
            isolation_policy_hash=isolation_policy_hash,
        )

    authorize_selected._pr5_auth_compat = True  # type: ignore[attr-defined]
    PlanningSession.authorize_selected = authorize_selected  # type: ignore[assignment]
