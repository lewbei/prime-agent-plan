"""Final cross-cutting closure for the PR5 audited runtime."""
from __future__ import annotations

from typing import Any, MutableMapping

from .runtime_closure_execution import install_execution_closure
from .runtime_closure_public import install_public_release_closure
from .runtime_closure_search import install_search_closure
from .runtime_closure_session import install_session_closure, install_world_state_identity
from .runtime_closure_validation import (
    install_causal_closure,
    install_execution_trace_closure,
    install_memory_closure,
)


def install_runtime_closure(ns: MutableMapping[str, Any]) -> None:
    if ns.get("_RUNTIME_CLOSURE_INSTALLED"):
        return
    install_session_closure(ns)
    install_world_state_identity()
    install_execution_closure()
    install_search_closure()
    install_execution_trace_closure(ns)
    install_memory_closure()
    install_causal_closure()
    install_public_release_closure(ns)
    ns["_RUNTIME_CLOSURE_INSTALLED"] = True
