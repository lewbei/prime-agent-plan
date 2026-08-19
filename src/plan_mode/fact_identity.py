"""Canonical typed fact identity helpers.

The planner/runtime historically used human-readable ``fact_key`` strings as
semantic state keys.  Plain ``str(arg)`` rendering collapses distinct typed
values (for example ``123`` and ``"123"``), so this module provides one
stable typed encoding shared by WorldFact and PredicateCondition.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# Preserve the existing readable keys for the overwhelmingly common case of
# simple string arguments (``service_running(prod)``), while encoding values
# that would otherwise be ambiguous.  Strings containing delimiters/reserved
# prefixes are encoded too, so multi-argument identities cannot collide.
_SAFE_STRING = re.compile(r"^[A-Za-z0-9_./:@+\-=]+$")
_TYPED_PREFIX = "@@"


def _typed_value(value: Any) -> Dict[str, Any]:
    """Return a deterministic JSON-serializable value preserving Python type."""
    if value is None:
        return {"type": "none", "value": None}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if isinstance(value, list):
        return {"type": "list", "value": [_typed_value(v) for v in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_typed_value(v) for v in value]}
    if isinstance(value, dict):
        pairs: List[List[Dict[str, Any]]] = [
            [_typed_value(k), _typed_value(v)] for k, v in value.items()
        ]
        pairs.sort(
            key=lambda pair: json.dumps(
                pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        return {"type": "dict", "value": pairs}

    # Pydantic's public IR currently permits Any.  Preserve type identity for
    # uncommon values rather than silently collapsing them.  Repr is only a
    # compatibility fallback; normal IR values should use the explicit types
    # above.
    value_type = type(value)
    return {
        "type": f"{value_type.__module__}.{value_type.__qualname__}",
        "value": repr(value),
    }


def canonical_typed_arg_token(value: Any) -> str:
    """Encode one argument into a collision-resistant typed identity token."""
    if (
        type(value) is str
        and _SAFE_STRING.fullmatch(value) is not None
        and not value.startswith(_TYPED_PREFIX)
    ):
        return value

    payload = json.dumps(
        _typed_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"{_TYPED_PREFIX}{payload}"


def canonical_fact_identity(predicate: str, args: List[Any]) -> str:
    """Return the authoritative semantic key for a predicate and typed args."""
    args_str = ",".join(canonical_typed_arg_token(arg) for arg in args)
    return f"{predicate}({args_str})"
