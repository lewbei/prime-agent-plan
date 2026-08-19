"""Canonical typed fact identity helpers.

The planner/runtime historically used human-readable ``fact_key`` strings as
semantic state keys. Plain ``str(arg)`` rendering collapses distinct typed
values (for example integer ``123`` and string ``"123"``), so this module
provides one stable typed encoding shared by WorldFact and PredicateCondition.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# Preserve the existing readable keys for common string and scalar arguments
# (e.g. service_running(prod), distance(a,b,42.5)). Strings that could collide
# with another primitive type, contain fact-key delimiters, or use the reserved
# prefix are encoded explicitly.
_SAFE_STRING = re.compile(r"^[A-Za-z0-9_./:@+\-=]+$")
_TYPED_PREFIX = "@@"
_INTEGER_TEXT = re.compile(r"^-?(0|[1-9][0-9]*)$")


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

    value_type = type(value)
    return {
        "type": f"{value_type.__module__}.{value_type.__qualname__}",
        "value": repr(value),
    }


def _string_collides_with_primitive(value: str) -> bool:
    """Return True when a string equals the display form of another primitive."""
    if value in {"None", "True", "False", "nan", "inf", "-inf"}:
        return True

    if _INTEGER_TEXT.fullmatch(value) is not None:
        return True

    try:
        parsed = float(value)
    except ValueError:
        return False
    return str(parsed) == value


def _encoded_token(value: Any) -> str:
    payload = json.dumps(
        _typed_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"{_TYPED_PREFIX}{payload}"


def canonical_typed_arg_token(value: Any) -> str:
    """Encode one argument into a collision-resistant typed identity token."""
    if type(value) is str:
        if (
            _SAFE_STRING.fullmatch(value) is not None
            and not value.startswith(_TYPED_PREFIX)
            and not _string_collides_with_primitive(value)
        ):
            return value
        return _encoded_token(value)

    # Preserve legacy readable representations for primitive non-string values;
    # ambiguous strings are encoded above, so these representations remain
    # type-safe while keeping existing keys stable.
    if value is None or type(value) in (bool, int, float):
        return str(value)

    return _encoded_token(value)


def canonical_fact_identity(predicate: str, args: List[Any]) -> str:
    """Return the authoritative semantic key for a predicate and typed args."""
    args_str = ",".join(canonical_typed_arg_token(arg) for arg in args)
    return f"{predicate}({args_str})"
