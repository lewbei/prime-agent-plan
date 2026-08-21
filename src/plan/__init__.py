"""Compatibility alias for the authoritative :mod:`plan_mode` runtime.

There is one public runtime object and one state/correctness surface. Importing
``plan`` therefore returns the exact same module object as ``plan_mode`` in any
fresh-interpreter import order.
"""
from __future__ import annotations

import sys as _sys
import plan_mode as _plan_mode

_sys.modules[__name__] = _plan_mode
