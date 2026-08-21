"""Compatibility alias for the authoritative :mod:`plan_mode` runtime.

PR #5 removes import-order-dependent wrapper installation. Importing ``plan``
returns the exact same module object as ``plan_mode`` so there is one runtime,
one state surface, and one set of correctness/security invariants.
"""
from __future__ import annotations

import sys as _sys
import plan_mode as _plan_mode

_sys.modules[__name__] = _plan_mode
