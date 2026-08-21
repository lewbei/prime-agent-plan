"""Public plan_mode API facade with late-bound correctness hardening.

The historical implementation is preserved byte-for-byte in ``_api_impl.py``
and executed in this module namespace. Executing it here, rather than importing
it as a separate module, preserves the existing public monkeypatch/testing
semantics: functions still resolve sibling API hooks through ``plan_mode``'s
own globals. After the implementation finishes loading, PR5 hardening replaces
only the audited public boundaries.
"""
from __future__ import annotations

from pathlib import Path as _BootstrapPath

_impl_path = _BootstrapPath(__file__).with_name("_api_impl.py")
_impl_source = _impl_path.read_text(encoding="utf-8")
exec(compile(_impl_source, str(_impl_path), "exec"), globals(), globals())
del _impl_source

from .api_hardening import install_api_hardening as _install_api_hardening
from .api_hardening_compat import install_api_hardening_compat as _install_api_hardening_compat
from .followup_hardening import install_followup_hardening as _install_followup_hardening
from .authorization_compat import install_authorization_compat as _install_authorization_compat

_install_api_hardening(globals())
_install_api_hardening_compat(globals())
_install_followup_hardening(globals())
_install_authorization_compat()
del _install_api_hardening
del _install_api_hardening_compat
del _install_followup_hardening
del _install_authorization_compat
