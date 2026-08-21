"""Public plan_mode API facade with late-bound correctness hardening.

The historical implementation is preserved byte-for-byte in ``_api_impl.py``
and executed in this module namespace. The established public wrappers are
likewise executed here from ``_public_api_impl.py`` so ``plan`` can remain an
exact alias without changing signatures, monkeypatch behavior, or public
semantics. The audited runtime-closure layer is installed last.
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
from .runtime_closure import install_runtime_closure as _install_runtime_closure

_install_api_hardening(globals())
_install_api_hardening_compat(globals())
_install_followup_hardening(globals())
_install_authorization_compat()

_public_api_path = _BootstrapPath(__file__).with_name("_public_api_impl.py")
_public_api_source = _public_api_path.read_text(encoding="utf-8")
exec(
    compile(_public_api_source, str(_public_api_path), "exec"),
    globals(),
    globals(),
)
del _public_api_source

_install_runtime_closure(globals())

del _install_api_hardening
del _install_api_hardening_compat
del _install_followup_hardening
del _install_authorization_compat
del _install_runtime_closure
del _public_api_path
del _impl_path
del _BootstrapPath
