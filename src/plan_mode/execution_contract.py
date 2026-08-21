"""Execution-contract facade with workspace-containment hardening."""
from __future__ import annotations

from pathlib import Path as _BootstrapPath

_impl_path = _BootstrapPath(__file__).with_name("_execution_contract_impl.py")
_impl_source = _impl_path.read_text(encoding="utf-8")
exec(compile(_impl_source, str(_impl_path), "exec"), globals(), globals())
del _impl_source

from .execution_contract_hardening import (
    install_execution_contract_hardening as _install_execution_contract_hardening,
)

_install_execution_contract_hardening(globals())
del _install_execution_contract_hardening
