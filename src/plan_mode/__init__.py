"""Authoritative plan_mode package entrypoint.

The pre-PR5 engine implementation is preserved byte-for-byte in
``_legacy_init_impl.py`` and executed in this module namespace so all existing
private/public globals, monkeypatch behavior, session formats, and callables keep
their historical module identity. The audited correctness layers are then
installed here, making direct ``import plan_mode`` and compatibility ``plan``
imports converge on the same hardened runtime semantics.
"""
from __future__ import annotations

from pathlib import Path as _Path

_legacy_path = _Path(__file__).with_name("_legacy_init_impl.py")
_legacy_source = _legacy_path.read_text(encoding="utf-8")
exec(compile(_legacy_source, str(_legacy_path), "exec"), globals(), globals())

from ._correctness_hardening import install as _install_correctness_hardening
from ._correctness_hardening_patch2 import patch as _patch_correctness_hardening
from ._correctness_hardening_patch3 import patch as _patch_correctness_api
from ._correctness_hardening_patch5 import patch as _patch_candidate_selection

_install_correctness_hardening(globals())
_patch_correctness_hardening(globals())
_patch_correctness_api(globals())
_patch_candidate_selection(globals())

del (
    _install_correctness_hardening,
    _patch_correctness_hardening,
    _patch_correctness_api,
    _patch_candidate_selection,
    _legacy_source,
    _legacy_path,
    _Path,
)
