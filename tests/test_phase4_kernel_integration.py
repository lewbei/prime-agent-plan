"""Integration tests that require a real Bubblewrap kernel isolation backend.

These are skipped on portable developer environments without bwrap, but the
CI ``phase4-kernel-isolation`` job installs bwrap and requires them to pass.
"""
from __future__ import annotations

import os
import shutil
import sys

import pytest

from plan_mode.runtime.sandbox import ExecutionSandbox, SecurityProfile


pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="Bubblewrap is exercised by the dedicated Phase 4 kernel-isolation CI job",
)


def _strict_sandbox(workspace: str) -> ExecutionSandbox:
    policy = SecurityProfile.get_profile(SecurityProfile.STRICT).model_copy(
        update={"workspace_dir": workspace}
    )
    sandbox = ExecutionSandbox(policy=policy)
    assert sandbox.kernel_isolation_ready is True
    assert sandbox.is_fail_closed is True
    return sandbox


def test_strict_bwrap_executes_inside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _strict_sandbox(str(workspace))
    target = workspace / "inside.txt"

    result = sandbox.execute_argv_pipeline(
        [["sh", "-c", "printf isolated > inside.txt"]],
        cwd=str(workspace),
    )

    assert result.returncode == 0, result.stderr
    assert target.read_text() == "isolated"


def test_strict_bwrap_blocks_write_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    sandbox = _strict_sandbox(str(workspace))

    result = sandbox.execute_argv_pipeline(
        [[sys.executable, "-S", "-c", f"open({str(outside)!r}, 'w').write('escape')"]],
        cwd=str(workspace),
    )

    assert result.returncode != 0
    assert not outside.exists()


def test_strict_bwrap_network_default_deny_without_python_hook(tmp_path):
    """Use -S so sitecustomize is not imported; failure must come from namespace isolation."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _strict_sandbox(str(workspace))
    script = (
        "import socket; "
        "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); "
        "s.settimeout(0.5); "
        "s.connect(('1.1.1.1',80))"
    )

    result = sandbox.execute_argv_pipeline(
        [[sys.executable, "-S", "-c", script]],
        cwd=str(workspace),
        timeout_seconds=2.0,
    )

    assert result.returncode != 0
