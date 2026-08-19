"""Integration tests that require a real Bubblewrap kernel isolation backend.

Portable jobs skip these when bwrap is absent.  The dedicated CI job installs
bwrap and requires the tests to run.  Network-deny is deliberately fail-closed:
if the host kernel refuses creation of a private network namespace, the command
must not execute on the host as a fallback.
"""
from __future__ import annotations

import shutil
import sys

import pytest

from plan_mode.runtime.sandbox import ExecutionSandbox, SecurityProfile


pytestmark = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="Bubblewrap is exercised by the dedicated Phase 4 kernel-isolation CI job",
)


def _sandbox(workspace: str, *, allow_network: bool) -> ExecutionSandbox:
    profile = SecurityProfile.NETWORK_ALLOWED if allow_network else SecurityProfile.STRICT
    policy = SecurityProfile.get_profile(profile).model_copy(update={"workspace_dir": workspace})
    sandbox = ExecutionSandbox(policy=policy)
    assert sandbox.kernel_isolation_ready is True
    assert sandbox.is_fail_closed is True
    return sandbox


def test_bwrap_executes_inside_workspace_under_real_kernel_boundary(tmp_path):
    """Prove bwrap itself can initialize and execute, not merely fail before the command."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Network is explicitly allowed only for this positive containment test so
    # hosts that prohibit CLONE_NEWNET can still exercise mount/PID/IPC/UTS.
    sandbox = _sandbox(str(workspace), allow_network=True)
    target = workspace / "inside.txt"

    result = sandbox.execute_argv_pipeline(
        [["sh", "-c", "printf isolated > inside.txt"]],
        cwd=str(workspace),
    )

    assert result.returncode == 0, result.stderr
    assert target.read_text() == "isolated"


def test_bwrap_read_only_root_blocks_write_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    sandbox = _sandbox(str(workspace), allow_network=True)

    result = sandbox.execute_argv_pipeline(
        [[sys.executable, "-S", "-c", f"open({str(outside)!r}, 'w').write('escape')"]],
        cwd=str(workspace),
    )

    assert result.returncode != 0
    assert not outside.exists()


def test_network_default_deny_never_falls_back_to_raw_host_execution(tmp_path):
    """Bypass Python sitecustomize with -S and prove a connection can never succeed.

    On kernels that permit CLONE_NEWNET, the connection fails inside the empty
    network namespace.  On restricted CI kernels, bwrap itself may refuse
    namespace initialization.  In either case the command must never be
    retried raw on the host.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    connected_marker = workspace / "NETWORK_CONNECTED"
    sandbox = _sandbox(str(workspace), allow_network=False)
    script = (
        "import pathlib,socket; "
        "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); "
        "s.settimeout(0.5); "
        "s.connect(('1.1.1.1',80)); "
        f"pathlib.Path({str(connected_marker)!r}).write_text('connected')"
    )

    result = sandbox.execute_argv_pipeline(
        [[sys.executable, "-S", "-c", script]],
        cwd=str(workspace),
        timeout_seconds=2.0,
    )

    assert result.returncode != 0
    assert not connected_marker.exists(), "network-deny silently fell back to host networking"
