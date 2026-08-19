"""Adversarial Isolation and Security Boundary Tests (Phase 4).

Verifies kernel namespaces/bwrap isolation, resource limits (CPU/RAM/forks),
network default-deny, filesystem jails, symlink escape defenses, path traversal
containment, secret stripping, and ephemeral workspace teardown.
"""

import json
import os
import shutil
import socket
import sys
import time
import pytest
from pathlib import Path

from plan_mode.runtime.sandbox import (
    ExecutionSandbox,
    SandboxExecutionResult,
    IsolationPolicy,
    EphemeralWorkspace,
    SecurityProfile,
    validate_path_within_workspace,
    PathTraversalEscapeError,
    SymlinkEscapeError,
)
from plan_mode.runtime.secret_scrubber import SecretScrubber
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    WorldFact,
)

def _cond(predicate: str, args: list, truth: FactTruth = FactTruth.VERIFIED_TRUE) -> PredicateCondition:
    return PredicateCondition(predicate=predicate, args=args, expected_truth=truth)

def _action(
    action_id: str,
    capability_name: str,
    parameters: dict | None = None,
    positive_effects: list[PredicateCondition] | None = None,
    negative_effects: list[PredicateCondition] | None = None,
    preconditions: list[PredicateCondition] | None = None,
) -> ActionIR:
    return ActionIR(
        action_id=action_id,
        capability_name=capability_name,
        parameters=parameters or {},
        preconditions=preconditions or [],
        positive_effects=positive_effects or [],
        negative_effects=negative_effects or [],
        provenance=Provenance(source_type=SourceType.PLANNER_INFERENCE, source_id=action_id),
    )


def test_isolation_blocks_write_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    forbidden_outside_file = tmp_path / "forbidden_outside.txt"
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), read_only_root=True))
    cmd = ["sh", "-c", f"echo 'hacked' > '{str(forbidden_outside_file)}'"]
    sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))
    assert not forbidden_outside_file.exists()


def test_isolation_blocks_read_forbidden_host_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sensitive_file = tmp_path / "host_sensitive.key"
    sensitive_file.write_text("SUPER_SECRET_HOST_KEY_DATA")
    sandbox = ExecutionSandbox(policy=IsolationPolicy(
        workspace_dir=str(workspace),
        blocked_paths=[str(sensitive_file), "/etc/shadow", "/etc/sudoers"],
    ))
    res = sandbox.execute_argv_pipeline([["cat", str(sensitive_file)]], cwd=str(workspace))
    assert res.returncode != 0 or "SUPER_SECRET_HOST_KEY_DATA" not in res.stdout


def test_isolation_blocks_network_without_permission(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), allow_network=False))
    script = (
        "import socket\n"
        "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
        "s.settimeout(1.0)\n"
        "s.connect(('1.1.1.1',80))\n"
    )
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", script]], cwd=str(workspace), timeout_seconds=3.0)
    assert res.returncode != 0


def test_isolation_allows_network_with_explicit_permission(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), allow_network=True))
    script = (
        "import socket\n"
        "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
        "s.bind(('127.0.0.1',0))\n"
        "print('BOUND_PORT:',s.getsockname()[1])\n"
        "s.close()\n"
    )
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", script]], cwd=str(workspace), timeout_seconds=3.0)
    assert res.returncode == 0
    assert "BOUND_PORT:" in res.stdout


def test_isolation_blocks_fork_bomb_or_process_explosion(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), max_processes=10))
    script = (
        "import os,time\n"
        "children=[]\n"
        "try:\n"
        "  for i in range(50):\n"
        "    pid=os.fork()\n"
        "    if pid==0:\n"
        "      time.sleep(1);os._exit(0)\n"
        "    children.append(pid)\n"
        "  print('SPAWNED_ALL')\n"
        "except (BlockingIOError,OSError):\n"
        "  print('FORK_BLOCKED_SUCCESSFULLY')\n"
    )
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", script]], cwd=str(workspace), timeout_seconds=5.0)
    assert "SPAWNED_ALL" not in res.stdout or "FORK_BLOCKED_SUCCESSFULLY" in res.stdout


def test_isolation_enforces_memory_limits(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), memory_limit_bytes=64 * 1024 * 1024))
    script = (
        "try:\n"
        " data=bytearray(256*1024*1024)\n"
        " print('ALLOCATED_EXCESSIVE_MEM')\n"
        "except MemoryError:\n"
        " print('MEMORY_LIMIT_CONTAINED')\n"
    )
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", script]], cwd=str(workspace), timeout_seconds=5.0)
    assert "ALLOCATED_EXCESSIVE_MEM" not in res.stdout


def test_isolation_enforces_cpu_limits_and_timeouts(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), cpu_time_limit_seconds=2.0))
    t0 = time.time()
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", "while True: pass"]], cwd=str(workspace), timeout_seconds=1.0)
    assert res.timeout_exceeded is True
    assert time.time() - t0 < 3.0


def test_isolation_enforces_output_size_limits(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), max_output_size_bytes=10 * 1024))
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", "print('A'*(1024*1024))"]], cwd=str(workspace), timeout_seconds=5.0)
    assert len(res.stdout) <= 15 * 1024
    assert "[TRUNCATED" in res.stdout or len(res.stdout) <= 10 * 1024


def test_isolation_strips_host_environment_secrets(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AKIAHOSTSECRET123456789")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-supersecretkey987654321")
    monkeypatch.setenv("DATABASE_URL", "postgres://admin:superpass@prod-db.internal/main")
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace)))
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", "import os;print(list(os.environ.keys()))"]], cwd=str(workspace))
    assert "AWS_SECRET_ACCESS_KEY" not in res.stdout
    assert "OPENAI_API_KEY" not in res.stdout
    assert "DATABASE_URL" not in res.stdout


def test_isolation_detects_and_blocks_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside_secret.txt"
    outside_target.write_text("OUTSIDE_SECRET")
    symlink_path = workspace / "escape_link.txt"
    os.symlink(str(outside_target), str(symlink_path))
    with pytest.raises((SymlinkEscapeError, PathTraversalEscapeError)):
        validate_path_within_workspace(str(symlink_path), str(workspace))


def test_isolation_blocks_path_traversal_escapes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathTraversalEscapeError):
        validate_path_within_workspace(str(workspace / "../../../etc/passwd"), str(workspace))


def test_isolation_prevents_shell_injection(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    marker_file = workspace / "injected_marker.txt"
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace)))
    malicious_param = f"hello; touch {str(marker_file)};"
    res = sandbox.execute_argv_pipeline([["echo", malicious_param]], cwd=str(workspace))
    assert res.returncode == 0
    assert not marker_file.exists()


def test_isolation_workspace_lifecycle_and_teardown(tmp_path):
    created = None
    with EphemeralWorkspace(base_dir=str(tmp_path)) as ws:
        created = Path(ws.path)
        assert created.exists()
        assert os.stat(ws.path).st_mode & 0o777 == 0o700
        (created / "temp_file.txt").write_text("ephemeral data")
    assert created is not None
    assert not created.exists()


def test_isolation_policy_capabilities_and_profiles():
    strict = SecurityProfile.get_profile(SecurityProfile.STRICT)
    assert strict.allow_network is False
    assert strict.read_only_root is True
    assert strict.max_processes <= 32
    assert strict.require_bwrap is True
    assert strict.allow_unisolated_fallback is False
    net_prof = SecurityProfile.get_profile(SecurityProfile.NETWORK_ALLOWED)
    assert net_prof.allow_network is True
    assert net_prof.require_bwrap is True


def test_isolation_refuses_execution_when_bwrap_unavailable(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), require_bwrap=True))
    sandbox._bwrap_binary = None
    res = sandbox.execute_argv_pipeline([["echo", "should_not_run"]], cwd=str(workspace))
    assert res.returncode == 126
    assert "refused" in res.stderr.lower() or "unavailable" in res.stderr.lower()


def test_isolation_blocks_non_shell_file_write_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside_py_write.txt"
    sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=str(workspace), read_only_root=True))
    res = sandbox.execute_argv_pipeline([[sys.executable, "-c", f"open('{str(outside_target)}','w').write('hacked')"]], cwd=str(workspace))
    assert not outside_target.exists()
    assert res.returncode != 0
    outside_fallback = tmp_path / "outside_fallback_write.txt"
    sandbox._bwrap_binary = None
    res_fallback = sandbox.execute_argv_pipeline([[sys.executable, "-c", f"open('{str(outside_fallback)}','w').write('hacked')"]], cwd=str(workspace))
    assert not outside_fallback.exists()
    assert res_fallback.returncode != 0


def test_transactional_execution_automatically_binds_and_wipes_ephemeral_workspace(tmp_path):
    """Compatibility test for an explicitly caller-managed development workspace.

    The production auto-workspace/fail-closed behavior is tested separately in
    test_phase45_gate_hardening.py.  This test intentionally opts into the
    Phase-3-only test seam so it does not make a security claim.
    """
    from plan_mode.registry import CapabilityRegistry, CapabilityEntry, ObservationVerifier
    from plan_mode.session import PlanningSession
    from plan_mode.runtime.ledger import EvidenceLedger
    from plan_mode.runtime.transaction import TransactionalExecutionManager, TransactionOutcome

    reg = CapabilityRegistry()
    reg.register(CapabilityEntry(
        name="fs.create_ephemeral",
        description="Create file",
        input_schema={"name": {"type": "str", "required": True}},
        positive_effects=[_cond("done", ["{name}"])],
        verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=["{name}"], command_template=["true"])],
        executor_command_template=["touch", "{name}"],
    ))

    with EphemeralWorkspace(base_dir=str(tmp_path)) as ws:
        target_file = os.path.join(ws.path, "test.txt")
        plan = PlanIR(
            plan_id="p_auto_ws",
            goal_description="Test caller managed ephemeral workspace",
            initial_state=[],
            actions=[_action("act1", "fs.create_ephemeral", {"name": target_file}, [_cond("done", [target_file])])],
        )
        session = PlanningSession(session_id="s_auto_ws")
        session.submit_draft(plan)
        session.validate_candidate(1, reg, observed_world_state=[])
        session.select_version(1)
        policy_hash = reg.compute_registry_hash()
        cert = session.authorize_selected(reg, policy_hash=policy_hash)
        session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])
        sandbox = ExecutionSandbox(policy=IsolationPolicy(
            workspace_dir=ws.path,
            use_bwrap=False,
            require_bwrap=False,
            allow_unisolated_fallback=True,
            read_only_root=False,
        ))
        manager = TransactionalExecutionManager(
            session=session,
            registry=reg,
            ledger=EvidenceLedger(session_id=session.session_id),
            observed_world_state=[],
            policy_hash=policy_hash,
            sandbox=sandbox,
            allow_insecure_test_sandbox=True,
        )
        summary = manager.execute_and_finalize(cert)
        assert summary.outcome == TransactionOutcome.COMMITTED
        assert os.path.exists(target_file)
    assert not os.path.exists(ws.path)
