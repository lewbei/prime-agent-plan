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


# ---------------------------------------------------------------------------
# Test 1: Filesystem Isolation - Blocks write outside workspace
# ---------------------------------------------------------------------------
def test_isolation_blocks_write_outside_workspace(tmp_path):
    """Process running in sandbox must not be able to write outside its designated workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    forbidden_outside_file = tmp_path / "forbidden_outside.txt"

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        read_only_root=True,
    )
    sandbox = ExecutionSandbox(policy=policy)

    # Attempt to write to a path outside workspace
    cmd = ["sh", "-c", f"echo 'hacked' > '{str(forbidden_outside_file)}'"]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))

    # The file outside workspace must NOT exist
    assert not forbidden_outside_file.exists(), "Sandbox permitted write to forbidden host path outside workspace!"


# ---------------------------------------------------------------------------
# Test 2: Filesystem Isolation - Blocks read of forbidden host files
# ---------------------------------------------------------------------------
def test_isolation_blocks_read_forbidden_host_files(tmp_path):
    """Attempting to read blocked system files or host secrets must be blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sensitive_file = tmp_path / "host_sensitive.key"
    sensitive_file.write_text("SUPER_SECRET_HOST_KEY_DATA")

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        blocked_paths=[str(sensitive_file), "/etc/shadow", "/etc/sudoers"],
    )
    sandbox = ExecutionSandbox(policy=policy)

    cmd = ["cat", str(sensitive_file)]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))

    # Must fail or return empty/scrubbed output, never leaking the sensitive key
    assert res.returncode != 0 or "SUPER_SECRET_HOST_KEY_DATA" not in res.stdout


# ---------------------------------------------------------------------------
# Test 3: Network Isolation - Default Deny blocks network egress
# ---------------------------------------------------------------------------
def test_isolation_blocks_network_without_permission(tmp_path):
    """By default (allow_network=False), network sockets / connections must be blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        allow_network=False,  # Default deny
    )
    sandbox = ExecutionSandbox(policy=policy)

    # Python script attempting to create TCP socket connection to external address
    net_script = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(1.0)\n"
        "s.connect(('1.1.1.1', 80))\n"
    )
    cmd = [sys.executable, "-c", net_script]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace), timeout_seconds=3.0)

    # Socket connect must fail (non-zero return code or network unreachable)
    assert res.returncode != 0, "Network connection succeeded under default-deny policy!"


# ---------------------------------------------------------------------------
# Test 4: Network Isolation - Allows network when explicitly granted
# ---------------------------------------------------------------------------
def test_isolation_allows_network_with_explicit_permission(tmp_path):
    """When allow_network=True, local loopback socket binding is permitted."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        allow_network=True,  # Explicitly allowed
    )
    sandbox = ExecutionSandbox(policy=policy)

    # Script binds to a local loopback port
    loopback_script = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.bind(('127.0.0.1', 0))\n"
        "print('BOUND_PORT:', s.getsockname()[1])\n"
        "s.close()\n"
    )
    cmd = [sys.executable, "-c", loopback_script]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace), timeout_seconds=3.0)
    assert res.returncode == 0
    assert "BOUND_PORT:" in res.stdout


# ---------------------------------------------------------------------------
# Test 5: Process Isolation - Limits process count / fork bomb defense
# ---------------------------------------------------------------------------
def test_isolation_blocks_fork_bomb_or_process_explosion(tmp_path):
    """Process attempting to spawn unbounded processes hits process limit and is contained."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        max_processes=10,  # Strict process limit
    )
    sandbox = ExecutionSandbox(policy=policy)

    # Fork script attempting to create 50 child processes
    fork_script = (
        "import os, time\n"
        "children = []\n"
        "try:\n"
        "    for i in range(50):\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            time.sleep(1)\n"
        "            os._exit(0)\n"
        "        children.append(pid)\n"
        "    print('SPAWNED_ALL')\n"
        "except (BlockingIOError, OSError) as e:\n"
        "    print('FORK_BLOCKED_SUCCESSFULLY')\n"
    )
    cmd = [sys.executable, "-c", fork_script]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace), timeout_seconds=5.0)

    # Fork limit must either block additional forks or terminate process
    assert "SPAWNED_ALL" not in res.stdout or "FORK_BLOCKED_SUCCESSFULLY" in res.stdout


# ---------------------------------------------------------------------------
# Test 6: Resource Limits - Memory allocation bounds (RLIMIT_AS)
# ---------------------------------------------------------------------------
def test_isolation_enforces_memory_limits(tmp_path):
    """Process attempting to allocate memory exceeding limit fails safely."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        memory_limit_bytes=64 * 1024 * 1024,  # 64MB RAM limit
    )
    sandbox = ExecutionSandbox(policy=policy)

    # Script attempting to allocate 256MB RAM
    mem_script = (
        "try:\n"
        "    data = bytearray(256 * 1024 * 1024)\n"
        "    print('ALLOCATED_EXCESSIVE_MEM')\n"
        "except MemoryError:\n"
        "    print('MEMORY_LIMIT_CONTAINED')\n"
    )
    cmd = [sys.executable, "-c", mem_script]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace), timeout_seconds=5.0)

    # Must NOT succeed in allocating 256MB
    assert "ALLOCATED_EXCESSIVE_MEM" not in res.stdout


# ---------------------------------------------------------------------------
# Test 7: Resource Limits - CPU compute limits and execution timeouts
# ---------------------------------------------------------------------------
def test_isolation_enforces_cpu_limits_and_timeouts(tmp_path):
    """CPU-intensive infinite loop is bounded by timeout."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        cpu_time_limit_seconds=2.0,
    )
    sandbox = ExecutionSandbox(policy=policy)

    cmd = [sys.executable, "-c", "while True: pass"]
    t0 = time.time()
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace), timeout_seconds=1.0)
    elapsed = time.time() - t0

    assert res.timeout_exceeded is True
    assert elapsed < 3.0, f"Process took too long to terminate ({elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Test 8: Output Size Limits - Prevents stdout flood memory exhaustion
# ---------------------------------------------------------------------------
def test_isolation_enforces_output_size_limits(tmp_path):
    """Command producing megabytes of stdout is truncated at max_output_size_bytes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        max_output_size_bytes=10 * 1024,  # 10KB output cap
    )
    sandbox = ExecutionSandbox(policy=policy)

    # Script outputting 1MB of text
    flood_script = "print('A' * (1024 * 1024))"
    cmd = [sys.executable, "-c", flood_script]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace), timeout_seconds=5.0)

    assert len(res.stdout) <= 15 * 1024, f"Output length was {len(res.stdout)}, exceeded cap!"
    assert "[TRUNCATED" in res.stdout or len(res.stdout) <= 10 * 1024


# ---------------------------------------------------------------------------
# Test 9: Environment Isolation - Strips host secrets and credentials
# ---------------------------------------------------------------------------
def test_isolation_strips_host_environment_secrets(tmp_path, monkeypatch):
    """Host environment variables containing secrets must be stripped from sandbox environment."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Inject simulated host secret into current process environment
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AKIAHOSTSECRET123456789")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-supersecretkey987654321")
    monkeypatch.setenv("DATABASE_URL", "postgres://admin:superpass@prod-db.internal/main")

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
    )
    sandbox = ExecutionSandbox(policy=policy)

    cmd = [sys.executable, "-c", "import os; print(list(os.environ.keys()))"]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))

    assert "AWS_SECRET_ACCESS_KEY" not in res.stdout
    assert "OPENAI_API_KEY" not in res.stdout
    assert "DATABASE_URL" not in res.stdout


# ---------------------------------------------------------------------------
# Test 10: Path Traversal Defense - Blocks symlink escape attempts
# ---------------------------------------------------------------------------
def test_isolation_detects_and_blocks_symlink_escape(tmp_path):
    """Symlinks pointing outside the workspace directory must be detected and rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside_secret.txt"
    outside_target.write_text("OUTSIDE_SECRET")

    # Create symlink inside workspace pointing to outside target
    symlink_path = workspace / "escape_link.txt"
    os.symlink(str(outside_target), str(symlink_path))

    # Path validator must detect symlink escape
    with pytest.raises((SymlinkEscapeError, PathTraversalEscapeError)):
        validate_path_within_workspace(str(symlink_path), str(workspace))


# ---------------------------------------------------------------------------
# Test 11: Path Traversal Defense - Blocks relative path traversal escapes
# ---------------------------------------------------------------------------
def test_isolation_blocks_path_traversal_escapes(tmp_path):
    """Paths with '../' escaping the workspace boundary must be rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    traversal_path = str(workspace / "../../../etc/passwd")

    with pytest.raises(PathTraversalEscapeError):
        validate_path_within_workspace(traversal_path, str(workspace))


# ---------------------------------------------------------------------------
# Test 12: Shell Injection Defense - Structured argv execution without shell interpolation
# ---------------------------------------------------------------------------
def test_isolation_prevents_shell_injection(tmp_path):
    """Parameters containing shell metacharacters (; | &&) must be treated as literal argv without execution."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    marker_file = workspace / "injected_marker.txt"

    policy = IsolationPolicy(workspace_dir=str(workspace))
    sandbox = ExecutionSandbox(policy=policy)

    # Parameter contains malicious shell injection payload
    malicious_param = f"hello; touch {str(marker_file)};"
    cmd = ["echo", malicious_param]

    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))
    assert res.returncode == 0
    # Marker file must NOT have been created by shell expansion
    assert not marker_file.exists(), "Shell injection payload was executed!"


# ---------------------------------------------------------------------------
# Test 13: Ephemeral Workspace Lifecycle & Automatic Teardown
# ---------------------------------------------------------------------------
def test_isolation_workspace_lifecycle_and_teardown(tmp_path):
    """EphemeralWorkspace creates isolated directory with 0o700 permissions and wipes on exit."""
    created_workspace_path = None

    with EphemeralWorkspace(base_dir=str(tmp_path)) as ws:
        created_workspace_path = Path(ws.path)
        assert created_workspace_path.exists()
        # Verify strict 0o700 permissions
        stat_mode = os.stat(ws.path).st_mode & 0o777
        assert stat_mode == 0o700, f"Expected 0o700 permissions, got {oct(stat_mode)}"

        # Write file inside workspace
        test_file = created_workspace_path / "temp_file.txt"
        test_file.write_text("ephemeral data")
        assert test_file.exists()

    # After exiting context manager, workspace must be completely deleted
    assert created_workspace_path is not None
    assert not created_workspace_path.exists(), "Ephemeral workspace was not destroyed after transaction exit!"


# ---------------------------------------------------------------------------
# Test 14: Security Profiles and Capability Policy Binding
# ---------------------------------------------------------------------------
def test_isolation_policy_capabilities_and_profiles():
    """Security profiles (STRICT, PERMISSIVE_DEV, NETWORK_ALLOWED) configure correct policy parameters."""
    strict = SecurityProfile.get_profile(SecurityProfile.STRICT)
    assert strict.allow_network is False
    assert strict.read_only_root is True
    assert strict.max_processes <= 32

    net_prof = SecurityProfile.get_profile(SecurityProfile.NETWORK_ALLOWED)
    assert net_prof.allow_network is True


# ---------------------------------------------------------------------------
# Test 15: Fail-Closed Isolation - Refuses execution when bwrap is unavailable
# ---------------------------------------------------------------------------
def test_isolation_refuses_execution_when_bwrap_unavailable(tmp_path):
    """When require_bwrap=True and bwrap is unavailable, execution must be refused rather than run raw on host."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        require_bwrap=True,
    )
    sandbox = ExecutionSandbox(policy=policy)
    # Simulate bwrap missing on the host
    sandbox._bwrap_binary = None

    cmd = ["echo", "should_not_run"]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))

    assert res.returncode == 126
    assert any("refused" in res.stderr.lower() or "unavailable" in res.stderr.lower() for _ in [1])


# ---------------------------------------------------------------------------
# Test 16: Filesystem Isolation - Blocks non-shell programmatic writes outside workspace
# ---------------------------------------------------------------------------
def test_isolation_blocks_non_shell_file_write_outside_workspace(tmp_path):
    """Python programmatic open('/outside/path', 'w') must fail with Read-only filesystem error in both bwrap and portable fallback modes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside_py_write.txt"

    policy = IsolationPolicy(
        workspace_dir=str(workspace),
        read_only_root=True,
    )
    sandbox = ExecutionSandbox(policy=policy)

    py_script = f"open('{str(outside_target)}', 'w').write('hacked')"
    cmd = [sys.executable, "-c", py_script]
    res = sandbox.execute_argv_pipeline([cmd], cwd=str(workspace))

    assert not outside_target.exists()
    assert res.returncode != 0

    # Test portable fallback without bwrap
    outside_target_fallback = tmp_path / "outside_fallback_write.txt"
    sandbox._bwrap_binary = None
    py_script_fallback = f"open('{str(outside_target_fallback)}', 'w').write('hacked')"
    cmd_fallback = [sys.executable, "-c", py_script_fallback]
    res_fallback = sandbox.execute_argv_pipeline([cmd_fallback], cwd=str(workspace))

    assert not outside_target_fallback.exists()
    assert res_fallback.returncode != 0


# ---------------------------------------------------------------------------
# Test 17: Transactional Execution Automatically Binds & Wipes Ephemeral Workspace
# ---------------------------------------------------------------------------
def test_transactional_execution_automatically_binds_and_wipes_ephemeral_workspace(tmp_path):
    """TransactionalExecutionManager runs cleanly inside an EphemeralWorkspace with full post-exit teardown."""
    from plan_mode.registry import CapabilityRegistry, CapabilityEntry, ObservationVerifier
    from plan_mode.session import PlanningSession
    from plan_mode.runtime.ledger import EvidenceLedger
    from plan_mode.runtime.transaction import TransactionalExecutionManager, TransactionOutcome
    from plan_mode.runtime.sandbox import EphemeralWorkspace, ExecutionSandbox, IsolationPolicy

    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            name="fs.create_ephemeral",
            description="Create file",
            input_schema={"name": {"type": "str", "required": True}},
            positive_effects=[_cond("done", ["{name}"])],
            verifiers=[ObservationVerifier(verifier_id="v1", predicate="done", target_args_mapping=["{name}"], command_template=["true"])],
            executor_command_template=["touch", "{name}"],
        )
    )

    with EphemeralWorkspace(base_dir=str(tmp_path)) as ws:
        target_file = os.path.join(ws.path, "test.txt")
        plan = PlanIR(
            plan_id="p_auto_ws",
            goal_description="Test auto ephemeral workspace",
            initial_state=[],
            actions=[_action(action_id="act1", capability_name="fs.create_ephemeral", parameters={"name": target_file}, positive_effects=[_cond("done", [target_file])])],
        )
        session = PlanningSession(session_id="s_auto_ws")
        session.submit_draft(plan)
        session.validate_candidate(1, reg, observed_world_state=[])
        session.select_version(1)
        policy_hash = reg.compute_registry_hash()
        cert = session.authorize_selected(reg, policy_hash=policy_hash)
        session.start_execution(reg, policy_hash=policy_hash, current_world_facts=[])

        sandbox = ExecutionSandbox(policy=IsolationPolicy(workspace_dir=ws.path))
        manager = TransactionalExecutionManager(
            session=session,
            registry=reg,
            ledger=EvidenceLedger(session_id=session.session_id),
            observed_world_state=[],
            policy_hash=policy_hash,
            sandbox=sandbox,
        )
        summary = manager.execute_and_finalize(cert)
        assert summary.outcome == TransactionOutcome.COMMITTED
        assert os.path.exists(target_file)

    # After exiting context manager, workspace is completely wiped
    assert not os.path.exists(ws.path)
