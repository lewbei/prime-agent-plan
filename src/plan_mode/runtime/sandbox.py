"""Hardened Execution Sandbox with Kernel Namespaces, Resource Limits, and Path Traversal Defenses (Phase 4).

Features:
- Linux Bubblewrap (bwrap) unprivileged container isolation (User, Mount, PID, Network namespaces) when available.
- Portable Defense-in-Depth Layer:
  - Command argument validation (blocking access to blocked host paths and writes outside workspace).
  - Network default-deny policy (network namespace unsharing via bwrap and socket blocking via environment hooks).
  - POSIX resource limits (RLIMIT_AS memory bounds, RLIMIT_CPU compute bounds, RLIMIT_NPROC process bounds, RLIMIT_FSIZE write caps).
  - Path traversal ('../') and symlink escape verification.
  - Output size truncation and secret scrubbing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.runtime.secret_scrubber import SecretScrubber

try:
    import resource
    HAS_RESOURCE_MODULE = True
except ImportError:
    HAS_RESOURCE_MODULE = False


class PathTraversalEscapeError(Exception):
    """Raised when a path or command attempts to escape the designated workspace boundary."""
    pass


class SymlinkEscapeError(Exception):
    """Raised when a symlink resolves to a target outside the workspace boundary."""
    pass


class SandboxResourceLimitExceededError(Exception):
    """Raised when an action exceeds configured CPU, memory, or process limits."""
    pass


class SandboxSecurityViolationError(Exception):
    """Raised when an action attempts an unauthorized network or filesystem operation."""
    pass


class SecurityProfileType(str, Enum):
    STRICT = "STRICT"
    PERMISSIVE_DEV = "PERMISSIVE_DEV"
    NETWORK_ALLOWED = "NETWORK_ALLOWED"


class IsolationPolicy(BaseModel):
    """Declarative security policy constraining capability and payload execution."""
    workspace_dir: Optional[str] = None
    allow_network: bool = False
    read_only_root: bool = True
    blocked_paths: List[str] = Field(
        default_factory=lambda: [
            "/etc/shadow",
            "/etc/sudoers",
            "/root",
            os.path.expanduser("~/.ssh"),
            os.path.expanduser("~/.aws"),
            os.path.expanduser("~/.gnupg"),
        ]
    )
    memory_limit_bytes: Optional[int] = 512 * 1024 * 1024  # 512MB RAM cap
    max_processes: Optional[int] = 32                      # 32 max child processes
    cpu_time_limit_seconds: Optional[float] = 10.0         # 10s CPU compute cap
    max_file_size_bytes: Optional[int] = 50 * 1024 * 1024  # 50MB max file write
    max_output_size_bytes: int = 100 * 1024                # 100KB stdout/stderr truncation cap
    env_whitelist: List[str] = Field(
        default_factory=lambda: ["PATH", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "HOME"]
    )
    use_bwrap: bool = True
    require_bwrap: bool = False
    allow_unisolated_fallback: bool = True


class SecurityProfile:
    """Pre-configured security policy profiles."""
    STRICT = SecurityProfileType.STRICT
    PERMISSIVE_DEV = SecurityProfileType.PERMISSIVE_DEV
    NETWORK_ALLOWED = SecurityProfileType.NETWORK_ALLOWED

    @classmethod
    def get_profile(cls, profile_type: SecurityProfileType | str) -> IsolationPolicy:
        if isinstance(profile_type, str):
            profile_type = SecurityProfileType(profile_type)

        if profile_type == SecurityProfileType.STRICT:
            return IsolationPolicy(
                allow_network=False,
                read_only_root=True,
                max_processes=16,
                memory_limit_bytes=256 * 1024 * 1024,
                cpu_time_limit_seconds=5.0,
            )
        elif profile_type == SecurityProfileType.PERMISSIVE_DEV:
            return IsolationPolicy(
                allow_network=False,
                read_only_root=False,
                max_processes=64,
                memory_limit_bytes=1024 * 1024 * 1024,
                cpu_time_limit_seconds=30.0,
            )
        elif profile_type == SecurityProfileType.NETWORK_ALLOWED:
            return IsolationPolicy(
                allow_network=True,
                read_only_root=True,
                max_processes=32,
                memory_limit_bytes=512 * 1024 * 1024,
                cpu_time_limit_seconds=10.0,
            )
        return IsolationPolicy()


def validate_path_within_workspace(path: str, workspace_dir: str) -> str:
    """Verify that path resides strictly inside workspace_dir, resolving symlinks and '../'."""
    real_ws = os.path.realpath(os.path.abspath(workspace_dir))
    norm_path = os.path.normpath(os.path.abspath(path))

    # Check for relative traversal escape
    try:
        common_norm = os.path.commonpath([norm_path, real_ws])
        if common_norm != real_ws and norm_path != real_ws:
            raise PathTraversalEscapeError(
                f"Path traversal escape detected: '{path}' escapes workspace '{workspace_dir}'"
            )
    except ValueError:
        raise PathTraversalEscapeError(f"Path '{path}' is on a different drive or invalid for workspace '{workspace_dir}'")

    # Check symlink target resolution
    if os.path.islink(path) or os.path.exists(path):
        real_target = os.path.realpath(path)
        try:
            common_real = os.path.commonpath([real_target, real_ws])
            if common_real != real_ws and real_target != real_ws:
                raise SymlinkEscapeError(
                    f"Symlink escape detected: '{path}' points to '{real_target}' outside workspace '{workspace_dir}'"
                )
        except ValueError:
            raise SymlinkEscapeError(f"Symlink '{path}' resolves outside workspace boundary")
        return real_target

    return norm_path


class EphemeralWorkspace:
    """Secure temporary workspace context manager with strict 0o700 permissions and automated cleanup."""

    def __init__(self, base_dir: Optional[str] = None, prefix: str = "prime_ws_"):
        self.base_dir = base_dir
        self.prefix = prefix
        self.path: str = ""

    def __enter__(self) -> EphemeralWorkspace:
        self.path = tempfile.mkdtemp(prefix=self.prefix, dir=self.base_dir)
        os.chmod(self.path, 0o700)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)


class SandboxExecutionResult(BaseModel):
    """Execution telemetry and scrubbed outputs from structured subprocess execution."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration_ms: float = 0.0
    timeout_exceeded: bool = False
    resource_limit_exceeded: bool = False


# Embedded sandbox security hook for environment isolation & fallback containment
_NET_BLOCKER_SCRIPT = """
import socket, builtins, os, io

# 1. Network Default-Deny Hook
if os.environ.get("PRIME_NETWORK_DENY") == "1":
    def _blocked_socket(*args, **kwargs):
        raise OSError(101, "Network unreachable (Default-Deny policy enforced by ExecutionSandbox)")
    socket.socket = _blocked_socket
    if hasattr(socket, 'create_connection'):
        socket.create_connection = _blocked_socket

# 2. Filesystem Read-Only Root & Workspace Confinement Hook
_ws_dir = os.environ.get("PRIME_WORKSPACE_DIR")
_ro_root = os.environ.get("PRIME_READ_ONLY_ROOT") == "1"

if _ro_root and _ws_dir:
    _orig_builtin_open = builtins.open
    _real_ws = os.path.realpath(os.path.abspath(_ws_dir))

    def _sandboxed_open(file, mode="r", *args, **kwargs):
        # Block write/append/truncate modes outside workspace
        if any(m in mode for m in ("w", "a", "+", "x")):
            file_str = str(file)
            abs_p = os.path.normpath(os.path.abspath(file_str)) if os.path.isabs(file_str) else os.path.normpath(os.path.abspath(os.path.join(os.getcwd(), file_str)))
            try:
                common = os.path.commonpath([abs_p, _real_ws])
                if common != _real_ws:
                    raise OSError(30, f"Read-only file system: write to '{file_str}' outside workspace '{_ws_dir}' is forbidden")
            except ValueError:
                raise OSError(30, f"Read-only file system: '{file_str}' outside workspace")
        return _orig_builtin_open(file, mode, *args, **kwargs)

    builtins.open = _sandboxed_open
    io.open = _sandboxed_open
"""


class ExecutionSandbox:
    """Executes structured argv pipelines with Bubblewrap/namespace isolation, rlimits, and secret scrubbing."""

    DEFAULT_ENV_WHITELIST = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    def __init__(
        self,
        policy: Optional[IsolationPolicy] = None,
        scrubber: Optional[SecretScrubber] = None,
    ):
        self.policy = policy or IsolationPolicy()
        self.scrubber = scrubber or SecretScrubber()
        self._bwrap_binary = shutil.which("bwrap") if self.policy.use_bwrap else None
        self._prlimit_binary = shutil.which("prlimit")

    def execute_argv_pipeline(
        self,
        pipeline: List[List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 10.0,
        input_data: Optional[str] = None,
    ) -> SandboxExecutionResult:
        """Execute a pipeline of commands: cmd_0 | cmd_1 | ... | cmd_n under full isolation."""
        if not pipeline:
            return SandboxExecutionResult(returncode=0)

        # Fail-closed check: if require_bwrap is enabled, refuse execution when bwrap is missing
        if self.policy.require_bwrap and not self._bwrap_binary:
            return SandboxExecutionResult(
                stderr="Security violation: Kernel container isolation backend (bwrap) is unavailable. Execution refused under fail-closed security policy.",
                returncode=126,
            )

        effective_cwd = cwd or self.policy.workspace_dir or os.getcwd()

        # Build sanitized execution environment (strictly whitelisted, stripping host secrets)
        exec_env = dict(self.DEFAULT_ENV_WHITELIST)
        for key in self.policy.env_whitelist:
            if key in os.environ and key not in exec_env:
                exec_env[key] = os.environ[key]
        if env:
            exec_env.update(env)

        # Static defense check: inspect command tokens for security violations
        sec_err = self._check_command_security(pipeline, effective_cwd)
        if sec_err:
            return SandboxExecutionResult(
                stderr=sec_err,
                returncode=126,
            )

        # If workspace_dir is specified, validate cwd
        if self.policy.workspace_dir:
            try:
                validate_path_within_workspace(effective_cwd, self.policy.workspace_dir)
            except (PathTraversalEscapeError, SymlinkEscapeError) as e:
                return SandboxExecutionResult(
                    stderr=f"Security violation: {str(e)}",
                    returncode=126,
                )

        # Security containment environment hooks
        if not self.policy.allow_network:
            exec_env["PRIME_NETWORK_DENY"] = "1"
        if self.policy.read_only_root:
            exec_env["PRIME_READ_ONLY_ROOT"] = "1"
        if self.policy.workspace_dir:
            exec_env["PRIME_WORKSPACE_DIR"] = self.policy.workspace_dir

        net_hook_dir = tempfile.mkdtemp(prefix="prime_sec_hook_")
        hook_file = os.path.join(net_hook_dir, "sitecustomize.py")
        with open(hook_file, "w", encoding="utf-8") as hf:
            hf.write(_NET_BLOCKER_SCRIPT)
        orig_pypath = exec_env.get("PYTHONPATH", "")
        exec_env["PYTHONPATH"] = f"{net_hook_dir}:{orig_pypath}" if orig_pypath else net_hook_dir

        start_time = time.time()
        processes: List[subprocess.Popen] = []

        preexec = self._build_preexec_fn()

        try:
            prev_stdout = None
            for idx, raw_cmd in enumerate(pipeline):
                is_first = (idx == 0)
                is_last = (idx == len(pipeline) - 1)

                sandboxed_cmd = self._wrap_command_with_bwrap(raw_cmd, effective_cwd)

                stdin_source = subprocess.PIPE if (is_first and input_data) else prev_stdout
                stdout_target = subprocess.PIPE if is_last else subprocess.PIPE
                stderr_target = subprocess.PIPE

                proc = subprocess.Popen(
                    sandboxed_cmd,
                    stdin=stdin_source,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    cwd=effective_cwd,
                    env=exec_env,
                    text=True,
                    preexec_fn=preexec,
                )
                processes.append(proc)

                if prev_stdout is not None:
                    prev_stdout.close()
                prev_stdout = proc.stdout

            last_proc = processes[-1]
            first_proc = processes[0]
            first_in = input_data if input_data else None

            stdout_out, stderr_out = last_proc.communicate(timeout=timeout_seconds)
            if first_in and first_proc != last_proc and first_proc.stdin:
                try:
                    first_proc.stdin.write(first_in)
                    first_proc.stdin.close()
                except Exception:
                    pass

            duration = (time.time() - start_time) * 1000.0
            returncode = last_proc.returncode

            clean_stdout = self._truncate_and_scrub(stdout_out or "")
            clean_stderr = self._truncate_and_scrub(stderr_out or "")

            return SandboxExecutionResult(
                stdout=clean_stdout,
                stderr=clean_stderr,
                returncode=returncode,
                duration_ms=round(duration, 2),
                timeout_exceeded=False,
            )

        except subprocess.TimeoutExpired:
            for p in processes:
                try:
                    p.kill()
                    p.wait(timeout=1.0)
                except Exception:
                    pass

            duration = (time.time() - start_time) * 1000.0
            return SandboxExecutionResult(
                stdout="",
                stderr="Execution timed out and was forcefully terminated.",
                returncode=124,
                duration_ms=round(duration, 2),
                timeout_exceeded=True,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000.0
            return SandboxExecutionResult(
                stdout="",
                stderr=f"Execution error: {str(e)}",
                returncode=1,
                duration_ms=round(duration, 2),
                timeout_exceeded=False,
            )
        finally:
            if net_hook_dir and os.path.exists(net_hook_dir):
                shutil.rmtree(net_hook_dir, ignore_errors=True)

    def _check_command_security(self, pipeline: List[List[str]], cwd: str) -> Optional[str]:
        """Inspect pipeline commands to block forbidden host files and writes outside workspace."""
        ws = self.policy.workspace_dir
        for cmd in pipeline:
            for token in cmd:
                # 1. Check blocked paths
                for blocked in self.policy.blocked_paths:
                    if blocked and blocked in token:
                        return f"Security violation: access to blocked path '{blocked}' is forbidden"

                # 2. Check shell redirection writes escaping workspace
                if ws and (">" in token or ">>" in token):
                    m = re.search(r">\s*['\"]?([^\s'\";|&]+)", token)
                    if m:
                        target_p = m.group(1).strip()
                        target_abs = os.path.normpath(os.path.abspath(os.path.join(cwd, target_p))) if not os.path.isabs(target_p) else os.path.normpath(os.path.abspath(target_p))
                        try:
                            real_ws = os.path.realpath(os.path.abspath(ws))
                            common = os.path.commonpath([target_abs, real_ws])
                            if common != real_ws and target_abs != real_ws:
                                return f"Security violation: write to '{target_p}' outside workspace '{ws}' is forbidden"
                        except ValueError:
                            return f"Security violation: path '{target_p}' is outside workspace"
        return None

    def _wrap_command_with_bwrap(self, cmd: List[str], cwd: str) -> List[str]:
        """Wrap an argument vector with Bubblewrap namespace flags if available."""
        if not self._bwrap_binary or not self.policy.use_bwrap:
            return cmd

        bwrap_args = [self._bwrap_binary]

        # Filesystem containment
        if self.policy.read_only_root:
            bwrap_args.extend(["--ro-bind", "/", "/"])
        else:
            bwrap_args.extend(["--bind", "/", "/"])

        # Mount devices, procfs, and tmpfs
        if os.path.exists("/dev"):
            bwrap_args.extend(["--dev", "/dev"])
        if os.path.exists("/proc"):
            bwrap_args.extend(["--proc", "/proc"])

        # Mount ephemeral workspace as read-write
        if self.policy.workspace_dir and os.path.exists(self.policy.workspace_dir):
            bwrap_args.extend(["--bind", self.policy.workspace_dir, self.policy.workspace_dir])
        elif not self.policy.workspace_dir:
            if os.path.exists("/tmp"):
                bwrap_args.extend(["--bind", "/tmp", "/tmp"])
            if cwd and os.path.exists(cwd):
                bwrap_args.extend(["--bind", cwd, cwd])

        # Mask blocked sensitive host paths
        for blocked in self.policy.blocked_paths:
            if blocked and os.path.exists(blocked):
                if os.path.isdir(blocked):
                    bwrap_args.extend(["--tmpfs", blocked])
                else:
                    bwrap_args.extend(["--ro-bind", "/dev/null", blocked])

        # Network namespace containment (default deny)
        if not self.policy.allow_network:
            bwrap_args.append("--unshare-net")

        # Process and lifecycle isolation
        bwrap_args.extend([
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--die-with-parent",
            "--chdir", cwd,
            "--",
        ])

        # If prlimit is available, wrap inner command inside container with strict limits
        if self._prlimit_binary:
            prlimit_flags = [self._prlimit_binary]
            if self.policy.max_processes is not None:
                prlimit_flags.append(f"--nproc={self.policy.max_processes}")
            if self.policy.memory_limit_bytes is not None:
                prlimit_flags.append(f"--as={self.policy.memory_limit_bytes}")
            if self.policy.cpu_time_limit_seconds is not None:
                prlimit_flags.append(f"--cpu={int(self.policy.cpu_time_limit_seconds)}")
            if self.policy.max_file_size_bytes is not None:
                prlimit_flags.append(f"--fsize={self.policy.max_file_size_bytes}")
            prlimit_flags.append("--")
            bwrap_args.extend(prlimit_flags)

        bwrap_args.extend(cmd)
        return bwrap_args

    def _build_preexec_fn(self) -> Optional[Any]:
        """Configure POSIX rlimits for memory, CPU, process count, and write sizes."""
        if not HAS_RESOURCE_MODULE:
            return None

        mem_limit = self.policy.memory_limit_bytes
        cpu_limit = int(self.policy.cpu_time_limit_seconds) if self.policy.cpu_time_limit_seconds else None
        nproc_limit = self.policy.max_processes
        fsize_limit = self.policy.max_file_size_bytes

        is_using_bwrap = bool(self._bwrap_binary and self.policy.use_bwrap)

        def _set_limits():
            try:
                if mem_limit is not None and hasattr(resource, "RLIMIT_AS"):
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                if cpu_limit is not None and hasattr(resource, "RLIMIT_CPU"):
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
                if not is_using_bwrap and nproc_limit is not None and hasattr(resource, "RLIMIT_NPROC"):
                    resource.setrlimit(resource.RLIMIT_NPROC, (nproc_limit, nproc_limit))
                if fsize_limit is not None and hasattr(resource, "RLIMIT_FSIZE"):
                    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_limit, fsize_limit))
            except Exception:
                pass

        return _set_limits

    def _truncate_and_scrub(self, text: str) -> str:
        """Apply output size limit cap and scrub secrets."""
        if len(text) > self.policy.max_output_size_bytes:
            truncated = text[: self.policy.max_output_size_bytes] + "\n[TRUNCATED_MAX_OUTPUT_EXCEEDED]"
        else:
            truncated = text
        return self.scrubber.scrub_text(truncated)
