"""Structured Subprocess Runner with argv Pipeline Chaining (No Shell Interpolation).

Note: This runner executes explicit argument vectors via native OS pipes and stripped environment variables.
It does NOT provide hostile-code containment or OS-level container isolation.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from plan_mode.runtime.secret_scrubber import SecretScrubber


class SandboxExecutionResult(BaseModel):
    """Execution telemetry and scrubbed outputs from structured subprocess execution."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration_ms: float = 0.0
    timeout_exceeded: bool = False


class ExecutionSandbox:
    """Executes structured argv pipelines with direct OS pipes and whitelisted environment variables."""

    DEFAULT_ENV_WHITELIST = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    def __init__(self, scrubber: Optional[SecretScrubber] = None):
        self.scrubber = scrubber or SecretScrubber()

    def execute_argv_pipeline(
        self,
        pipeline: List[List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 10.0,
        input_data: Optional[str] = None,
    ) -> SandboxExecutionResult:
        """Execute a pipeline of commands: cmd_0 | cmd_1 | ... | cmd_n via direct OS pipes."""
        if not pipeline:
            return SandboxExecutionResult(returncode=0)

        # Build sanitized execution environment
        exec_env = dict(self.DEFAULT_ENV_WHITELIST)
        if env:
            exec_env.update(env)

        start_time = time.time()
        processes: List[subprocess.Popen] = []

        try:
            prev_stdout = None
            for idx, cmd in enumerate(pipeline):
                is_first = (idx == 0)
                is_last = (idx == len(pipeline) - 1)

                stdin_source = subprocess.PIPE if (is_first and input_data) else prev_stdout
                stdout_target = subprocess.PIPE if is_last else subprocess.PIPE
                stderr_target = subprocess.PIPE

                proc = subprocess.Popen(
                    cmd,
                    stdin=stdin_source,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    cwd=cwd,
                    env=exec_env,
                    text=True,
                )
                processes.append(proc)

                # Close intermediate reading descriptor in parent so child gets EOF
                if prev_stdout is not None:
                    prev_stdout.close()
                prev_stdout = proc.stdout

            # Send input to first process if provided
            last_proc = processes[-1]
            first_proc = processes[0]
            first_in = input_data if input_data else None

            # Communicate with pipeline
            stdout_out, stderr_out = last_proc.communicate(timeout=timeout_seconds)
            if first_in and first_proc != last_proc and first_proc.stdin:
                try:
                    first_proc.stdin.write(first_in)
                    first_proc.stdin.close()
                except Exception:
                    pass

            duration = (time.time() - start_time) * 1000.0
            returncode = last_proc.returncode

            # Scrub output
            clean_stdout = self.scrubber.scrub_text(stdout_out or "")
            clean_stderr = self.scrubber.scrub_text(stderr_out or "")

            return SandboxExecutionResult(
                stdout=clean_stdout,
                stderr=clean_stderr,
                returncode=returncode,
                duration_ms=round(duration, 2),
                timeout_exceeded=False,
            )

        except subprocess.TimeoutExpired:
            # Terminate all processes in pipeline
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
