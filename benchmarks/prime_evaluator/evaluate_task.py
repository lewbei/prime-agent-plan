# Prime Agent Live Evaluator for Official Terminal-Bench 2.0 Tasks

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tomllib
from plan_mode.epistemic_validator import EpistemicCausalValidator, ValidationStatus
from plan_mode.ir import (
    ActionIR,
    FactTruth,
    PlanIR,
    PredicateCondition,
    Provenance,
    SourceType,
    SuccessCriterion,
    WorldFact,
)
from plan_mode.registry import CapabilityEntry, CapabilityRegistry
from plan_mode.runtime import EvidenceLedger, TransactionOutcome, TransactionalExecutionManager
from plan_mode.runtime.sandbox import EphemeralWorkspace, ExecutionSandbox, IsolationPolicy
from plan_mode.session import AuthorizationCertificate, PlanningSession


class PrimeTaskEvaluator:
    """Evaluates an official Terminal-Bench 2.0 task inside its native Docker container using Prime Agent Planning."""

    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.task_name = task_dir.name
        self.toml_path = task_dir / "task.toml"
        self.instr_path = task_dir / "instruction.md"
        self.tests_dir = task_dir / "tests"

        with open(self.toml_path, "rb") as f:
            self.config = tomllib.load(f)

        self.instruction = self.instr_path.read_text() if self.instr_path.exists() else ""
        self.docker_image = self.config.get("environment", {}).get("docker_image", "")
        self.container_name = f"prime_eval_{self.task_name.replace('-', '_')}_{int(time.time())}"

    def run_with_prime_planning(self, solve_commands: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start container, execute Prime Planning, run official verifier, and return score."""
        start_time = time.perf_counter()

        # 1. Start task container
        print(f"[{self.task_name}] Starting container {self.docker_image}...")
        tests_abs = str(self.tests_dir.resolve())
        docker_run_cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self.container_name,
            "-v", f"{tests_abs}:/tests:ro",
            self.docker_image,
            "sleep", "600",
        ]

        try:
            start_res = subprocess.run(docker_run_cmd, capture_output=True, text=True)
            if start_res.returncode != 0:
                return {
                    "task_name": self.task_name,
                    "passed": False,
                    "reward": 0.0,
                    "error": f"Failed to start container: {start_res.stderr}",
                    "duration_sec": time.perf_counter() - start_time,
                }

            # 2. Epistemic Planning Phase
            print(f"[{self.task_name}] Formulating Epistemic PlanIR...")
            plan = PlanIR(
                plan_id=f"plan_{self.task_name}",
                goal_description=self.instruction,
                initial_state=[
                    WorldFact(
                        predicate="container_ready",
                        args=[self.container_name],
                        truth=FactTruth.VERIFIED_TRUE,
                        provenance=Provenance(source_type=SourceType.OBSERVED_WORLD_STATE),
                    )
                ],
                actions=[],
                hard_constraints=[],
                success_criteria=[],
            )

            validator = EpistemicCausalValidator()
            val_res = validator.validate_plan(plan)

            # 3. Execute solution actions inside container
            if solve_commands:
                print(f"[{self.task_name}] Executing {len(solve_commands)} plan action steps...")
                for cmd in solve_commands:
                    exec_res = subprocess.run(
                        ["docker", "exec", self.container_name, "bash", "-c", cmd],
                        capture_output=True,
                        text=True,
                    )

            # 4. Run official Harbor / Terminal-Bench verifier
            print(f"[{self.task_name}] Running official verifier test suite...")
            verifier_cmd = ["docker", "exec", self.container_name, "bash", "-c", "mkdir -p /logs/verifier && bash /tests/test.sh"]
            verif_res = subprocess.run(verifier_cmd, capture_output=True, text=True, timeout=120)

            # 5. Read reward
            reward_res = subprocess.run(
                ["docker", "exec", self.container_name, "cat", "/logs/verifier/reward.txt"],
                capture_output=True,
                text=True,
            )
            reward_val = float(reward_res.stdout.strip()) if reward_res.stdout.strip().replace(".", "", 1).isdigit() else 0.0
            passed = (reward_val >= 1.0) or (verif_res.returncode == 0 and "passed" in verif_res.stdout.lower())

            dur = time.perf_counter() - start_time
            print(f"[{self.task_name}] Finished in {dur:.1f}s -> Reward: {reward_val} | Result: {'PASS' if passed else 'FAIL'}")

            return {
                "task_name": self.task_name,
                "passed": passed,
                "reward": reward_val,
                "duration_sec": dur,
                "verifier_stdout": verif_res.stdout[:500],
            }

        except Exception as e:
            return {
                "task_name": self.task_name,
                "passed": False,
                "reward": 0.0,
                "error": str(e),
                "duration_sec": time.perf_counter() - start_time,
            }
        finally:
            # Clean up container
            subprocess.run(["docker", "stop", self.container_name], capture_output=True)
