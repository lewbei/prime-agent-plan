#!/usr/bin/env python3
"""Runner for evaluating all 89 official Terminal-Bench 2.0 tasks using PrimeHarborAgent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATASET_PATH = os.path.join(_ROOT, "benchmarks", "harbor_dataset", "terminal-bench")


def run_full_evaluation(
    arm: str = "A6",
    model: str = "claude-3-5-sonnet",
    concurrency: int = 4,
    limit: int | None = None,
    output_dir: str = "results",
) -> None:
    """Execute the Harbor benchmark across all 89 official Terminal-Bench 2.0 tasks."""
    print("=" * 75)
    print(f"LAUNCHING OFFICIAL TERMINAL-BENCH 2.0 EVALUATION (89 TASKS)")
    print(f"Agent Arm: {arm} | Foundation Model: {model} | Concurrency: {concurrency}")
    print(f"Dataset Path: {DATASET_PATH}")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"harbor_tb2_arm_{arm.lower()}_{int(time.time())}.json")

    cmd = [
        "harbor", "run",
        "--path", DATASET_PATH,
        "--agent", "benchmarks.harbor.harbor_adapter:PrimeHarborAgent",
        "--model", model,
        "--agent-kwarg", f"ablation_arm={arm}",
        "--n-concurrent", str(concurrency),
        "--yes",
    ]

    if limit:
        cmd.extend(["--n-tasks", str(limit)])

    print(f"Running Harbor Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        res = subprocess.run(cmd, check=True)
        dur = time.time() - t0
        print("=" * 75)
        print(f"OFFICIAL TERMINAL-BENCH 2.0 EVALUATION FINISHED in {dur:.1f}s")
        print("=" * 75)
    except subprocess.CalledProcessError as e:
        print(f"Harbor run exited with returncode {e.returncode}")
    except Exception as ex:
        print(f"Exception during Harbor execution: {ex}")


def main():
    parser = argparse.ArgumentParser(description="Official Terminal-Bench 2.0 (89 Tasks) Evaluator")
    parser.add_argument("--arm", default="A6", help="Ablation arm to evaluate (A0-A6)")
    parser.add_argument("--model", default="claude-3-5-sonnet", help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent container trials")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks (e.g. 5 for smoke test)")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    run_full_evaluation(
        arm=args.arm,
        model=args.model,
        concurrency=args.concurrency,
        limit=args.limit,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
