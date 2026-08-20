#!/usr/bin/env python3
"""Harbor Evaluation Driver for Official Terminal-Bench 2.0 Benchmarking.

Harbor (https://github.com/harbor-framework/harbor) is the official evaluation harness for Terminal-Bench 2.0.

Usage:
  # 1. Run live evaluation with Claude 3.7 / GPT-4o on official Terminal-Bench 2.0:
  python benchmarks/harbor/run_harbor_evaluation.py --mode=live --dataset=terminal-bench@2.0 --model=claude-3-7-sonnet

  # 2. Run local integration verification:
  python benchmarks/harbor/run_harbor_evaluation.py --mode=smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.harbor.harbor_adapter import PrimeHarborAgent, PrimeHarborConfig


def run_live_harbor(dataset: str, model: str, arm: str, out_file: str) -> None:
    """Execute official Harbor evaluation via the harbor CLI."""
    cmd = [
        "harbor", "run",
        "--dataset", dataset,
        "--agent", "custom",
        "--agent-config", "benchmarks/harbor/harbor_adapter.py:PrimeHarborAgent",
        "--config", f'{{"ablation_arm": "{arm}", "model_name": "{model}"}}',
        "--out", out_file,
    ]
    print(f"Executing official Harbor benchmark: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True)
        print(f"Harbor run finished with returncode {res.returncode}")
    except FileNotFoundError:
        print("ERROR: Harbor CLI is not installed. Install via `uv tool install harbor` or `pip install harbor`.")
        sys.exit(1)


def run_smoke_verification() -> None:
    """Verify the PrimeHarborAgent adapter protocol locally."""
    print("Running local Harbor adapter integration verification...")
    config = PrimeHarborConfig(ablation_arm="A6")
    agent = PrimeHarborAgent(config=config)
    res = agent.run_task(
        task_instruction="Configure reverse proxy in nginx.conf to 8080",
        workspace_dir="/tmp",
    )
    print("Harbor Agent Adapter Verification Result:", res)
    assert res["status"] in ("COMPLETED", "COMMITTED", "REJECTED_PREFLIGHT")
    print("Harbor adapter verified successfully!")


def main():
    parser = argparse.ArgumentParser(description="Official Terminal-Bench 2.0 Harbor Evaluator")
    parser.add_argument("--mode", choices=["smoke", "live"], default="smoke", help="Execution mode")
    parser.add_argument("--dataset", default="terminal-bench@2.0", help="Harbor dataset selector")
    parser.add_argument("--model", default="claude-3-7-sonnet", help="Foundation model")
    parser.add_argument("--arm", default="A6", help="Ablation arm (A0-A6)")
    parser.add_argument("--out", default="results/harbor_tb2_results.json", help="Output file")
    args = parser.parse_args()

    if args.mode == "live":
        run_live_harbor(dataset=args.dataset, model=args.model, arm=args.arm, out_file=args.out)
    else:
        run_smoke_verification()


if __name__ == "__main__":
    main()
