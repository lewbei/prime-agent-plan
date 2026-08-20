#!/usr/bin/env python3
"""Sequential runner executing all 89 official Terminal-Bench 2.0 tasks with automatic storage safety."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATASET_DIR = Path(_ROOT) / "benchmarks" / "harbor_dataset" / "terminal-bench"


def get_available_disk_gb() -> float:
    """Return free disk space in GB on root mount."""
    stat = shutil.disk_usage("/")
    return stat.free / (1024 ** 3)


def prune_docker_storage() -> None:
    """Prune stopped containers and unused images to preserve disk headroom."""
    try:
        subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
        subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
    except Exception:
        pass


def run_all_89_tasks(arm: str = "A6", model: str = "vertex_ai/gemini-2.0-flash", provider: str = "vertex_ai", max_tasks: int | None = None, out_dir: str = "results") -> None:
    task_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    if max_tasks:
        task_dirs = task_dirs[:max_tasks]

    total_tasks = len(task_dirs)
    print("=" * 75)
    print(f"STARTING SEQUENTIAL EVALUATION OF OFFICIAL TERMINAL-BENCH 2.0 ({total_tasks} TASKS)")
    print(f"Agent Arm: {arm} | Model: {model} | Free Disk: {get_available_disk_gb():.1f} GB")
    print("=" * 75)

    os.makedirs(out_dir, exist_ok=True)
    results_records: List[Dict[str, Any]] = []

    for i, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        free_gb = get_available_disk_gb()
        if free_gb < 20.0:
            print(f"Low disk warning ({free_gb:.1f} GB free). Pruning Docker cache...")
            prune_docker_storage()

        print(f"[{i:02d}/{total_tasks:02d}] Executing Task: {task_name} (Disk Free: {get_available_disk_gb():.1f} GB) ...")
        t0 = time.time()

        cmd = [
            "harbor", "run",
            "--path", str(task_dir),
            "--agent", "benchmarks.harbor.harbor_adapter:PrimeHarborAgent",
            "--agent-kwarg", f"ablation_arm={arm}",
            "--agent-kwarg", f"provider={provider}",
            "--model", model,
            "--n-attempts", "1",
            "--n-concurrent", "1",
            "--delete",
            "--yes",
            "--quiet",
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            dur = time.time() - t0
            passed = ("Mean: 1.000" in res.stdout) or ("1.0" in res.stdout and "Reward" in res.stdout)
            print(f"       -> Finished in {dur:.1f}s | Result: {'PASS' if passed else 'FAIL'}")
            results_records.append({
                "task_name": task_name,
                "passed": passed,
                "duration_sec": dur,
                "return_code": res.returncode,
            })
        except subprocess.TimeoutExpired:
            print(f"       -> TIMEOUT after 300s")
            results_records.append({"task_name": task_name, "passed": False, "duration_sec": 300.0, "return_code": -1})
        except Exception as ex:
            print(f"       -> ERROR: {ex}")
            results_records.append({"task_name": task_name, "passed": False, "duration_sec": 0.0, "return_code": -2})

        # Regular lightweight cleanup between tasks
        if i % 5 == 0:
            prune_docker_storage()

    # Generate Report
    report_file = os.path.join(out_dir, "OFFICIAL_TERMINAL_BENCH_2_0_EVALUATION.md")
    passed_count = sum(1 for r in results_records if r["passed"])
    pass_rate = (passed_count / total_tasks * 100.0) if total_tasks else 0.0

    lines = [
        "# Official Terminal-Bench 2.0 Empirical Evaluation Report",
        "",
        "> **Benchmark**: Terminal-Bench 2.0 (*arXiv:2601.11868*)  ",
        f"> **Evaluation Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"> **Harness**: Harbor Framework (`harbor-framework/terminal-bench-2`)  ",
        f"> **Agent Configuration**: `PrimeHarborAgent` (Arm: `{arm}`, Model: `{model}`)  ",
        f"> **Tasks Evaluated**: {total_tasks} / 89 Official Containerized Tasks  ",
        "",
        "---",
        "",
        "## Summary Metrics",
        "",
        f"- **Total Tasks**: {total_tasks}",
        f"- **Passed Tasks**: {passed_count}",
        f"- **Pass Rate**: **{pass_rate:.1f}%**",
        f"- **Mean Task Duration**: {sum(r['duration_sec'] for r in results_records)/total_tasks:.1f}s",
        "",
        "---",
        "",
        "## Per-Task Results Table",
        "",
        "| # | Task Name | Status | Duration (s) |",
        "| :---: | :--- | :---: | :---: |",
    ]

    for idx, r in enumerate(results_records, 1):
        st = "**PASS**" if r["passed"] else "FAIL"
        lines.append(f"| {idx:02d} | `{r['task_name']}` | {st} | {r['duration_sec']:.1f}s |")

    with open(report_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    json_file = os.path.join(out_dir, "official_terminal_bench_results.json")
    with open(json_file, "w") as f:
        json.dump({
            "total_tasks": total_tasks,
            "passed_tasks": passed_count,
            "pass_rate": pass_rate,
            "arm": arm,
            "model": model,
            "results": results_records,
        }, f, indent=2)

    print("=" * 75)
    print(f"ALL {total_tasks} TASKS EVALUATION COMPLETE!")
    print(f"Overall Pass Rate: {passed_count}/{total_tasks} ({pass_rate:.1f}%)")
    print(f"Report Written To: {report_file}")
    print(f"JSON Results To: {json_file}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Official Terminal-Bench 2.0 Sequential Evaluator")
    parser.add_argument("--arm", default="A6", help="Ablation arm (A0-A6)")
    parser.add_argument("--model", default="vertex_ai/gemini-2.0-flash", help="Model name (e.g. vertex_ai/gemini-2.0-flash, claude-3-7-sonnet)")
    parser.add_argument("--provider", default="vertex_ai", help="Provider (vertex_ai/gemini/anthropic/openai)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    run_all_89_tasks(arm=args.arm, model=args.model, provider=args.provider, max_tasks=args.limit, out_dir=args.out)


if __name__ == "__main__":
    main()
