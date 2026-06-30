"""Run a suite of training experiments sequentially and log summary CSV."""
import os
import sys
import argparse
import subprocess
import json
import csv
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXPERIMENTS = [
    # (config, exp_name)
    ("configs/mappo_grid4x4.yaml", "mappo_grid4x4"),
    ("configs/gatmappo_grid4x4_static.yaml", "gatmappo_grid4x4_static"),
    ("configs/gatmappo_grid4x4_flow.yaml", "gatmappo_grid4x4_flow"),
    ("configs/mappo_hangzhou4x4.yaml", "mappo_hangzhou4x4"),
    ("configs/gatmappo_hangzhou4x4.yaml", "gatmappo_hangzhou4x4_static"),
    ("configs/gatmappo_hangzhou4x4_flow.yaml", "gatmappo_hangzhou4x4_flow"),
]


def run_experiment(config, exp_name, gpus, summary_path):
    cmd = [
        "python", "scripts/train.py",
        "--config", config,
        "--exp_name", exp_name,
        "--gpus", gpus,
    ]
    print(f"\n{'='*60}")
    print(f"Starting: {exp_name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        status = "success"
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {exp_name}: {e}")
        status = "failed"
    elapsed = time.time() - start

    # Extract final eval reward
    result_file = f"results/{exp_name}/eval_results.csv"
    final_reward = None
    final_queue = None
    final_travel = None
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            reader = list(csv.DictReader(f))
            if reader:
                final = reader[-1]
                final_reward = final.get("episode_reward")
                final_queue = final.get("queue_length")
                final_travel = final.get("average_travel_time")

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            exp_name,
            config,
            status,
            f"{elapsed:.1f}",
            final_reward,
            final_queue,
            final_travel,
        ])
    print(f"Finished {exp_name} in {elapsed:.1f}s (status={status})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="2,3")
    parser.add_argument("--summary", type=str, default="results/experiment_suite_summary.csv")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated list of experiment names to run")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "exp_name", "config", "status", "elapsed_sec", "final_reward", "final_queue", "final_travel_time"])

    selected = args.experiments.split(",") if args.experiments else None
    for config, exp_name in EXPERIMENTS:
        if selected and exp_name not in selected:
            continue
        run_experiment(config, exp_name, args.gpus, args.summary)

    print(f"\nSummary saved to {args.summary}")


if __name__ == "__main__":
    main()
