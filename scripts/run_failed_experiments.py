"""Re-run experiments that failed in the first batch."""
import os
import sys
import argparse
import subprocess
import time
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXPERIMENTS = [
    ("configs/gatmappo_grid4x4_static.yaml", "gatmappo_grid4x4_static"),
    ("configs/gatmappo_grid4x4_flow.yaml", "gatmappo_grid4x4_flow"),
    ("configs/mappo_hangzhou4x4.yaml", "mappo_hangzhou4x4"),
    ("configs/gatmappo_hangzhou4x4.yaml", "gatmappo_hangzhou4x4_static"),
    ("configs/gatmappo_hangzhou4x4_flow.yaml", "gatmappo_hangzhou4x4_flow"),
]


def run_experiment(config, exp_name, gpus, summary_path):
    # Skip if already has eval results
    result_file = f"results/{exp_name}/eval_results.csv"
    if os.path.exists(result_file) and os.path.getsize(result_file) > 100:
        print(f"Skipping {exp_name} (already has results)")
        return

    # Clean old partial logs
    for d in [f"logs/{exp_name}", f"results/{exp_name}"]:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)

    cmd = ["python", "scripts/train.py", "--config", config, "--exp_name", exp_name, "--gpus", gpus]
    print(f"\nStarting: {exp_name}")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        status = "success"
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {exp_name}: {e}")
        status = "failed"
    elapsed = time.time() - start

    final_reward, final_queue, final_travel = None, None, None
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                final_reward = last.get("episode_reward")
                final_queue = last.get("queue_length")
                final_travel = last.get("average_travel_time")

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), exp_name, config, status, f"{elapsed:.1f}", final_reward, final_queue, final_travel])
    print(f"Finished {exp_name} in {elapsed:.1f}s (status={status})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="2,3")
    parser.add_argument("--summary", type=str, default="results/experiment_suite_summary.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    if not os.path.exists(args.summary):
        with open(args.summary, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "exp_name", "config", "status", "elapsed_sec", "final_reward", "final_queue", "final_travel_time"])

    for config, exp_name in EXPERIMENTS:
        run_experiment(config, exp_name, args.gpus, args.summary)


if __name__ == "__main__":
    main()
