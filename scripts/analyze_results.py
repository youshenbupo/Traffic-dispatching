"""Analyze experiment results and print comparison table."""
import os
import sys
import json
import csv
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v
                         for k, v in r.items()})
    return rows


def analyze_experiment(exp_dir):
    """Analyze from training_metrics.csv (contains both train and eval rows)."""
    metrics_csv = os.path.join(exp_dir, "training_metrics.csv")
    if not os.path.exists(metrics_csv):
        return None
    rows = load_csv(metrics_csv)
    eval_rows = [r for r in rows if r.get("phase") == "eval"]
    if not eval_rows:
        return None
    first = eval_rows[0]
    last = eval_rows[-1]
    best = min(eval_rows, key=lambda r: r.get("queue_length", float("inf")))
    return {
        "episodes": len(eval_rows),
        "initial_reward": first.get("episode_reward"),
        "final_reward": last.get("episode_reward"),
        "best_queue": best.get("queue_length"),
        "final_queue": last.get("queue_length"),
        "final_travel": last.get("average_travel_time"),
        "final_wait": last.get("average_waiting_time"),
        "final_throughput": last.get("throughput"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--baseline_dir", type=str, default="results")
    parser.add_argument("--filter", type=str, default=None, help="Substring filter for experiment names")
    args = parser.parse_args()

    result_path = Path(args.result_dir)
    experiments = sorted([d.name for d in result_path.iterdir() if d.is_dir() and (d / "training_metrics.csv").exists()])
    if args.filter:
        experiments = [e for e in experiments if args.filter in e]

    print("\n" + "="*110)
    print(f"{'Experiment':<38} {'Init Reward':>12} {'Final Reward':>13} {'Final Queue':>12} {'Best Queue':>11} {'Travel':>10} {'Wait':>10}")
    print("="*110)
    for exp in experiments:
        exp_dir = os.path.join(args.result_dir, exp)
        stats = analyze_experiment(exp_dir)
        if stats:
            print(f"{exp:<38} {stats['initial_reward']:>12.1f} {stats['final_reward']:>13.1f} {stats['final_queue']:>12.2f} {stats['best_queue']:>11.2f} {stats['final_travel']:>10.1f} {stats['final_wait']:>10.1f}")
        else:
            print(f"{exp:<38} {'N/A':>12} {'N/A':>13} {'N/A':>12} {'N/A':>11} {'N/A':>10} {'N/A':>10}")
    print("="*110)

    baseline_files = [
        ("MaxPressure grid4x4 (500)", "maxpressure_grid4x4_500.json"),
        ("MaxPressure hangzhou4x4 (200)", "maxpressure_hangzhou4x4_200.json"),
    ]
    print("\nBaselines:")
    for name, fname in baseline_files:
        path = os.path.join(args.baseline_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"  {name}: queue={data.get('queue_length', 'N/A'):.2f}, reward={data.get('episode_reward', 'N/A'):.1f}")


if __name__ == "__main__":
    main()
