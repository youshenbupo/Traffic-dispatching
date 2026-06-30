"""Plot learning curves across experiments."""
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_training_metrics(exp_dir: Path):
    path = exp_dir / "training_metrics.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["episode"] = int(float(r["episode"]))
                r["episode_reward"] = float(r["episode_reward"])
                if r.get("queue_length"):
                    r["queue_length"] = float(r["queue_length"])
                else:
                    r["queue_length"] = None
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def main(patterns=None):
    if patterns is None:
        exp_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    else:
        exp_dirs = []
        for pat in patterns:
            exp_dirs.extend(RESULTS_DIR.glob(pat))
        exp_dirs = [d for d in exp_dirs if d.is_dir()]

    series_queue = {}
    series_reward = {}
    for d in sorted(exp_dirs):
        rows = load_training_metrics(d)
        train_rows = [r for r in rows if r.get("phase") == "train" and r.get("queue_length") is not None]
        reward_rows = [r for r in rows if r.get("phase") == "train"]
        if train_rows:
            series_queue[d.name] = sorted(train_rows, key=lambda r: r["episode"])
        if reward_rows:
            series_reward[d.name] = sorted(reward_rows, key=lambda r: r["episode"])

    if not series_queue and not series_reward:
        print("No training metrics found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for exp, rows in series_queue.items():
        episodes = [r["episode"] for r in rows]
        queues = [r["queue_length"] for r in rows]
        axes[0].plot(episodes, queues, label=exp, marker="o", markersize=3)

    for exp, rows in series_reward.items():
        episodes = [r["episode"] for r in rows]
        rewards = [r["episode_reward"] for r in rows]
        axes[1].plot(episodes, rewards, label=exp, marker="o", markersize=3)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Queue length")
    axes[0].set_title("Training queue length")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode reward")
    axes[1].set_title("Training reward")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    out_path = RESULTS_DIR / "learning_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    patterns = sys.argv[1:] if len(sys.argv) > 1 else None
    main(patterns)
