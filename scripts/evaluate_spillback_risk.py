"""Evaluate a trained semantic spillback-risk predictor.

The training script is intentionally small; this companion script turns a
checkpoint and an `.npz` dataset into paper-facing diagnostics: ranking metrics,
calibration error, threshold quality, and per-seed early-warning lead time.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.risk import SemanticSpillbackPredictor


def roc_auc_score(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1.0
    negatives = labels == 0.0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[positives].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
    precision = true_positives / ranks
    return float((precision * sorted_labels).sum() / n_pos)


def threshold_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(scores) >= threshold
    tp = float(((predictions == 1) & (labels == 1)).sum())
    fp = float(((predictions == 1) & (labels == 0)).sum())
    fn = float(((predictions == 0) & (labels == 1)).sum())
    tn = float(((predictions == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(labels, scores):
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    if len(candidates) == 0:
        return threshold_metrics(labels, scores, 0.5)
    best = threshold_metrics(labels, scores, float(candidates[0]))
    for threshold in candidates[1:]:
        current = threshold_metrics(labels, scores, float(threshold))
        if current["f1"] > best["f1"]:
            best = current
    return best


def per_seed_lead_times(seeds, steps, labels, scores, threshold):
    rows = []
    for seed in sorted(set(int(item) for item in seeds.tolist())):
        indices = np.flatnonzero(seeds == seed)
        seed_steps = steps[indices]
        seed_labels = labels[indices]
        seed_scores = scores[indices]
        positives = seed_steps[seed_labels == 1.0]
        alerts = seed_steps[seed_scores >= threshold]
        first_positive = int(positives.min()) if len(positives) else None
        first_alert = int(alerts.min()) if len(alerts) else None
        rows.append({
            "seed": seed,
            "positive_samples": int(seed_labels.sum()),
            "first_positive_step": first_positive,
            "first_alert_step": first_alert,
            "lead_steps_to_first_positive": (
                None
                if first_positive is None or first_alert is None
                else int(first_positive - first_alert)
            ),
            "false_alert_only": bool(first_positive is None and first_alert is not None),
            "missed_positive_seed": bool(first_positive is not None and first_alert is None),
            "max_probability": float(seed_scores.max()) if len(seed_scores) else math.nan,
        })
    return rows


def parse_seed_list(text):
    if not text:
        return None
    return [int(item) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval_seeds", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset)
    checkpoint = torch.load(args.model, map_location=device)
    sequences = torch.from_numpy(data["sequences"])
    failure_masks = torch.from_numpy(data["failure_masks"])
    labels = data["labels"][:, 0].astype(np.float32)
    seeds = data["seeds"].astype(np.int64)
    steps = data["steps"].astype(np.int64)
    adjacency = torch.from_numpy(data["adjacency"].astype(np.float32)).to(device)

    eval_seeds = parse_seed_list(args.eval_seeds)
    if eval_seeds is None:
        if "validation_seeds" in checkpoint:
            eval_seeds = [int(seed) for seed in checkpoint["validation_seeds"]]
        else:
            eval_seeds = [
                int(checkpoint.get("validation_seed", np.unique(seeds)[-1]))
            ]
    eval_indices = np.flatnonzero(np.isin(seeds, np.asarray(eval_seeds)))
    if len(eval_indices) == 0:
        raise ValueError(f"No samples found for eval seeds: {eval_seeds}")

    model = SemanticSpillbackPredictor(
        token_dim=int(checkpoint["token_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(eval_indices), args.batch_size):
            indices = eval_indices[start:start + args.batch_size]
            logits, _ = model(
                sequences[indices].to(device),
                failure_masks[indices].to(device),
                adjacency,
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    eval_labels = labels[eval_indices]
    eval_seeds_array = seeds[eval_indices]
    eval_steps = steps[eval_indices]

    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1 - eps)
    loss = F.binary_cross_entropy(
        torch.from_numpy(clipped.astype(np.float32)),
        torch.from_numpy(eval_labels.astype(np.float32)),
    ).item()
    selected_threshold = threshold_metrics(
        eval_labels, probabilities, args.threshold
    )
    best_threshold = best_f1_threshold(eval_labels, probabilities)
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "samples": int(len(eval_labels)),
        "positive_rate": float(eval_labels.mean()) if len(eval_labels) else math.nan,
        "binary_cross_entropy": float(loss),
        "roc_auc": roc_auc_score(eval_labels, probabilities),
        "average_precision": average_precision_score(eval_labels, probabilities),
        "brier": float(np.mean((probabilities - eval_labels) ** 2)),
        "probability_mean": float(probabilities.mean()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "selected_threshold": selected_threshold,
        "best_f1_threshold": best_threshold,
        "per_seed": per_seed_lead_times(
            eval_seeds_array,
            eval_steps,
            eval_labels,
            probabilities,
            selected_threshold["threshold"],
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
