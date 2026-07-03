"""Train a lightweight counterfactual intervention-benefit classifier."""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.risk import (
    GraphInterventionBenefitPredictor,
    SemanticSpillbackPredictor,
)


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
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_score(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def threshold_f1(labels, scores, threshold=0.5):
    predictions = scores >= threshold
    labels = labels.astype(bool)
    tp = float((predictions & labels).sum())
    fp = float((predictions & ~labels).sum())
    fn = float((~predictions & labels).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def parse_seeds(text):
    return np.asarray([
        int(item) for item in text.split(",") if item.strip()
    ], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation_seeds", required=True)
    parser.add_argument(
        "--model_type",
        choices=("temporal", "graph"),
        default="temporal",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset)
    sequences = torch.from_numpy(data["sequences"])
    failure_masks = torch.from_numpy(data["failure_masks"])
    labels = torch.from_numpy(data["labels"].astype(np.float32))
    adjacency = torch.from_numpy(data["adjacency"].astype(np.float32)).to(device)
    start_steps = torch.from_numpy(data["start_steps"].astype(np.float32))
    validation_seeds = parse_seeds(args.validation_seeds)
    train_indices = np.flatnonzero(~np.isin(data["seeds"], validation_seeds))
    validation_indices = np.flatnonzero(np.isin(data["seeds"], validation_seeds))
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("Need non-empty train and validation splits")

    if args.model_type == "graph":
        model = GraphInterventionBenefitPredictor(
            token_dim=sequences.shape[-1],
            hidden_dim=args.hidden_dim,
        ).to(device)
    else:
        model = SemanticSpillbackPredictor(
            token_dim=sequences.shape[-1],
            hidden_dim=args.hidden_dim,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    positives = labels[train_indices].sum().item()
    negatives = len(train_indices) - positives
    pos_weight = torch.tensor(
        negatives / max(positives, 1.0), device=device
    )
    generator = torch.Generator().manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        order = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            if args.model_type == "graph":
                logits = model(
                    sequences[indices].to(device),
                    failure_masks[indices].to(device),
                    adjacency,
                    start_steps[indices].to(device),
                )
            else:
                logits, _ = model(
                    sequences[indices].to(device),
                    failure_masks[indices].to(device),
                    adjacency,
                )
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels[indices].to(device),
                pos_weight=pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if args.model_type == "graph":
                val_logits = model(
                    sequences[validation_indices].to(device),
                    failure_masks[validation_indices].to(device),
                    adjacency,
                    start_steps[validation_indices].to(device),
                )
            else:
                val_logits, _ = model(
                    sequences[validation_indices].to(device),
                    failure_masks[validation_indices].to(device),
                    adjacency,
                )
            probabilities = torch.sigmoid(val_logits).cpu().numpy()
            val_labels = labels[validation_indices].numpy()
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits,
                labels[validation_indices].to(device),
            ).item()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            precision, recall, f1 = threshold_f1(val_labels, probabilities)
            print(
                f"epoch={epoch + 1} train_loss={np.mean(losses):.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_auc={roc_auc_score(val_labels, probabilities):.4f} "
                f"val_ap={average_precision_score(val_labels, probabilities):.4f} "
                f"val_f1@0.5={f1:.4f} "
                f"val_prob_mean={probabilities.mean():.4f}"
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "token_dim": sequences.shape[-1],
        "hidden_dim": args.hidden_dim,
        "model_type": args.model_type,
        "validation_seeds": validation_seeds.tolist(),
        "agent_order": data["agent_order"].tolist(),
    }, args.output)


if __name__ == "__main__":
    main()
