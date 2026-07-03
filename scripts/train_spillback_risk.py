"""Train the semantic temporal spillback-risk predictor."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.risk import SemanticSpillbackPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation_seeds",
        default=None,
        help=(
            "Comma-separated episode seeds held out for validation. "
            "Defaults to the last seed in the dataset."
        ),
    )
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = np.load(args.dataset)
    sequences = torch.from_numpy(data["sequences"])
    failure_masks = torch.from_numpy(data["failure_masks"])
    labels = torch.from_numpy(data["labels"][:, 0])
    adjacency_array = (
        data["adjacency"]
        if "adjacency" in data.files
        else np.ones(
            (sequences.shape[2], sequences.shape[2]),
            dtype=np.float32,
        )
    )
    adjacency = torch.from_numpy(adjacency_array).to(device)
    unique_seeds = np.unique(data["seeds"])
    if args.validation_seeds:
        validation_seeds = np.asarray([
            int(item)
            for item in args.validation_seeds.split(",")
            if item.strip()
        ], dtype=np.int64)
    else:
        validation_seeds = np.asarray([unique_seeds[-1]], dtype=np.int64)
    train_indices = np.flatnonzero(
        ~np.isin(data["seeds"], validation_seeds)
    )
    validation_indices = np.flatnonzero(
        np.isin(data["seeds"], validation_seeds)
    )
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("Dataset needs at least two episode seeds")

    model = SemanticSpillbackPredictor(
        token_dim=sequences.shape[-1],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    positives = labels[train_indices].sum().item()
    negatives = len(train_indices) - positives
    pos_weight = torch.tensor(
        negatives / max(positives, 1.0), device=device
    )

    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = train_indices[
            torch.randperm(
                len(train_indices), generator=generator
            ).numpy()
        ]
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
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
            val_logits, _ = model(
                sequences[validation_indices].to(device),
                failure_masks[validation_indices].to(device),
                adjacency,
            )
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits,
                labels[validation_indices].to(device),
            ).item()
            val_probability = torch.sigmoid(val_logits)
        print(
            f"epoch={epoch + 1} train_loss={np.mean(losses):.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_probability_mean={val_probability.mean().item():.4f}"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "token_dim": sequences.shape[-1],
        "hidden_dim": args.hidden_dim,
        "validation_seed": int(validation_seeds[-1]),
        "validation_seeds": validation_seeds.tolist(),
        "agent_order": data["agent_order"].tolist(),
    }, args.output)


if __name__ == "__main__":
    main()
