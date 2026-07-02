"""Build causal semantic sequences with privileged future-spillback labels."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.risk import future_spillback_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    episodes = []
    adjacency = None
    for path in sorted(set(paths)):
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        episodes.extend(payload["episodes"])
        candidate = payload.get("adjacency")
        if candidate is not None:
            candidate = np.asarray(candidate, dtype=np.float32)
            if adjacency is None:
                adjacency = candidate
            elif not np.array_equal(candidate, adjacency):
                raise ValueError("Adjacency differs across input traces")
    if not episodes:
        raise ValueError("No risk-probe episodes found")

    sequences = []
    masks = []
    labels = []
    seeds = []
    steps = []
    agent_order = None
    for episode in episodes:
        trace = episode["trace"]
        if not trace or "semantic_tokens" not in trace[0]:
            raise ValueError(
                "Input trace lacks semantic tokens; rerun the probe with "
                "--include_semantic_tokens"
            )
        current_agents = sorted(trace[0]["semantic_tokens"])
        if agent_order is None:
            agent_order = current_agents
        elif current_agents != agent_order:
            raise ValueError("Agent ordering differs across input traces")
        queue = [item["queue_length"] for item in trace]
        active = [item["active_vehicles"] for item in trace]
        throughput = [item["throughput"] for item in trace]
        targets = future_spillback_targets(
            queue, active, throughput, horizon=args.horizon
        )
        token_series = np.asarray([
            [item["semantic_tokens"][aid] for aid in agent_order]
            for item in trace
        ], dtype=np.float32)
        failure_series = np.asarray([
            [float(aid in item["failed_agents"]) for aid in agent_order]
            for item in trace
        ], dtype=np.float32)
        for step in range(args.history - 1, len(trace), args.stride):
            if targets["valid"][step] == 0.0:
                continue
            start = step - args.history + 1
            sequences.append(token_series[start:step + 1])
            masks.append(failure_series[start:step + 1])
            labels.append([
                targets["spillback_risk"][step],
                targets["future_queue_mean"][step],
                targets["future_queue_peak"][step],
                targets["future_active_growth"][step],
                targets["future_departures"][step],
            ])
            seeds.append(episode["seed"])
            steps.append(step)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        sequences=np.asarray(sequences, dtype=np.float32),
        failure_masks=np.asarray(masks, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
        agent_order=np.asarray(agent_order),
        label_order=np.asarray([
            "spillback_risk",
            "future_queue_mean",
            "future_queue_peak",
            "future_active_growth",
            "future_departures",
        ]),
        adjacency=(
            adjacency
            if adjacency is not None
            else np.ones(
                (len(agent_order), len(agent_order)), dtype=np.float32
            )
        ),
    )
    positive_rate = float(np.mean(np.asarray(labels)[:, 0]))
    print(
        f"samples={len(labels)} positive_rate={positive_rate:.4f} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
