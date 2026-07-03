"""Build a small counterfactual intervention-benefit dataset.

Inputs:
- observed semantic traces from ``probe_dual_policy_risk.py`` with
  ``--include_semantic_tokens``;
- fixed-step intervention outcome JSON files produced with
  ``--control_source temporal_after_step``.

Each sample asks: given the observed semantic history before a proposed switch
step, would switching to temporal fallback be beneficial?
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


STEP_RE = re.compile(r"(?:start_|start_?step_|start)(\d+)")


def parse_start_step(path, payload):
    for episode in payload["episodes"]:
        trace = episode.get("trace", [])
        if trace and "temporal_start_step" in trace[0]:
            return int(trace[0]["temporal_start_step"])
    match = STEP_RE.search(os.path.basename(path))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer temporal start step from {path}")


def load_semantic_traces(patterns):
    traces = {}
    adjacency = None
    agent_order = None
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
            if adjacency is None and payload.get("adjacency") is not None:
                adjacency = np.asarray(payload["adjacency"], dtype=np.float32)
            if agent_order is None and payload.get("agent_order") is not None:
                agent_order = list(payload["agent_order"])
            for episode in payload["episodes"]:
                trace = episode["trace"]
                if not trace or "semantic_tokens" not in trace[0]:
                    continue
                traces[int(episode["seed"])] = trace
    if not traces:
        raise ValueError("No semantic traces found")
    return traces, agent_order, adjacency


def load_interventions(patterns):
    interventions = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
            start_step = parse_start_step(path, payload)
            for episode in payload["episodes"]:
                interventions.append({
                    "path": path,
                    "seed": int(episode["seed"]),
                    "start_step": start_step,
                    "total_time_spent": float(
                        episode["metrics"]["total_time_spent"]
                    ),
                    "queue_length": float(episode["metrics"]["queue_length"]),
                    "throughput": float(episode["metrics"]["throughput"]),
                })
    if not interventions:
        raise ValueError("No intervention outcomes found")
    return interventions


CONTEXT_FEATURE_NAMES = [
    "failure_fraction_current",
    "failure_fraction_history",
    "disagreement_fraction_current",
    "disagreement_fraction_history",
    "queue_mean_current",
    "queue_max_current",
    "active_service_queue_mean",
    "inactive_service_queue_mean",
    "active_service_fraction",
    "unserved_queue_pressure",
    "queue_mean_growth",
]


def masked_mean(values, mask):
    total = float(mask.sum())
    if total <= 0.0:
        return 0.0
    return float((values * mask).sum() / total)


def build_context_features(trace, begin, end, agent_order):
    """Summarize switch-state compatibility cues at the candidate step."""
    window = trace[begin:end]
    current = window[-1]
    agent_count = max(len(agent_order), 1)

    current_failed = len(current.get("failed_agents", [])) / agent_count
    history_failed = float(np.mean([
        len(step.get("failed_agents", [])) / agent_count
        for step in window
    ]))
    current_disagreement = (
        len(current.get("disagreement_agents", [])) / agent_count
    )
    history_disagreement = float(np.mean([
        len(step.get("disagreement_agents", [])) / agent_count
        for step in window
    ]))

    current_tokens = np.asarray([
        current["semantic_tokens"][aid] for aid in agent_order
    ], dtype=np.float32)
    first_tokens = np.asarray([
        window[0]["semantic_tokens"][aid] for aid in agent_order
    ], dtype=np.float32)
    queue = current_tokens[..., 0]
    first_queue = first_tokens[..., 0]
    lane_presence = current_tokens[..., -1].clip(0.0, 1.0)
    active_service = current_tokens[..., -2].clip(0.0, 1.0)
    active_mask = lane_presence * (active_service > 0.5)
    inactive_mask = lane_presence * (active_service <= 0.5)

    queue_mean = masked_mean(queue, lane_presence)
    queue_max = float((queue * lane_presence).max())
    active_queue = masked_mean(queue, active_mask)
    inactive_queue = masked_mean(queue, inactive_mask)
    active_fraction = masked_mean(active_service, lane_presence)
    first_queue_mean = masked_mean(first_queue, lane_presence)
    return np.asarray([
        current_failed,
        history_failed,
        current_disagreement,
        history_disagreement,
        queue_mean,
        queue_max,
        active_queue,
        inactive_queue,
        active_fraction,
        inactive_queue - active_queue,
        queue_mean - first_queue_mean,
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_inputs", nargs="+", required=True)
    parser.add_argument("--intervention_inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=60)
    parser.add_argument(
        "--observed_totals",
        required=True,
        help="Comma-separated seed:total mapping, e.g. 62000:365255,...",
    )
    parser.add_argument(
        "--rescue_threshold",
        type=float,
        default=250000.0,
        help="Total-time threshold for a clearly rescued outcome.",
    )
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.0,
        help="Required total-time improvement over observed baseline.",
    )
    args = parser.parse_args()

    observed_totals = {}
    for item in args.observed_totals.split(","):
        seed, total = item.split(":")
        observed_totals[int(seed)] = float(total)

    traces, agent_order, adjacency = load_semantic_traces(
        args.semantic_inputs
    )
    interventions = load_interventions(args.intervention_inputs)

    sequences = []
    failure_masks = []
    labels = []
    benefits = []
    seeds = []
    start_steps = []
    totals = []
    queues = []
    throughputs = []
    context_features = []
    paths = []
    seen = set()

    for item in interventions:
        seed = item["seed"]
        start_step = item["start_step"]
        key = (seed, start_step)
        if key in seen:
            continue
        seen.add(key)
        if seed not in observed_totals or seed not in traces:
            continue
        trace = traces[seed]
        # Need a causal window ending immediately before the intervention.
        end = start_step
        begin = end - args.history
        if begin < 0 or end > len(trace):
            continue
        current_agents = sorted(trace[0]["semantic_tokens"])
        if agent_order is None:
            agent_order = current_agents
        token_series = np.asarray([
            [step["semantic_tokens"][aid] for aid in agent_order]
            for step in trace[begin:end]
        ], dtype=np.float32)
        failure_series = np.asarray([
            [float(aid in step["failed_agents"]) for aid in agent_order]
            for step in trace[begin:end]
        ], dtype=np.float32)
        observed_total = observed_totals[seed]
        benefit = observed_total - item["total_time_spent"]
        rescued = (
            item["total_time_spent"] <= args.rescue_threshold
            and benefit > args.min_improvement
        )
        sequences.append(token_series)
        failure_masks.append(failure_series)
        labels.append(float(rescued))
        benefits.append(float(benefit))
        seeds.append(seed)
        start_steps.append(start_step)
        totals.append(item["total_time_spent"])
        queues.append(item["queue_length"])
        throughputs.append(item["throughput"])
        context_features.append(
            build_context_features(trace, begin, end, agent_order)
        )
        paths.append(os.path.basename(item["path"]))

    if not sequences:
        raise ValueError("No usable intervention-benefit samples built")

    if adjacency is None:
        adjacency = np.ones(
            (len(agent_order), len(agent_order)), dtype=np.float32
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        sequences=np.asarray(sequences, dtype=np.float32),
        failure_masks=np.asarray(failure_masks, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        benefits=np.asarray(benefits, dtype=np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
        start_steps=np.asarray(start_steps, dtype=np.int64),
        total_time_spent=np.asarray(totals, dtype=np.float32),
        queue_length=np.asarray(queues, dtype=np.float32),
        throughput=np.asarray(throughputs, dtype=np.float32),
        context_features=np.asarray(context_features, dtype=np.float32),
        context_feature_names=np.asarray(CONTEXT_FEATURE_NAMES),
        source_files=np.asarray(paths),
        agent_order=np.asarray(agent_order),
        adjacency=adjacency,
    )
    print(
        f"samples={len(labels)} "
        f"positive_rate={float(np.mean(labels)):.4f} "
        f"output={args.output}"
    )
    for seed in sorted(set(seeds)):
        indices = [i for i, value in enumerate(seeds) if value == seed]
        print(
            f"seed={seed} samples={len(indices)} "
            f"positives={int(np.sum(np.asarray(labels)[indices]))} "
            f"steps={[int(start_steps[i]) for i in indices]}"
        )


if __name__ == "__main__":
    main()
