"""Trace observed-versus-temporal teacher disagreement and gridlock risk."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from scripts.eval import make_agent, set_gpus
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.trainers.online_trainer import _agent_act
from src.utils.config import load_config, merge_config
from src.utils.metrics import MetricsTracker
from src.risk import semantic_lane_tokens


def temporal_carry_forward(observations, observation_masks, cache):
    """Causally fill missing features with their last observed values."""
    filled = {}
    for aid, value in observations.items():
        mask = observation_masks[aid].astype(bool)
        previous = cache.get(aid, np.zeros_like(value))
        current = np.where(mask, value, previous).astype(
            value.dtype, copy=False
        )
        cache[aid] = current.copy()
        filled[aid] = current
    return filled


def queue_shield_triggered(queue_history, window, threshold):
    """Return true after sustained congestion exceeds the shield threshold."""
    if len(queue_history) < window:
        return False
    return float(np.mean(queue_history[-window:])) > threshold


def policy_actions(
    agent,
    observations,
    masks,
    adjacency,
    observation_masks,
    clean_observations,
    failure_age,
    structure_context,
):
    return _agent_act(
        agent,
        observations,
        masks,
        adjacency,
        explore=False,
        obs_mask=observation_masks,
        clean_obs=clean_observations,
        failure_age=failure_age,
        structure_context=structure_context,
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--control_source",
        choices=("observed", "temporal", "queue_shield"),
        default="observed",
    )
    parser.add_argument("--shield_window", type=int, default=60)
    parser.add_argument("--shield_threshold", type=float, default=0.5)
    parser.add_argument(
        "--include_semantic_tokens",
        action="store_true",
        help="Record observed movement-semantic lane tokens for risk training.",
    )
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpus is not None:
        config = merge_config(
            config, {"training": {"gpus": args.gpus}}
        )
        set_gpus(args.gpus)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    env = SUMOMultiAgentEnv(config)
    env.reset(seed=config.get("seed", 42))
    agent = make_agent(env, config)
    agent.load(args.model_path)
    graph_cfg = config.get("dynamic_graph", {})
    graph_type = graph_cfg.get("graph_type", "static")
    neighbor_k = graph_cfg.get("neighbor_k", 1)
    results = []

    for seed in seeds:
        observations, _ = env.reset(seed=seed)
        temporal_cache = {}
        tracker = MetricsTracker()
        trace = []
        queue_history = []
        shield_active = False
        done = False
        while not done:
            masks = env.get_legal_actions()
            adjacency = env.get_adjacency(
                mode=graph_type, k=neighbor_k
            )
            observation_masks = {
                aid: value.copy()
                for aid, value in env.last_observation_masks.items()
            }
            clean_observations = {
                aid: value.copy()
                for aid, value in env.last_clean_observations.items()
            }
            failure_age = dict(env.last_failure_age)
            structure_context = {
                aid: value.copy()
                for aid, value in env.last_structure_context.items()
            }
            temporal_observations = temporal_carry_forward(
                observations, observation_masks, temporal_cache
            )
            observed_actions = policy_actions(
                agent,
                observations,
                masks,
                adjacency,
                observation_masks,
                clean_observations,
                failure_age,
                structure_context,
            )
            temporal_actions = policy_actions(
                agent,
                temporal_observations,
                masks,
                adjacency,
                observation_masks,
                clean_observations,
                failure_age,
                structure_context,
            )
            failed = [
                aid for aid in env.agent_ids
                if np.min(observation_masks[aid]) < 1.0
            ]
            disagreements = [
                aid for aid in env.agent_ids
                if observed_actions[aid] != temporal_actions[aid]
                and np.asarray(masks[aid]).sum() > 1
            ]
            if args.control_source == "queue_shield":
                shield_active = shield_active or queue_shield_triggered(
                    queue_history,
                    args.shield_window,
                    args.shield_threshold,
                )
                actions = (
                    temporal_actions if shield_active else observed_actions
                )
            elif args.control_source == "temporal":
                actions = temporal_actions
            else:
                actions = observed_actions
            next_observations, rewards, terminated, truncated, _ = (
                env.step(actions)
            )
            step_metrics = MetricsTracker.compute_from_env(env)
            tracker.record_step({
                "reward": sum(rewards.values()),
                **step_metrics,
            })
            queue_history.append(step_metrics["queue_length"])
            trace.append({
                "step": len(trace),
                "failed_agents": failed,
                "disagreement_agents": disagreements,
                "queue_length": step_metrics["queue_length"],
                "active_vehicles": step_metrics["active_vehicles"],
                "throughput": step_metrics["throughput"],
                "shield_active": shield_active,
                **({
                    "semantic_tokens": {
                        aid: semantic_lane_tokens(
                            observations[aid],
                            observation_masks[aid],
                            env.phase_lane_matrix[aid],
                            structure_context[aid][
                                :env.max_incoming_lanes
                            ],
                        ).tolist()
                        for aid in env.agent_ids
                    }
                } if args.include_semantic_tokens else {}),
            })
            observations = next_observations
            done = terminated or truncated

        tracker.finalize(env.get_arrived_vehicle_info())
        metrics = tracker.aggregate()
        results.append({
            "seed": seed,
            "control_source": args.control_source,
            "metrics": metrics,
            "trace": trace,
        })
        print(
            f"seed={seed} source={args.control_source} "
            f"total_time_spent={metrics['total_time_spent']:.1f}"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump({"episodes": results}, stream, indent=2)
    env.close()


if __name__ == "__main__":
    main()
