"""Counterfactual probes for observation responsiveness of trained policies."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval import make_agent, set_gpus
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.trainers.online_trainer import _agent_act
from src.utils.config import load_config, merge_config


def policy_actions(agent, observations, masks, adjacency, observation_masks):
    return _agent_act(
        agent,
        observations,
        masks,
        adjacency,
        explore=False,
        obs_mask=observation_masks,
        clean_obs=observations,
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=35000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpus is not None:
        config = merge_config(config, {"training": {"gpus": args.gpus}})
        set_gpus(args.gpus)
    env = SUMOMultiAgentEnv(config)
    env.reset(seed=config.get("seed", 42))
    agent = make_agent(env, config)
    agent.load(args.model_path)

    dropout_flips = 0
    dropout_trials = 0
    congestion_flips = 0
    congestion_trials = 0
    action_changes = 0
    action_change_trials = 0
    action_counts = {
        aid: np.zeros(len(env.phases[aid]), dtype=np.int64)
        for aid in env.agent_ids
    }
    previous_actions = {}

    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed_start + episode)
        done = False
        while not done:
            masks = env.get_legal_actions()
            adjacency = env.get_adjacency(
                mode=config.get("dynamic_graph", {}).get(
                    "graph_type", "static"
                ),
                k=config.get("dynamic_graph", {}).get("neighbor_k", 1),
            )
            clean_obs = {
                aid: value.copy()
                for aid, value in env.last_clean_observations.items()
            }
            clean_masks = {
                aid: np.ones_like(value, dtype=np.float32)
                for aid, value in clean_obs.items()
            }
            base_actions = policy_actions(
                agent, clean_obs, masks, adjacency, clean_masks
            )

            eligible = {
                aid for aid in env.agent_ids if np.asarray(masks[aid]).sum() > 1
            }
            for aid, action in base_actions.items():
                action_counts[aid][action] += 1
                if aid in eligible and aid in previous_actions:
                    action_changes += int(action != previous_actions[aid])
                    action_change_trials += 1
                previous_actions[aid] = action

            zero_obs = {aid: value.copy() for aid, value in clean_obs.items()}
            zero_masks = {
                aid: value.copy() for aid, value in clean_masks.items()
            }
            for aid in env.agent_ids:
                n_lanes = len(env.incoming_lanes[aid])
                zero_obs[aid][:3 * n_lanes] = 0.0
                zero_masks[aid][:3 * n_lanes] = 0.0
            zero_actions = policy_actions(
                agent, zero_obs, masks, adjacency, zero_masks
            )
            for aid in eligible:
                dropout_flips += int(zero_actions[aid] != base_actions[aid])
                dropout_trials += 1

            n_lanes = min(len(env.incoming_lanes[aid]) for aid in env.agent_ids)
            for lane_idx in range(n_lanes):
                stressed = {
                    aid: value.copy() for aid, value in clean_obs.items()
                }
                for aid in env.agent_ids:
                    local_lanes = len(env.incoming_lanes[aid])
                    stressed[aid][lane_idx] += 10.0
                    stressed[aid][local_lanes + lane_idx] += 100.0
                    stressed[aid][2 * local_lanes + lane_idx] += 10.0
                stressed_actions = policy_actions(
                    agent, stressed, masks, adjacency, clean_masks
                )
                for aid in eligible:
                    congestion_flips += int(
                        stressed_actions[aid] != base_actions[aid]
                    )
                    congestion_trials += 1

            next_obs, _, terminated, truncated, _ = env.step(base_actions)
            done = terminated or truncated
            obs = next_obs

    entropies = []
    for counts in action_counts.values():
        total = counts.sum()
        if total == 0 or len(counts) <= 1:
            continue
        probabilities = counts[counts > 0] / total
        entropy = -float(
            np.sum(probabilities * np.log(probabilities))
        ) / math.log(len(counts))
        entropies.append(entropy)
    result = {
        "dropout_action_flip_rate": dropout_flips / max(dropout_trials, 1),
        "congestion_probe_flip_rate": (
            congestion_flips / max(congestion_trials, 1)
        ),
        "eligible_action_change_rate": (
            action_changes / max(action_change_trials, 1)
        ),
        "normalized_action_entropy": float(np.mean(entropies)),
        "dropout_trials": dropout_trials,
        "congestion_trials": congestion_trials,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
