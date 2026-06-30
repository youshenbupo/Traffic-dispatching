"""One-round DAgger for Hangzhou offline BC.

1. Load a trained policy.
2. Roll out N episodes with that policy.
3. Query MaxPressure expert for actions at each visited state.
4. Mix with original offline data and save aggregated dataset.
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.utils.config import load_config, merge_config
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents import MAPPOAgent, MaxPressureController


def set_gpus(gpu_str: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"Using GPUs: {gpu_str}")


def make_agent(env, config):
    obs_sample = env._get_observations()
    obs_dim = obs_sample[env.agent_ids[0]].shape[0]
    action_dim = len(env.phases[env.agent_ids[0]])
    return MAPPOAgent(obs_dim, action_dim, env.agent_ids, config)


def collect_policy_rollout(env, agent, num_episodes: int, seed_offset: int = 10000):
    """Return list of dicts with obs at each step."""
    trajectories = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_data = []
        while not done:
            masks = env.get_legal_actions()
            adj = env.get_adjacency(mode="static", k=2)
            actions, _, _ = agent.act(obs, masks, explore=False, adj=adj)
            ep_data.append({"obs": obs, "actions": actions})
            next_obs, rewards, terminated, truncated, info_step = env.step(actions)
            done = terminated or truncated
            obs = next_obs
        trajectories.append(ep_data)
        print(f"DAgger rollout episode {ep + 1}/{num_episodes} completed, {len(ep_data)} steps")
    return trajectories


def label_with_maxpressure(env, trajectories):
    """For each recorded obs, query MaxPressure expert action."""
    expert = MaxPressureController(env.phases, env.incoming_lanes, env.outgoing_lanes,
                                   yellow_time=env.yellow_time, min_green=env.min_green)
    dataset = []
    for ep_idx, ep_data in enumerate(trajectories):
        # Reset expert internal clock each episode
        expert.reset()
        # We need to replay the env state to query expert. Since we cannot rewind env,
        # we instead instantiate a fresh env and roll MaxPressure on it independently,
        # then align by step index. This is approximate but sufficient for a single DAgger round.
        # Better: query expert during the policy rollout by running MaxPressure in parallel.
        # We implement the parallel approach below.
        pass
    return dataset


def collect_and_label(env, agent, num_episodes: int, seed_offset: int = 10000):
    """Roll out policy while simultaneously querying MaxPressure for expert labels."""
    expert = MaxPressureController(env.phases, env.incoming_lanes, env.outgoing_lanes,
                                   yellow_time=env.yellow_time, min_green=env.min_green)
    dataset = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        expert.reset()
        done = False
        step = 0
        while not done:
            masks = env.get_legal_actions()
            adj = env.get_adjacency(mode="static", k=2)
            # Policy action
            actions, _, _ = agent.act(obs, masks, explore=False, adj=adj)
            # Expert action on current env state
            expert_actions = expert.act(obs, info=info, env=env)
            dataset.append({"obs": obs, "action": expert_actions,
                            "policy_action": actions, "reward": {}})
            next_obs, rewards, terminated, truncated, info_step = env.step(actions)
            done = terminated or truncated
            obs = next_obs
            step += 1
        print(f"DAgger episode {ep + 1}/{num_episodes} completed, {step} steps, total transitions {len(dataset)}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mappo_hangzhou4x4_offline_bc_5ep_longer_lowlr_finetune.yaml")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing mappo.pth of trained policy")
    parser.add_argument("--num_rollout_episodes", type=int, default=5)
    parser.add_argument("--original_dataset", type=str,
                        default="data/datasets/hangzhou4x4_maxpressure_5ep.pkl")
    parser.add_argument("--output", type=str, default="data/datasets/hangzhou4x4_dagger_round1.pkl")
    parser.add_argument("--gpus", type=str, default="2,3")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpus:
        config = merge_config(config, {"training": {"gpus": args.gpus}})
        set_gpus(args.gpus)

    env = SUMOMultiAgentEnv(config)
    env.reset(seed=config.get("seed", 42))

    agent = make_agent(env, config)
    agent.load(args.model_path)
    # MAPPOAgent does not have eval(); act(explore=False) gives deterministic actions.
    if hasattr(agent, "eval"):
        agent.eval()

    print(f"Collecting {args.num_rollout_episodes} DAgger episodes with current policy...")
    dagger_data = collect_and_label(env, agent, args.num_rollout_episodes)

    if args.original_dataset and os.path.exists(args.original_dataset):
        with open(args.original_dataset, "rb") as f:
            original_data = pickle.load(f)
        print(f"Original dataset: {len(original_data)} transitions")
        # Convert original transitions to (obs, action) pairs if needed
        converted = []
        for tr in original_data:
            if "action" in tr:
                converted.append({"obs": tr["obs"], "action": tr["action"]})
            elif "actions" in tr:
                converted.append({"obs": tr["obs"], "action": tr["actions"]})
        mixed = converted + dagger_data
        print(f"Mixed dataset: {len(mixed)} transitions ({len(converted)} original + {len(dagger_data)} new)")
    else:
        mixed = dagger_data
        print(f"No original dataset; using {len(mixed)} DAgger transitions")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(mixed, f)
    print(f"Saved mixed DAgger dataset to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
