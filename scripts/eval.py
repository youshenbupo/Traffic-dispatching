"""Evaluation script for trained TSC agents."""
import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, merge_config
from src.utils.logger import setup_logger
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents import DQNAgent, PPOAgent, MAPPOAgent, QMIXAgent
from src.trainers.online_trainer import _agent_act
from src.utils.metrics import MetricsTracker


def set_gpus(gpu_str: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"Using GPUs: {gpu_str}")


def make_agent(env, config):
    agent_name = config.get("agent", {}).get("name", "mappo").lower()
    obs_sample = env._get_observations()
    obs_dim = obs_sample[env.agent_ids[0]].shape[0]
    action_dim = (
        env.max_action_dim
        if env.pad_heterogeneous_spaces
        else len(env.phases[env.agent_ids[0]])
    )

    if agent_name == "dqn":
        return DQNAgent(obs_dim, action_dim, env.agent_ids, config)
    elif agent_name == "ppo":
        return PPOAgent(obs_dim, action_dim, env.agent_ids, config)
    elif agent_name == "mappo":
        return MAPPOAgent(obs_dim, action_dim, env.agent_ids, config)
    elif agent_name == "qmix":
        return QMIXAgent(obs_dim, action_dim, env.agent_ids, config)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=20000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpus:
        config = merge_config(config, {"training": {"gpus": args.gpus}})
        set_gpus(args.gpus)

    env = SUMOMultiAgentEnv(config)
    env.reset(seed=config.get("seed", 42))
    agent = make_agent(env, config)
    agent.load(args.model_path)
    graph_cfg = config.get("dynamic_graph", {})
    graph_type = graph_cfg.get("graph_type", "static")
    neighbor_k = graph_cfg.get("neighbor_k", 1)

    logger = setup_logger(config.get("log_dir", "./logs"), "eval")
    metrics_list = []
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed_start + ep)
        tracker = MetricsTracker()
        done = False
        while not done:
            masks = env.get_legal_actions()
            adj = env.get_adjacency(mode=graph_type, k=neighbor_k)
            obs_mask = {
                aid: value.copy()
                for aid, value in env.last_observation_masks.items()
            }
            clean_obs = {
                aid: value.copy()
                for aid, value in env.last_clean_observations.items()
            }
            failure_age = dict(
                getattr(env, "last_failure_age", {})
            )
            structure_context = {
                aid: value.copy()
                for aid, value in getattr(
                    env, "last_structure_context", {}
                ).items()
            }
            actions, _, _ = _agent_act(
                agent, obs, masks, adj, explore=False,
                obs_mask=obs_mask, clean_obs=clean_obs,
                failure_age=failure_age,
                structure_context=structure_context,
            )
            next_obs, rewards, terminated, truncated, info_step = env.step(actions)
            done = terminated or truncated
            step_metrics = MetricsTracker.compute_from_env(env)
            step_metrics.update(getattr(agent, "last_comm_stats", {}))
            tracker.record_step({"reward": sum(rewards.values()), **step_metrics})
            obs = next_obs
        arrived = env.get_arrived_vehicle_info()
        tracker.finalize(arrived)
        metrics_list.append(tracker.aggregate())
        logger.info(f"Eval episode {ep}: {metrics_list[-1]}")

    result = {}
    statistics = {}
    for key in metrics_list[0]:
        values = np.asarray([m[key] for m in metrics_list], dtype=np.float64)
        result[key] = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        statistics[key] = {
            "mean": float(values.mean()),
            "std": std,
            "ci95": float(1.96 * std / np.sqrt(len(values))),
        }
    result["_statistics"] = statistics
    result["_episodes"] = metrics_list
    result["_seeds"] = [args.seed_start + ep for ep in range(args.num_episodes)]
    out_path = args.output or os.path.join(
        config.get("result_dir", "./results"), "eval_results.json"
    )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Average eval metrics: {result}")
    env.close()


if __name__ == "__main__":
    main()
