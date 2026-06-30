"""Main training script for TSC experiments."""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, merge_config, build_arg_parser
from src.utils.logger import setup_logger
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents import FixedTimeController, MaxPressureController, DQNAgent, PPOAgent, MAPPOAgent, QMIXAgent
from src.trainers import OnlineTrainer, OfflineTrainer


def set_gpus(gpu_str: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    print(f"Using GPUs: {gpu_str}")


def make_agent(env, config):
    agent_name = config.get("agent", {}).get("name", "mappo").lower()
    obs_sample = env._get_observations()
    obs_dim = obs_sample[env.agent_ids[0]].shape[0]
    action_dim = len(env.phases[env.agent_ids[0]])

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
    parser = build_arg_parser()
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pretrained actor weights (.pth)")
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {}
    if args.gpus is not None:
        overrides.setdefault("training", {})["gpus"] = args.gpus
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.exp_name is not None:
        overrides["experiment_name"] = args.exp_name
    config = merge_config(config, overrides)

    gpus = config.get("training", {}).get("gpus", "")
    if gpus:
        set_gpus(gpus)

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    exp_name = config.get("experiment_name", "exp")
    log_dir = os.path.join(config.get("log_dir", "./logs"), exp_name)
    result_dir = os.path.join(config.get("result_dir", "./results"), exp_name)
    config["log_dir"] = log_dir
    config["result_dir"] = result_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger(log_dir, exp_name)
    logger.info(f"Config: {config}")

    env = SUMOMultiAgentEnv(config)
    env.reset(seed=config.get("seed", 42))
    agent = make_agent(env, config)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pretrained model from {args.pretrained_model}")
        agent.load(args.pretrained_model)
        transfer_cfg = config.get("transfer", {})
        if transfer_cfg.get("freeze_actor", False):
            print("Freezing actor")
            agent.freeze_actor()
        if transfer_cfg.get("freeze_critic", False):
            print("Freezing critic")
            agent.freeze_critic()

    if config.get("offline", {}).get("enabled", False):
        trainer = OfflineTrainer(env, agent, config, logger)
    else:
        trainer = OnlineTrainer(env, agent, config, logger)

    trainer.run()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
