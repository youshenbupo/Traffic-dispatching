"""Probe a SUMO scenario's MARL dimensions without starting training."""
import argparse
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.envs.sumo_env import SUMOMultiAgentEnv
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    env = SUMOMultiAgentEnv(config)
    try:
        observations, _ = env.reset(seed=config.get("seed", 42))
        print("start_time", env.episode_start_time)
        print("num_agents", env.num_agents)
        for agent_id in env.agent_ids:
            print(
                agent_id,
                "obs_dim", observations[agent_id].shape[0],
                "actions", len(env.phases[agent_id]),
                "policy_actions", env.get_legal_actions()[agent_id].shape[0],
                "incoming_lanes", len(env.incoming_lanes[agent_id]),
            )
        actions = {
            agent_id: env.current_phase[agent_id]
            for agent_id in env.agent_ids
        }
        _, _, terminated, truncated, _ = env.step(actions)
        print("first_step_terminated", terminated)
        print("first_step_truncated", truncated)
    finally:
        env.close()


if __name__ == "__main__":
    main()
