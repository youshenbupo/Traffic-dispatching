"""Generate offline dataset using behavior policies (fixed-time / max-pressure)."""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents import FixedTimeController, MaxPressureController


def generate_dataset(config, behavior_policy: str, num_episodes: int, output_path: str):
    env = SUMOMultiAgentEnv(config)
    obs, info = env.reset(seed=42)

    if behavior_policy == "fixed_time":
        controller = FixedTimeController(env.phases)
    elif behavior_policy == "max_pressure":
        controller = MaxPressureController(env.phases, env.incoming_lanes, env.outgoing_lanes)
    else:
        raise ValueError(f"Unknown behavior policy: {behavior_policy}")

    dataset = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        while not done:
            masks = env.get_legal_actions()
            if behavior_policy == "max_pressure":
                actions = controller.act(obs, info=info, env=env)
            else:
                actions = controller.act(obs, info=info)
            next_obs, rewards, terminated, truncated, info_step = env.step(actions)
            done = terminated or truncated
            dataset.append({
                "obs": obs,
                "action": actions,
                "reward": rewards,
                "next_obs": next_obs,
                "done": float(done),
            })
            obs = next_obs
        print(f"Episode {ep + 1}/{num_episodes} completed. Total transitions: {len(dataset)}")

    env.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {output_path} with {len(dataset)} transitions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--policy", type=str, default="max_pressure", choices=["fixed_time", "max_pressure", "mixed"])
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="data/datasets/offline_grid3x3.pkl")
    args = parser.parse_args()

    config = load_config(args.config)
    generate_dataset(config, args.policy, args.num_episodes, args.output)


if __name__ == "__main__":
    main()
