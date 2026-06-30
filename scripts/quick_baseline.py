"""Quick MaxPressure baseline for a scenario."""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents.max_pressure import MaxPressureController
from src.utils.metrics import MetricsTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    env = SUMOMultiAgentEnv(config)
    obs, info = env.reset(seed=42)
    controller = MaxPressureController(env.phases, env.incoming_lanes, env.outgoing_lanes)
    tracker = MetricsTracker()

    done = False
    step = 0
    while not done and step < args.num_steps:
        masks = env.get_legal_actions()
        actions = controller.act(obs, info=info, env=env)
        obs, rewards, terminated, truncated, info_step = env.step(actions)
        done = terminated or truncated
        step_metrics = MetricsTracker.compute_from_env(env)
        tracker.record_step({"reward": sum(rewards.values()), **step_metrics})
        step += 1

    arrived = env.get_arrived_vehicle_info()
    tracker.finalize(arrived)
    metrics = tracker.aggregate()
    env.close()

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"MaxPressure baseline: {metrics}")


if __name__ == "__main__":
    main()
