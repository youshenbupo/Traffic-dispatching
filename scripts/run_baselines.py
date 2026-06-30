"""Run all baseline controllers on a scenario and print comparison table."""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents.fixed_time import FixedTimeController
from src.agents.max_pressure import MaxPressureController
from src.utils.metrics import MetricsTracker


def run_controller(env, controller, num_steps: int):
    obs, info = env.reset(seed=42)
    controller.reset()
    tracker = MetricsTracker()
    done = False
    step = 0
    while not done and step < num_steps:
        masks = env.get_legal_actions()
        if isinstance(controller, MaxPressureController):
            actions = controller.act(obs, info=info, env=env)
        else:
            actions = controller.act(obs, info=info)
        next_obs, rewards, terminated, truncated, info_step = env.step(actions)
        done = terminated or truncated
        step_metrics = MetricsTracker.compute_from_env(env)
        tracker.record_step({"reward": sum(rewards.values()), **step_metrics})
        obs = next_obs
        step += 1
    arrived = env.get_arrived_vehicle_info()
    tracker.finalize(arrived)
    return tracker.aggregate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.scenario:
        config["env"]["scenario_dir"] = args.scenario
        config["env"]["config_file"] = "grid.sumocfg"

    results = {}
    for controller_name in ["fixed_time", "max_pressure"]:
        env = SUMOMultiAgentEnv(config)
        obs, info = env.reset(seed=42)
        if controller_name == "fixed_time":
            controller = FixedTimeController(env.phases)
        else:
            controller = MaxPressureController(env.phases, env.incoming_lanes, env.outgoing_lanes)
        metrics = run_controller(env, controller, args.num_steps)
        results[controller_name] = metrics
        env.close()
        print(f"{controller_name}: {metrics}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
