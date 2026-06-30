"""Run a minimal demo with baseline controllers."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents.fixed_time import FixedTimeController
from src.agents.max_pressure import MaxPressureController
from src.utils.metrics import MetricsTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--controller", type=str, default="fixed_time", choices=["fixed_time", "max_pressure"])
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.scenario:
        config["env"]["scenario_dir"] = args.scenario
        config["env"]["config_file"] = args.config_file or "grid.sumocfg"

    env = SUMOMultiAgentEnv(config)
    obs, info = env.reset(seed=42)

    if args.controller == "fixed_time":
        controller = FixedTimeController(env.phases)
    elif args.controller == "max_pressure":
        controller = MaxPressureController(env.phases, env.incoming_lanes, env.outgoing_lanes)
    else:
        raise ValueError(f"Unknown controller: {args.controller}")
    controller.reset()

    tracker = MetricsTracker()
    done = False
    step = 0
    while not done and step < args.steps:
        masks = env.get_legal_actions()
        if args.controller == "max_pressure":
            actions = controller.act(obs, info=info, env=env)
        else:
            actions = controller.act(obs, info=info)
        next_obs, rewards, terminated, truncated, info_step = env.step(actions)
        done = terminated or truncated
        step_metrics = MetricsTracker.compute_from_env(env)
        tracker.record_step({"reward": sum(rewards.values()), **step_metrics})
        obs = next_obs
        step += 1
        if step % 100 == 0:
            print(f"Step {step}, time={info_step['time']:.1f}, arrived={info_step['arrived']}")

    arrived = env.get_arrived_vehicle_info()
    tracker.finalize(arrived)
    metrics = tracker.aggregate()
    print("Final metrics:", metrics)
    env.close()


if __name__ == "__main__":
    main()
