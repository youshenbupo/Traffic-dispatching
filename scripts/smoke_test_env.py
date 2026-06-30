"""Quick smoke test: load a SUMO scenario and run a few steps."""
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.envs.sumo_env import SUMOMultiAgentEnv


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env = SUMOMultiAgentEnv(cfg)
    obs, info = env.reset(seed=cfg.get("seed", 42))
    print(f"Agents: {info['num_agents']}, IDs: {info['tls_ids'][:5]}...")
    obs_shapes = {k: v.shape for k, v in obs.items()}
    print(f"Obs shapes (first 3): {dict(list(obs_shapes.items())[:3])}")
    print(f"Unique obs shapes: {set(obs_shapes.values())}")

    # Check phase counts
    phase_counts = {k: len(v) for k, v in env.phases.items()}
    print(f"Phase counts (first 3): {dict(list(phase_counts.items())[:3])}")
    print(f"Unique phase counts: {set(phase_counts.values())}")

    # Run 3 random steps
    for step in range(3):
        actions = {tl_id: 0 for tl_id in env.tls_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"Step {step+1}: time={info['time']}, vehicles={info['vehicles']}, arrived={info['arrived']}")

    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    main(sys.argv[1])
