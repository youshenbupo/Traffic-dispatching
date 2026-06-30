"""Unit tests for SUMO environment."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.envs.sumo_env import SUMOMultiAgentEnv


class TestSUMOEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/default.yaml")
        cls.config["env"]["num_seconds"] = 300

    def test_reset_and_step(self):
        if not os.path.exists(os.path.join(self.config["env"]["scenario_dir"], self.config["env"]["config_file"])):
            self.skipTest("Scenario not generated yet")
        env = SUMOMultiAgentEnv(self.config)
        obs, info = env.reset(seed=42)
        self.assertIn("num_agents", info)
        self.assertGreater(len(obs), 0)
        masks = env.get_legal_actions()
        actions = {aid: 0 for aid in env.agent_ids}
        next_obs, rewards, terminated, truncated, info_step = env.step(actions)
        self.assertEqual(set(next_obs.keys()), set(obs.keys()))
        self.assertEqual(set(rewards.keys()), set(env.agent_ids))
        env.close()


if __name__ == "__main__":
    unittest.main()
