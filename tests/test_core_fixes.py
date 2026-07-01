"""Regression tests for correctness issues that affect reported results."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.mappo import MAPPOAgent
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.networks.base import (
    AnchoredResidualActor,
    FailureGatedAnchoredActor,
    ResidualCommActor,
)
from src.networks.comm import ReliabilityAwareCommLayer
from src.utils.metrics import MetricsTracker
from src.utils.config import load_config


class _TrafficLightStub:
    def __init__(self):
        self.states = []

    def setRedYellowGreenState(self, tl_id, state):
        self.states.append((tl_id, state))


class _SumoStub:
    def __init__(self):
        self.trafficlight = _TrafficLightStub()

    def close(self):
        pass


class TestSignalTransitions(unittest.TestCase):
    def setUp(self):
        self.env = SUMOMultiAgentEnv.__new__(SUMOMultiAgentEnv)
        self.env.sumo = _SumoStub()
        self.env.tls_ids = ["A"]
        self.env.phases = {"A": ["GGrr", "rrGG"]}
        self.env.current_phase = {"A": 0}
        self.env.phase_duration = {"A": 7}
        self.env.pending_phase = {"A": None}
        self.env.transition_remaining = {"A": 0}
        self.env.min_green = 5
        self.env.max_green = 60
        self.env.yellow_time = 3

    def test_phase_duration_advances_one_second(self):
        self.env._update_phase_durations()
        self.assertEqual(self.env.phase_duration["A"], 8)

    def test_green_switch_uses_yellow_transition(self):
        self.env._apply_action("A", 1)
        self.assertEqual(self.env.sumo.trafficlight.states[-1], ("A", "yyrr"))
        self.assertEqual(self.env.pending_phase["A"], 1)

        for _ in range(3):
            self.env._update_phase_durations()

        self.assertEqual(self.env.sumo.trafficlight.states[-1], ("A", "rrGG"))
        self.assertEqual(self.env.current_phase["A"], 1)
        self.assertIsNone(self.env.pending_phase["A"])


class TestMetrics(unittest.TestCase):
    def test_arrival_throughput_and_trip_metrics(self):
        tracker = MetricsTracker()
        tracker.record_step({"reward": -1.0, "queue_length": 2.0, "throughput": 3.0})
        tracker.record_step({"reward": -2.0, "queue_length": 4.0, "throughput": 2.0})
        tracker.finalize([
            {"travel_time": 10.0, "waiting_time": 2.0},
            {"travel_time": 20.0, "waiting_time": 6.0},
        ])
        result = tracker.aggregate()
        self.assertEqual(result["throughput"], 5.0)
        self.assertEqual(result["average_travel_time"], 15.0)
        self.assertEqual(result["average_waiting_time"], 4.0)


class TestConfig(unittest.TestCase):
    def test_recursive_base_config(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "racc_grid4x4_no_reliability.yaml",
        )
        config = load_config(path)
        self.assertEqual(config["training"]["num_episodes"], 100)
        self.assertEqual(config["dynamic_graph"]["reliability_coef"], 0.0)


class TestMAPPOUpdate(unittest.TestCase):
    def test_no_message_reconstruction_cannot_hallucinate(self):
        import torch

        layer = ReliabilityAwareCommLayer(3, 4)
        observed = torch.tensor([[[1.0, 0.0, 3.0]]])
        mask = torch.tensor([[[1.0, 0.0, 1.0]]])
        messages = torch.zeros(1, 1, 4)
        reconstructed, confidence = layer.reconstruct_with_confidence(
            messages, observed, mask
        )
        self.assertTrue(torch.allclose(reconstructed, observed))
        self.assertTrue(torch.allclose(confidence, torch.zeros_like(confidence)))
        _, active_confidence = layer.reconstruct_with_confidence(
            torch.ones(1, 1, 4), observed, mask
        )
        self.assertTrue(torch.all(active_confidence >= 0))
        self.assertTrue(torch.all(active_confidence <= 1))

    def test_no_message_has_zero_decision_confidence(self):
        import torch

        layer = ReliabilityAwareCommLayer(4, 8)
        observed = torch.randn(2, 3, 4)
        no_messages = torch.zeros(2, 3, 8)
        confidence = layer.decision_confidence(no_messages, observed)
        self.assertTrue(torch.equal(confidence, torch.zeros_like(confidence)))
        messages = torch.randn(2, 3, 8)
        active = layer.decision_confidence(messages, observed)
        self.assertTrue(torch.all(active >= 0))
        self.assertTrue(torch.all(active <= 1))

    def test_residual_actor_starts_as_local_policy(self):
        import torch

        actor = ResidualCommActor(3, 4, 2, hidden_dim=8)
        local = torch.tensor([[1.0, 2.0, 3.0]])
        first = actor(torch.cat([local, torch.zeros(1, 4)], dim=-1)).probs
        second = actor(torch.cat([local, torch.ones(1, 4)], dim=-1)).probs
        self.assertTrue(torch.allclose(first, second))
        with torch.no_grad():
            actor.comm_net.net[-1].weight.fill_(1.0)
            actor.comm_net.net[-1].bias.fill_(1.0)
        fallback = actor(torch.cat([local, torch.zeros(1, 4)], dim=-1)).probs
        local_only = torch.softmax(actor.local_net(local), dim=-1)
        self.assertTrue(torch.allclose(fallback, local_only))

    def test_anchored_actor_starts_at_frozen_teacher(self):
        import torch

        actor = AnchoredResidualActor(3, 4, 2, hidden_dim=8)
        observation = torch.randn(5, 7)
        expected = torch.softmax(
            actor.base_net(observation[:, :3]), dim=-1
        )
        actual = actor(observation).probs
        self.assertTrue(torch.allclose(actual, expected))
        actor.freeze_anchor()
        self.assertTrue(
            all(not parameter.requires_grad for parameter in actor.base_net.parameters())
        )

    def test_failure_gate_preserves_clean_teacher(self):
        import torch

        actor = FailureGatedAnchoredActor(3, 4, 2, hidden_dim=8)
        with torch.no_grad():
            actor.adapter.net[-1].weight.fill_(1.0)
            actor.adapter.net[-1].bias.fill_(1.0)
        clean_input = torch.randn(5, 7)
        clean_input[:, 3:] = 0.0
        expected = torch.softmax(
            actor.base_net(clean_input[:, :3]), dim=-1
        )
        self.assertTrue(torch.allclose(actor(clean_input).probs, expected))

    def test_multi_agent_ratio_update_is_finite(self):
        config = {
            "agent": {"hidden_dim": 16},
            "training": {"batch_size": 2, "use_gpu": False},
            "dynamic_graph": {"enabled": False, "comm_type": "racc"},
        }
        ids = ["A", "B"]
        agent = MAPPOAgent(3, 2, ids, config)
        obs = {aid: np.zeros(3, dtype=np.float32) for aid in ids}
        masks = {aid: np.ones(2, dtype=bool) for aid in ids}
        adj = np.eye(2, dtype=np.float32)
        for step in range(2):
            actions, log_probs, values = agent.act(obs, masks, explore=True, adj=adj)
            rewards = {aid: 1.0 for aid in ids}
            agent.store_transition(
                obs, actions, rewards, values, log_probs, masks, adj, step == 1
            )
        result = agent.update()
        self.assertTrue(math.isfinite(result["mean_mappo_loss"]))

    def test_reliability_aware_communication_update_is_finite(self):
        config = {
            "agent": {"hidden_dim": 8},
            "training": {"batch_size": 2, "use_gpu": False},
            "dynamic_graph": {
                "enabled": True,
                "comm_type": "racc",
                "gate_threshold": 0.2,
                "counterfactual_samples": 1,
            },
        }
        ids = ["A", "B"]
        agent = MAPPOAgent(3, 2, ids, config)
        obs = {
            "A": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "B": np.array([2.0, 1.0, 0.0], dtype=np.float32),
        }
        masks = {aid: np.ones(2, dtype=bool) for aid in ids}
        adj = np.ones((2, 2), dtype=np.float32)
        for step in range(2):
            actions, log_probs, values = agent.act(obs, masks, explore=True, adj=adj)
            agent.store_transition(
                obs,
                actions,
                {aid: 1.0 for aid in ids},
                values,
                log_probs,
                masks,
                adj,
                step == 1,
            )
        result = agent.update()
        self.assertTrue(math.isfinite(result["mean_mappo_loss"]))
        self.assertIn("counterfactual_loss", result)
        self.assertIn("confidence_loss", result)
        self.assertEqual(agent.update_count, 1)
        self.assertGreaterEqual(agent.last_comm_stats["communication_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
