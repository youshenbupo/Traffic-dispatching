"""Independent DQN agent for multi-agent TSC."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List

from src.networks.base import MLP

Experience = namedtuple("Experience", ["obs", "action", "reward", "next_obs", "done"])


class DQNAgent:
    """One DQN per intersection agent."""

    def __init__(self, obs_dim: int, action_dim: int, agent_ids: List[str], config: Dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.cfg = config
        self.agent_cfg = config.get("agent", {})
        self.lr = self.agent_cfg.get("lr", 3e-4)
        self.gamma = self.agent_cfg.get("gamma", 0.99)
        self.batch_size = config.get("training", {}).get("batch_size", 64)
        self.hidden_dim = self.agent_cfg.get("hidden_dim", 128)
        self.tau = 0.005
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("use_gpu", True) else "cpu")

        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}
        self.buffers = {aid: deque(maxlen=config.get("training", {}).get("buffer_size", 100000)) for aid in agent_ids}

        for aid in agent_ids:
            q = MLP(obs_dim, self.hidden_dim, action_dim, num_layers=2).to(self.device)
            t = MLP(obs_dim, self.hidden_dim, action_dim, num_layers=2).to(self.device)
            t.load_state_dict(q.state_dict())
            self.q_networks[aid] = q
            self.target_networks[aid] = t
            self.optimizers[aid] = torch.optim.Adam(q.parameters(), lr=self.lr)

    def act(self, obs: Dict[str, np.ndarray], masks: Dict[str, np.ndarray] = None,
            explore: bool = True) -> Dict[str, int]:
        actions = {}
        for aid in self.agent_ids:
            o = torch.FloatTensor(obs[aid]).unsqueeze(0).to(self.device)
            mask = None
            if masks is not None:
                mask = torch.BoolTensor(masks[aid]).unsqueeze(0).to(self.device)

            if explore and random.random() < self.epsilon:
                legal = list(range(self.action_dim))
                if mask is not None:
                    legal = [i for i in legal if mask[0, i].item()]
                actions[aid] = random.choice(legal) if legal else 0
            else:
                with torch.no_grad():
                    q = self.q_networks[aid](o)
                    if mask is not None:
                        q = q.masked_fill(~mask, float("-inf"))
                    actions[aid] = int(q.argmax(dim=-1).item())
        return actions

    def store(self, experiences: List[Dict]):
        for exp in experiences:
            for aid in self.agent_ids:
                self.buffers[aid].append(Experience(
                    obs=exp["obs"][aid],
                    action=exp["action"][aid],
                    reward=exp["reward"][aid],
                    next_obs=exp["next_obs"][aid],
                    done=exp["done"],
                ))

    def update(self) -> Dict[str, float]:
        losses = {}
        for aid in self.agent_ids:
            if len(self.buffers[aid]) < self.batch_size:
                continue
            batch = random.sample(self.buffers[aid], self.batch_size)
            obs = torch.FloatTensor(np.stack([e.obs for e in batch])).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_obs = torch.FloatTensor(np.stack([e.next_obs for e in batch])).to(self.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

            q_values = self.q_networks[aid](obs)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = self.target_networks[aid](next_obs).max(dim=1)[0]
                target = rewards + self.gamma * next_q * (1 - dones)

            loss = F.mse_loss(q_values, target)
            self.optimizers[aid].zero_grad()
            loss.backward()
            self.optimizers[aid].step()

            # Soft update target
            for p, t in zip(self.q_networks[aid].parameters(), self.target_networks[aid].parameters()):
                t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)

            losses[aid] = loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return {"mean_dqn_loss": float(np.mean(list(losses.values()))) if losses else 0.0}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for aid in self.agent_ids:
            torch.save(self.q_networks[aid].state_dict(), os.path.join(path, f"q_{aid}.pth"))

    def load(self, path: str):
        for aid in self.agent_ids:
            self.q_networks[aid].load_state_dict(torch.load(os.path.join(path, f"q_{aid}.pth"), map_location=self.device))
