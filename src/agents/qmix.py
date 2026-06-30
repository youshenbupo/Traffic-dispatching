"""QMIX agent for multi-agent TSC."""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List
from collections import deque, namedtuple

from src.networks.base import MLP
from src.networks.qmix import QMIXMixer

Experience = namedtuple("Experience", ["obs", "action", "reward", "next_obs", "done", "state", "next_state"])


class QMIXAgent:
    """QMIX with parameter-shared agent networks and hypernetwork mixer."""

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

        self.q_network = MLP(obs_dim, self.hidden_dim, action_dim, num_layers=2).to(self.device)
        self.target_q_network = MLP(obs_dim, self.hidden_dim, action_dim, num_layers=2).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        state_dim = obs_dim * self.n_agents
        self.mixer = QMIXMixer(self.n_agents, state_dim, hidden_dim=32).to(self.device)
        self.target_mixer = QMIXMixer(self.n_agents, state_dim, hidden_dim=32).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        self.buffer = deque(maxlen=config.get("training", {}).get("buffer_size", 100000))

    def act(self, obs: Dict[str, np.ndarray], masks: Dict[str, np.ndarray] = None,
            explore: bool = True) -> Dict[str, int]:
        obs_tensor = torch.FloatTensor(np.stack([obs[aid] for aid in self.agent_ids])).unsqueeze(0).to(self.device)
        mask_tensor = None
        if masks is not None:
            mask_tensor = torch.BoolTensor(np.stack([masks[aid] for aid in self.agent_ids])).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.q_network(obs_tensor)
            if mask_tensor is not None:
                q = q.masked_fill(~mask_tensor, float("-inf"))

        actions = {}
        for i, aid in enumerate(self.agent_ids):
            if explore and random.random() < self.epsilon:
                legal = list(range(self.action_dim))
                if masks is not None:
                    legal = [j for j in legal if masks[aid][j]]
                actions[aid] = random.choice(legal) if legal else 0
            else:
                actions[aid] = int(q[0, i].argmax().item())
        return actions

    def store(self, obs, action, reward, next_obs, done):
        state = np.concatenate([obs[aid] for aid in self.agent_ids])
        next_state = np.concatenate([next_obs[aid] for aid in self.agent_ids])
        self.buffer.append(Experience(obs, action, reward, next_obs, done, state, next_state))

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {"mean_qmix_loss": 0.0}

        batch = random.sample(self.buffer, self.batch_size)
        obs = torch.FloatTensor(np.stack([np.stack([e.obs[aid] for aid in self.agent_ids]) for e in batch])).to(self.device)
        actions = torch.LongTensor(np.stack([[e.action[aid] for aid in self.agent_ids] for e in batch])).to(self.device)
        rewards = torch.FloatTensor([np.mean([e.reward[aid] for aid in self.agent_ids]) for e in batch]).to(self.device)
        next_obs = torch.FloatTensor(np.stack([np.stack([e.next_obs[aid] for aid in self.agent_ids]) for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        states = torch.FloatTensor(np.stack([e.state for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.stack([e.next_state for e in batch])).to(self.device)

        q_values = self.q_network(obs)
        q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(q_values, states)

        with torch.no_grad():
            next_q = self.target_q_network(next_obs).max(dim=2)[0]
            next_q_tot = self.target_mixer(next_q, next_states)
            target = rewards + self.gamma * next_q_tot * (1 - dones)

        loss = F.mse_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        for p, t in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)
        for p, t in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return {"mean_qmix_loss": loss.item()}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "q": self.q_network.state_dict(),
            "mixer": self.mixer.state_dict(),
        }, os.path.join(path, "qmix.pth"))

    def load(self, path: str):
        ckpt = torch.load(os.path.join(path, "qmix.pth"), map_location=self.device)
        self.q_network.load_state_dict(ckpt["q"])
        self.target_q_network.load_state_dict(ckpt["q"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(ckpt["mixer"])
