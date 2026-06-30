"""Single-agent PPO (independent per intersection)."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from torch.distributions import Categorical

from src.networks.base import MLPPolicy


class PPOAgent:
    """Independent PPO for each intersection."""

    def __init__(self, obs_dim: int, action_dim: int, agent_ids: List[str], config: Dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.cfg = config
        self.agent_cfg = config.get("agent", {})
        self.lr = self.agent_cfg.get("lr", 3e-4)
        self.gamma = self.agent_cfg.get("gamma", 0.99)
        self.gae_lambda = self.agent_cfg.get("gae_lambda", 0.95)
        self.clip_param = self.agent_cfg.get("clip_param", 0.2)
        self.entropy_coef = self.agent_cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = self.agent_cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = self.agent_cfg.get("max_grad_norm", 0.5)
        self.hidden_dim = self.agent_cfg.get("hidden_dim", 128)
        self.ppo_epochs = 4
        self.batch_size = config.get("training", {}).get("batch_size", 256)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("use_gpu", True) else "cpu")

        self.policies = {}
        self.optimizers = {}
        self.buffers = {aid: [] for aid in agent_ids}

        for aid in agent_ids:
            policy = MLPPolicy(obs_dim, action_dim, self.hidden_dim).to(self.device)
            self.policies[aid] = policy
            self.optimizers[aid] = torch.optim.Adam(policy.parameters(), lr=self.lr)

    def act(self, obs: Dict[str, np.ndarray], masks: Dict[str, np.ndarray] = None,
            explore: bool = True):
        actions = {}
        log_probs = {}
        values = {}
        for aid in self.agent_ids:
            o = torch.FloatTensor(obs[aid]).unsqueeze(0).to(self.device)
            mask = None
            if masks is not None:
                mask = torch.BoolTensor(masks[aid]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist, value = self.policies[aid](o, mask)
                if explore:
                    action = dist.sample()
                else:
                    action = dist.probs.argmax(dim=-1)
                actions[aid] = int(action.item())
                log_probs[aid] = float(dist.log_prob(action).item())
                values[aid] = float(value.squeeze().item())
        return actions, log_probs, values

    def store_transition(self, obs, action, reward, value, log_prob, mask, done):
        for aid in self.agent_ids:
            self.buffers[aid].append({
                "obs": obs[aid],
                "action": action[aid],
                "reward": reward[aid],
                "value": value[aid] if isinstance(value, dict) else 0.0,
                "log_prob": log_prob[aid] if isinstance(log_prob, dict) else 0.0,
                "mask": mask[aid] if mask is not None else np.ones(self.action_dim, dtype=bool),
                "done": done,
            })

    def compute_returns_and_advantages(self, aid: str):
        buf = self.buffers[aid]
        rewards = np.array([t["reward"] for t in buf])
        values = np.array([t["value"] for t in buf])
        dones = np.array([t["done"] for t in buf])

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(buf))):
            if t == len(buf) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return returns, advantages

    def update(self) -> Dict[str, float]:
        total_loss = []
        for aid in self.agent_ids:
            if len(self.buffers[aid]) < self.batch_size:
                continue
            returns, advantages = self.compute_returns_and_advantages(aid)
            obs = torch.FloatTensor(np.stack([t["obs"] for t in self.buffers[aid]])).to(self.device)
            actions = torch.LongTensor([t["action"] for t in self.buffers[aid]]).to(self.device)
            old_log_probs = torch.FloatTensor([t["log_prob"] for t in self.buffers[aid]]).to(self.device)
            masks = torch.BoolTensor(np.stack([t["mask"] for t in self.buffers[aid]])).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(self.ppo_epochs):
                dist, values = self.policies[aid](obs, masks)
                log_probs = dist.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -dist.entropy().mean()
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizers[aid].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policies[aid].parameters(), self.max_grad_norm)
                self.optimizers[aid].step()
                total_loss.append(loss.item())

            self.buffers[aid] = []
        return {"mean_ppo_loss": float(np.mean(total_loss)) if total_loss else 0.0}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for aid in self.agent_ids:
            torch.save(self.policies[aid].state_dict(), os.path.join(path, f"policy_{aid}.pth"))

    def load(self, path: str):
        for aid in self.agent_ids:
            self.policies[aid].load_state_dict(torch.load(os.path.join(path, f"policy_{aid}.pth"), map_location=self.device))
