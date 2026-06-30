"""Base neural network modules for MARL agents."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPActor(nn.Module):
    """Discrete action actor with optional action masking."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, action_dim, num_layers=2)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None) -> torch.distributions.Categorical:
        logits = self.net(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return torch.distributions.Categorical(logits=logits)


class MLPCritic(nn.Module):
    """State-value critic."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, 1, num_layers=2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class MLPPolicy(nn.Module):
    """Actor-Critic policy for PPO/MAPPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = MLPActor(obs_dim, action_dim, hidden_dim)
        self.critic = MLPCritic(obs_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        dist = self.actor(obs, mask)
        value = self.critic(obs)
        return dist, value
