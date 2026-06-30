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


class ResidualCommActor(nn.Module):
    """Local actor plus a zero-initialized communication correction."""

    def __init__(
        self,
        obs_dim: int,
        comm_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.local_net = MLP(obs_dim, hidden_dim, action_dim, num_layers=2)
        self.comm_net = MLP(
            obs_dim + comm_dim, hidden_dim, action_dim, num_layers=2
        )
        final_layer = self.comm_net.net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self, actor_input: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.distributions.Categorical:
        local_obs = actor_input[..., :self.obs_dim]
        comm_features = actor_input[..., self.obs_dim:]
        # Enforce an exact fallback: when every message is rejected, the
        # communication branch contributes exactly zero, including its biases.
        message_present = (
            comm_features.abs().sum(dim=-1, keepdim=True) > 1e-8
        ).to(actor_input.dtype)
        correction = self.comm_net(actor_input) * message_present
        logits = self.local_net(local_obs) + correction
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
