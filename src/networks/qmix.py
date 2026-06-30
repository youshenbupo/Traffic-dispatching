"""QMIX mixing network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXMixer(nn.Module):
    """Hypernetwork-based mixing network for QMIX."""

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: [batch, n_agents]
            state: [batch, state_dim]
        Returns:
            q_tot: [batch, 1]
        """
        batch = q_values.size(0)
        q_values = q_values.view(batch, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch)
