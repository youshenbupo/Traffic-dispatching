"""Lightweight communication modules for multi-agent TSC."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolCommLayer(nn.Module):
    """Simple mean-pooling communication over neighbors.

    For each agent i, compute the mean observation of its neighbors
    (including itself) according to the adjacency matrix, then project.

    Args:
        in_dim: input feature dimension per agent
        out_dim: output feature dimension per agent
        dropout: dropout rate
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_agents, in_dim]
            adj: [batch, num_agents, num_agents]
        Returns:
            out: [batch, num_agents, out_dim]
        """
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # neighbor_sum[b, i, f] = sum_j adj[b, i, j] * x[b, j, f]
        neighbor_sum = torch.einsum("bij,bjf->bif", adj, x)
        neighbor_count = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neighbor_mean = neighbor_sum / neighbor_count

        out = self.proj(neighbor_mean)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out.squeeze(0) if x.shape[0] == 1 else out


class ReliabilityAwareCommLayer(nn.Module):
    """Sparse messages gated by sender reliability and pairwise utility."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        gate_threshold: float = 0.5,
    ):
        super().__init__()
        self.gate_threshold = gate_threshold
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )
        self.message = nn.Linear(out_dim, out_dim)
        self.reliability_head = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(3 * out_dim + 1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_details: bool = False,
        hard: bool = True,
    ):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        _, n_agents, _ = x.shape
        h = self.encoder(x)
        receiver = h.unsqueeze(2).expand(-1, -1, n_agents, -1)
        sender = h.unsqueeze(1).expand(-1, n_agents, -1, -1)
        reliability = torch.sigmoid(self.reliability_head(x))
        sender_reliability = reliability.unsqueeze(1).expand(-1, n_agents, -1, -1)
        gate_input = torch.cat(
            [receiver, sender, sender - receiver, sender_reliability], dim=-1
        )
        learned_gate = torch.sigmoid(self.edge_gate(gate_input).squeeze(-1))
        gate_prob = learned_gate * sender_reliability.squeeze(-1)

        eye = torch.eye(n_agents, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        candidate_mask = (adj > 0).to(adj.dtype) * (1.0 - eye)
        gate_prob = gate_prob * candidate_mask
        if hard:
            hard_gate = (
                (gate_prob >= self.gate_threshold).to(gate_prob.dtype)
                * candidate_mask
            )
            gate = hard_gate + gate_prob - gate_prob.detach()
        else:
            hard_gate = gate_prob
            gate = gate_prob

        messages = self.message(h).unsqueeze(1).expand(-1, n_agents, -1, -1)
        weights = gate.unsqueeze(-1)
        aggregated = (weights * messages).sum(dim=2)
        aggregated = aggregated / weights.sum(dim=2).clamp(min=1.0)
        aggregated = self.dropout(aggregated)

        if not return_details:
            return aggregated
        return aggregated, {
            "gate_prob": gate_prob,
            "hard_gate": hard_gate * candidate_mask,
            "candidate_mask": candidate_mask,
            "reliability": reliability.squeeze(-1),
        }

    def reliability_loss(
        self,
        clean_obs: torch.Tensor,
        corruption_prob: float = 0.3,
        noise_std: float = 0.2,
    ) -> torch.Tensor:
        """Teach the reliability head to reject missing/noisy observations."""
        clean_logits = self.reliability_head(clean_obs).squeeze(-1)
        feature_mask = (
            torch.rand_like(clean_obs) < corruption_prob
        ).to(clean_obs.dtype)
        full_dropout = (
            torch.rand(
                *clean_obs.shape[:-1], 1,
                device=clean_obs.device,
            ) < 0.25
        ).to(clean_obs.dtype)
        feature_mask = torch.maximum(feature_mask, full_dropout)
        scale = clean_obs.detach().std(dim=-1, keepdim=True).clamp(min=1.0)
        corrupted = clean_obs * (1.0 - feature_mask)
        corrupted += feature_mask * torch.randn_like(clean_obs) * scale * noise_std
        corrupt_logits = self.reliability_head(corrupted).squeeze(-1)
        return 0.5 * (
            F.binary_cross_entropy_with_logits(
                clean_logits, torch.ones_like(clean_logits)
            )
            + F.binary_cross_entropy_with_logits(
                corrupt_logits, torch.zeros_like(corrupt_logits)
            )
        )
