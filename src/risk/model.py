"""Failure-only semantic temporal model for proactive spillback prediction."""
from typing import Tuple

import torch
import torch.nn as nn


class SemanticSpillbackPredictor(nn.Module):
    """Predict network spillback from semantic lane-token histories."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lane_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.neighbor_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )
        self.risk_head = nn.Linear(hidden_dim, 1)

    def encode(
        self,
        sequences: torch.Tensor,
        failure_masks: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-agent temporal embeddings before risk prediction."""
        if sequences.ndim != 5:
            raise ValueError("sequences must have shape [B,T,N,L,F]")
        batch, steps, agents, _, _ = sequences.shape
        if failure_masks.shape != (batch, steps, agents):
            raise ValueError("failure_masks must have shape [B,T,N]")
        if adjacency.shape != (agents, agents):
            raise ValueError("adjacency must have shape [N,N]")

        lane_features = self.lane_encoder(sequences)
        lane_presence = sequences[..., -1:].clamp(0.0, 1.0)
        local = (lane_features * lane_presence).sum(dim=-2)
        local = local / lane_presence.sum(dim=-2).clamp(min=1.0)

        neighbor_adjacency = adjacency.to(local.dtype).clone()
        neighbor_adjacency.fill_diagonal_(0.0)
        healthy_sender = 1.0 - failure_masks
        weights = (
            neighbor_adjacency[None, None, :, :]
            * healthy_sender[:, :, None, :]
        )
        neighbor = torch.einsum("btij,btjh->btih", weights, local)
        neighbor = neighbor / weights.sum(dim=-1, keepdim=True).clamp(
            min=1.0
        )
        correction = self.neighbor_fusion(
            torch.cat([local, neighbor], dim=-1)
        )
        # Healthy agents are an exact local path; neighbor information can
        # alter representations only where a failure is explicitly detected.
        fused = local + failure_masks[..., None] * correction

        temporal_input = fused.permute(0, 2, 1, 3).reshape(
            batch * agents, steps, -1
        )
        temporal_output, _ = self.temporal(temporal_input)
        return temporal_output[:, -1].reshape(batch, agents, -1)

    def forward(
        self,
        sequences: torch.Tensor,
        failure_masks: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(sequences, failure_masks, adjacency)
        per_agent_logits = self.risk_head(embeddings).squeeze(-1)
        network_logits = per_agent_logits.max(dim=-1).values
        return network_logits, per_agent_logits


def risk_gated_logits(
    nominal_logits: torch.Tensor,
    fallback_logits: torch.Tensor,
    failure_detected: torch.Tensor,
    risk_probability: torch.Tensor,
    uncertainty: torch.Tensor,
    risk_threshold: float = 0.5,
    uncertainty_threshold: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select fallback logits only for confident high-risk failures."""
    while failure_detected.ndim < nominal_logits.ndim:
        failure_detected = failure_detected.unsqueeze(-1)
    while risk_probability.ndim < nominal_logits.ndim:
        risk_probability = risk_probability.unsqueeze(-1)
    while uncertainty.ndim < nominal_logits.ndim:
        uncertainty = uncertainty.unsqueeze(-1)
    gate = (
        failure_detected.bool()
        & (risk_probability >= risk_threshold)
        & (uncertainty <= uncertainty_threshold)
    )
    selected = torch.where(gate, fallback_logits, nominal_logits)
    return selected, gate.squeeze(-1)
