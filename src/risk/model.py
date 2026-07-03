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
        self.auxiliary_head = nn.Linear(hidden_dim, 4)

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

    def predict_targets(
        self,
        sequences: torch.Tensor,
        failure_masks: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict binary risk and normalized continuous future targets."""
        embeddings = self.encode(sequences, failure_masks, adjacency)
        per_agent_logits = self.risk_head(embeddings).squeeze(-1)
        network_logits = per_agent_logits.max(dim=-1).values
        network_embedding = embeddings.max(dim=1).values
        auxiliary_targets = self.auxiliary_head(network_embedding)
        return network_logits, per_agent_logits, auxiliary_targets


class GraphInterventionBenefitPredictor(nn.Module):
    """Predict whether a candidate fallback switch is beneficial.

    The model keeps the lane-level semantic encoder used by the risk model, then
    performs explicit message passing on the traffic-intersection graph. A
    scalar candidate switch time is embedded and injected into every node so the
    gate can learn non-monotonic intervention windows such as "step 60 is bad
    but step 105 is safe" for the same seed.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        max_step: float = 720.0,
    ):
        super().__init__()
        self.max_step = float(max_step)
        self.lane_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_layer_1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_layer_2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.benefit_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _normalized_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        agents = adjacency.shape[0]
        graph = adjacency.to(dtype=torch.float32).clone()
        graph = graph + torch.eye(agents, device=graph.device)
        return graph / graph.sum(dim=-1, keepdim=True).clamp(min=1.0)

    def encode_nodes(
        self,
        sequences: torch.Tensor,
        failure_masks: torch.Tensor,
        start_steps: torch.Tensor,
    ) -> torch.Tensor:
        if sequences.ndim != 5:
            raise ValueError("sequences must have shape [B,T,N,L,F]")
        batch, steps, agents, _, _ = sequences.shape
        if failure_masks.shape != (batch, steps, agents):
            raise ValueError("failure_masks must have shape [B,T,N]")

        lane_features = self.lane_encoder(sequences)
        lane_presence = sequences[..., -1:].clamp(0.0, 1.0)
        local = (lane_features * lane_presence).sum(dim=-2)
        local = local / lane_presence.sum(dim=-2).clamp(min=1.0)

        temporal_input = local.permute(0, 2, 1, 3).reshape(
            batch * agents, steps, -1
        )
        temporal_output, _ = self.temporal(temporal_input)
        temporal_embedding = temporal_output[:, -1].reshape(
            batch, agents, -1
        )
        if start_steps.ndim == 1:
            start_steps = start_steps[:, None]
        time_feature = (start_steps.to(sequences.device).float()
                        / self.max_step).clamp(0.0, 1.5)
        time_embedding = self.time_encoder(time_feature)
        time_embedding = time_embedding[:, None, :].expand(
            batch, agents, -1
        )
        current_failure = failure_masks[:, -1, :, None]
        return self.node_projection(torch.cat([
            temporal_embedding,
            time_embedding,
            current_failure,
        ], dim=-1))

    def forward(
        self,
        sequences: torch.Tensor,
        failure_masks: torch.Tensor,
        adjacency: torch.Tensor,
        start_steps: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.encode_nodes(sequences, failure_masks, start_steps)
        graph = self._normalized_adjacency(adjacency).to(nodes.device)

        neighbor = torch.einsum("ij,bjh->bih", graph, nodes)
        nodes = self.graph_layer_1(torch.cat([nodes, neighbor], dim=-1))
        neighbor = torch.einsum("ij,bjh->bih", graph, nodes)
        nodes = self.graph_layer_2(torch.cat([nodes, neighbor], dim=-1))

        pooled = torch.cat([
            nodes.max(dim=1).values,
            nodes.mean(dim=1),
        ], dim=-1)
        return self.benefit_head(pooled).squeeze(-1)


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
