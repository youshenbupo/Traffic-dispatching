"""Graph Attention Network (GAT) communication layer for multi-agent TSC."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATCommLayer(nn.Module):
    """Multi-head Graph Attention communication layer with residual and layer norm.

    Args:
        in_dim: input feature dimension per agent
        out_dim: output feature dimension per agent
        edge_dim: optional edge feature dimension
        heads: number of attention heads
        dropout: dropout rate
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        assert self.head_dim * heads == out_dim, "out_dim must be divisible by heads"

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        if edge_dim > 0:
            self.W_e = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.W_e = None
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Stability improvements
        self.layer_norm = nn.LayerNorm(out_dim)
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, num_agents, in_dim]
            adj: [batch, num_agents, num_agents] or [num_agents, num_agents]
            edge_attr: [batch, num_agents, num_agents, edge_dim] or None
        Returns:
            out: [batch, num_agents, out_dim]
        """
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch, n, _ = x.shape

        h = self.W(x)  # [batch, n, out_dim]
        h = h.view(batch, n, self.heads, self.head_dim)

        attn_src = (h * self.att_src).sum(dim=-1, keepdim=True)  # [batch, n, heads, 1]
        attn_dst = (h * self.att_dst).sum(dim=-1, keepdim=True)  # [batch, n, heads, 1]

        # [batch, heads, n, n]
        attn = attn_src.expand(-1, -1, -1, n).permute(0, 2, 1, 3) + \
               attn_dst.expand(-1, -1, -1, n).permute(0, 2, 3, 1)
        attn = self.leaky_relu(attn)

        if edge_attr is not None and self.W_e is not None:
            edge_bias = self.W_e(edge_attr).permute(0, 3, 1, 2)  # [batch, heads, n, n]
            attn = attn + edge_bias

        # Apply adjacency mask
        adj = adj.unsqueeze(1).expand(-1, self.heads, -1, -1)  # [batch, heads, n, n]
        attn = attn.masked_fill(adj == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bjhd->bihd", attn, h)  # [batch, n, heads, head_dim]
        out = out.reshape(batch, n, self.out_dim)

        # Residual connection + layer norm
        residual = x if self.residual_proj is None else self.residual_proj(x)
        out = self.layer_norm(residual + self.dropout(out))

        return out.squeeze(0) if x.shape[0] == 1 else out
