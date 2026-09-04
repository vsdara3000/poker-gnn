"""Policy / value GNN over poker infoset graphs."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from poker_gnn.games.base import Action

NUM_ACTIONS = len(Action)


class PokerGNN(nn.Module):
    """Two SAGEConv layers over an infoset graph, mean-pooled, then a
    per-action policy/advantage head and a scalar value head."""

    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        """Return (action_logits [B, num_actions], value [B])."""
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        pooled = global_mean_pool(x, batch.batch)
        return self.policy_head(pooled), self.value_head(pooled).squeeze(-1)
