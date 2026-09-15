"""Policy / value GNN over poker infoset graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from poker_gnn.games.base import Action
from poker_gnn.graphs.infoset_graph import ROLE_ACTION, ROLE_INFORMATIVE_CARD

NUM_ACTIONS = len(Action)


class PokerGNN(nn.Module):
    """Two SAGEConv layers over an infoset graph, then pooled and fed to a
    per-action policy/advantage head and a scalar value head.

    Pooling is masked rather than one mean over every node: only
    "informative" card nodes (hero's/board's, per `node_role`) and action
    nodes get pooled, each into their own vector, concatenated with the
    game's own `hand_strength` hint. A flat mean over every node would
    dilute this signal badly at HULHE's scale, where as few as 7 of 52
    card nodes are ever informative -- the other ~45 are identical,
    contentless placeholders that would otherwise wash out the average.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # informative-card pool + action pool + hand_strength + hand_strength_known
        pooled_dim = 2 * hidden_dim + 2
        self.policy_head = nn.Linear(pooled_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(pooled_dim, 1)

    def forward(self, batch):
        """Return (action_logits [B, num_actions], value [B])."""
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))

        num_graphs = batch.num_graphs
        card_mask = batch.node_role == ROLE_INFORMATIVE_CARD
        action_mask = batch.node_role == ROLE_ACTION
        card_pool = global_mean_pool(x[card_mask], batch.batch[card_mask], size=num_graphs)
        action_pool = global_mean_pool(x[action_mask], batch.batch[action_mask], size=num_graphs)

        pooled = torch.cat(
            [card_pool, action_pool, batch.hand_strength, batch.hand_strength_known], dim=1
        )
        return self.policy_head(pooled), self.value_head(pooled).squeeze(-1)
