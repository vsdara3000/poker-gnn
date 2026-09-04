"""Build a graph for one infoset (hero's card + board + betting history).

Conceptually heterogeneous — a card sub-graph glued to a betting sub-graph —
but represented as a single homogeneous `torch_geometric.data.Data` whose
node features carry a type flag, so one GNN can message-pass over both
without PyG's HeteroData machinery. That's fine at Kuhn/Leduc scale; revisit
for HULHE if the two node types need genuinely different conv weights.

Never encodes the opponent's hole card: only the rank at `state.hole_cards
[player]` is flagged as "hero", and ranks in `state.board` are flagged as
"board" (both empty in Kuhn, which has no board). Every other card node is
indistinguishable info the GNN can't use to infer the opponent's hand.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from poker_gnn.graphs.betting_graph import ACTION_FEATURE_DIM, betting_graph
from poker_gnn.graphs.card_graph import CARD_FEATURE_DIM, card_graph

# [is_card_node, is_action_node, is_hero_card, is_board_card] + card features + action features
NODE_TYPE_DIM = 4
NODE_FEATURE_DIM = NODE_TYPE_DIM + CARD_FEATURE_DIM + ACTION_FEATURE_DIM


def infoset_graph(game, state, player: int) -> Data:
    """Encode what `player` sees: their own hole card, the public board
    card(s), and the public betting history, as one graph the GNN can
    consume."""
    card_x, card_edges = card_graph(game)
    action_x, action_edges = betting_graph(state)
    num_cards = card_x.shape[0]
    num_actions = action_x.shape[0]
    hero_card = state.hole_cards[player]
    board_cards = set(state.board)

    nodes = []
    for i in range(num_cards):
        is_hero = 1.0 if i == hero_card else 0.0
        is_board = 1.0 if i in board_cards else 0.0
        nodes.append(
            [1.0, 0.0, is_hero, is_board] + card_x[i].tolist() + [0.0] * ACTION_FEATURE_DIM
        )
    for i in range(num_actions):
        nodes.append([0.0, 1.0, 0.0, 0.0] + [0.0] * CARD_FEATURE_DIM + action_x[i].tolist())
    x = torch.tensor(nodes, dtype=torch.float32)

    edges = [(a, b) for a, b in card_edges.t().tolist()]
    offset = num_cards
    edges += [(a + offset, b + offset) for a, b in action_edges.t().tolist()]
    # fuse every publicly-known card (hero's hole card, board cards) with
    # every betting node so the GNN can mix hand strength with the action
    # sequence
    known_cards = set(board_cards)
    if hero_card is not None:
        known_cards.add(hero_card)
    for card in known_cards:
        for i in range(num_actions):
            edges.append((card, offset + i))
            edges.append((offset + i, card))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    return Data(x=x, edge_index=edge_index)
