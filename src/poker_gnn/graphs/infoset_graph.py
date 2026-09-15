"""Build a graph for one infoset (hero's card(s) + board + betting history).

Conceptually heterogeneous — a card sub-graph glued to a betting sub-graph —
but represented as a single homogeneous `torch_geometric.data.Data` whose
node features carry a type flag, so one GNN can message-pass over both
without PyG's HeteroData machinery. That's fine at Kuhn/Leduc scale; revisit
for HULHE if the two node types need genuinely different conv weights.

Never encodes the opponent's hole card(s): only `state.hole_cards[player]`
and `state.hole_cards2[player]` (HULHE deals two; Kuhn/Leduc leave the
latter None) are flagged "hero", and ranks in `state.board` are flagged
"board" (empty in Kuhn, which has no board). Every other card node is
indistinguishable info the GNN can't use to infer the opponent's hand.

Two things exist here specifically so `PokerGNN` doesn't have to rediscover
poker hand rankings from raw graph structure alone, which is a much bigger
ask at HULHE's 52-card scale than at Kuhn/Leduc's:
- `data.node_role` flags every node as an "other" card (0), an informative
  card -- hero's or the board's (1), or an action node (2). `PokerGNN`
  pools informative-card and action nodes separately using this, instead
  of one mean over everything: at HULHE scale as few as 7 of 52 card nodes
  are ever informative, so a single mean would mostly average in identical,
  contentless "other card" nodes and dilute the signal that matters.
- `data.hand_strength` is `game.hand_strength(...)`'s explicit [0, 1]
  made-hand-strength scalar (None -> 0.0, with `data.hand_strength_known`
  set accordingly), concatenated onto the pooled representation directly.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from poker_gnn.graphs.betting_graph import ACTION_FEATURE_DIM, betting_graph
from poker_gnn.graphs.card_graph import CARD_FEATURE_DIM, card_graph

# [is_card_node, is_action_node, is_hero_card, is_board_card] + card features + action features
NODE_TYPE_DIM = 4
NODE_FEATURE_DIM = NODE_TYPE_DIM + CARD_FEATURE_DIM + ACTION_FEATURE_DIM

ROLE_OTHER_CARD = 0
ROLE_INFORMATIVE_CARD = 1
ROLE_ACTION = 2


def infoset_graph(game, state, player: int) -> Data:
    """Encode what `player` sees: their own hole card(s), the public board
    card(s), and the public betting history, as one graph the GNN can
    consume."""
    card_x, card_edges = card_graph(game)
    action_x, action_edges = betting_graph(state)
    num_cards = card_x.shape[0]
    num_actions = action_x.shape[0]
    hero_cards = tuple(
        c for c in (state.hole_cards[player], state.hole_cards2[player]) if c is not None
    )
    hero_card_set = set(hero_cards)
    board_cards = set(state.board)

    nodes = []
    node_role = []
    for i in range(num_cards):
        is_hero = 1.0 if i in hero_card_set else 0.0
        is_board = 1.0 if i in board_cards else 0.0
        nodes.append(
            [1.0, 0.0, is_hero, is_board] + card_x[i].tolist() + [0.0] * ACTION_FEATURE_DIM
        )
        node_role.append(ROLE_INFORMATIVE_CARD if (is_hero or is_board) else ROLE_OTHER_CARD)
    for i in range(num_actions):
        nodes.append([0.0, 1.0, 0.0, 0.0] + [0.0] * CARD_FEATURE_DIM + action_x[i].tolist())
        node_role.append(ROLE_ACTION)
    x = torch.tensor(nodes, dtype=torch.float32)

    edges = [(a, b) for a, b in card_edges.t().tolist()]
    offset = num_cards
    edges += [(a + offset, b + offset) for a, b in action_edges.t().tolist()]
    # fuse every publicly-known card (hero's hole card(s), board cards) with
    # every betting node so the GNN can mix hand strength with the action
    # sequence
    known_cards = board_cards | hero_card_set
    for card in known_cards:
        for i in range(num_actions):
            edges.append((card, offset + i))
            edges.append((offset + i, card))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    strength = game.hand_strength(hero_cards, state.board)
    data = Data(x=x, edge_index=edge_index)
    data.node_role = torch.tensor(node_role, dtype=torch.long)
    data.hand_strength = torch.tensor([[strength if strength is not None else 0.0]])
    data.hand_strength_known = torch.tensor([[1.0 if strength is not None else 0.0]])
    return data
