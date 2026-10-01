"""Betting-history graph: the public action sequence as a path.

One node per action taken so far (across *every* betting round of the hand
-- `state.history` never resets, see `games/base.py`), plus a leading
"start" node so an empty history still produces a non-trivial graph. Edges
chain them in order: start -> action_0 -> action_1 -> ... (both directions).
"""

from __future__ import annotations

import torch

from poker_gnn.games.base import Action

NUM_ACTION_TYPES = len(Action)
# action one-hot + is_start flag + acting-player flag + round index + normalized position
ACTION_FEATURE_DIM = NUM_ACTION_TYPES + 1 + 1 + 1 + 1


def _round_index_and_actor(i: int, round_starts: tuple[int, ...]) -> tuple[int, int]:
    """Which round action `i` (an index into the full `history`) belongs to,
    and its position parity within that round (0 = first to act).

    `round_starts` holds the history index where each round after the first
    began (see `State.round_starts`). Each game fixes who acts first in a
    given round, so (round, parity) also determines the seat."""
    round_idx = 0
    start = 0
    for r, s in enumerate(round_starts, start=1):
        if i < s:
            break
        round_idx, start = r, s
    return round_idx, (i - start) % 2


def betting_graph(state):
    """Node features [n, ACTION_FEATURE_DIM] and path edges for `state.history`,
    where n = len(history) + 1."""
    history = state.history
    n = len(history) + 1  # +1 for the start node

    feats = [[0.0] * NUM_ACTION_TYPES + [1.0, 0.0, 0.0, 0.0]]  # start node
    for i, action in enumerate(history):
        onehot = [0.0] * NUM_ACTION_TYPES
        onehot[int(action)] = 1.0
        round_idx, acting_player = _round_index_and_actor(i, state.round_starts)
        position = (i + 1) / n
        feats.append(onehot + [0.0, float(acting_player), float(round_idx), position])
    x = torch.tensor(feats, dtype=torch.float32)

    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, edge_index
