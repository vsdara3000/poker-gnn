"""Card relation graph (rank / suit / board links).

One node per card id. Card `i` has rank `i // game.num_suits()` (Kuhn: 1 suit,
so every card is its own rank; Leduc: 2 suits, so ranks come in pairs). Two
edge types: a hand-strength chain between adjacent ranks (every card in rank
r <-> every card in rank r+1), and same-rank "pair" edges between cards that
share a rank (a no-op for Kuhn, where no rank has more than one card).
"""

from __future__ import annotations

from functools import lru_cache

import torch

CARD_FEATURE_DIM = 1  # normalized rank value


def card_graph(game):
    """Node features (normalized rank) and rank-adjacency/pair edges.

    Depends only on the deck's shape (`num_cards`/`num_suits`), never on any
    per-instance state, so it's cached: `infoset_graph` calls this once per
    infoset encoded, and at HULHE's 52-card scale the O(n^2) edge-building
    below was ~44% of total training time before caching -- pure waste,
    since it recomputed the exact same 52-node graph every single time."""
    return _card_graph_cached(game.num_cards(), game.num_suits())


@lru_cache(maxsize=None)
def _card_graph_cached(n: int, num_suits: int):
    num_ranks = n // num_suits
    ranks = [i // num_suits for i in range(n)]

    x = torch.tensor(
        [[ranks[i] / (num_ranks - 1) if num_ranks > 1 else 0.0] for i in range(n)],
        dtype=torch.float32,
    )

    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if ranks[a] == ranks[b] or abs(ranks[a] - ranks[b]) == 1:
                edges.append((a, b))
                edges.append((b, a))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, edge_index
