"""Card relation graph (rank / suit / board links).

One node per card id. Card `i` has rank `i // game.num_suits()` (Kuhn: 1 suit,
so every card is its own rank; Leduc: 2 suits, so ranks come in pairs). Two
edge types: a hand-strength chain between adjacent ranks (every card in rank
r <-> every card in rank r+1), and same-rank "pair" edges between cards that
share a rank (a no-op for Kuhn, where no rank has more than one card).
"""

from __future__ import annotations

import torch

CARD_FEATURE_DIM = 1  # normalized rank value


def card_graph(game):
    """Node features (normalized rank) and rank-adjacency/pair edges."""
    n = game.num_cards()
    num_suits = game.num_suits()
    num_ranks = n // num_suits

    def rank(card: int) -> int:
        return card // num_suits

    x = torch.tensor(
        [[rank(i) / (num_ranks - 1) if num_ranks > 1 else 0.0] for i in range(n)],
        dtype=torch.float32,
    )

    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if rank(a) == rank(b) or abs(rank(a) - rank(b)) == 1:
                edges.append((a, b))
                edges.append((b, a))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, edge_index
