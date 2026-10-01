"""Turn a game state / infoset into GNN input: the one place solvers call to
get a graph, so they never touch the graph builders directly."""

from __future__ import annotations

from poker_gnn.graphs.infoset_graph import NODE_FEATURE_DIM, infoset_graph


class InfosetEncoder:
    """Wraps `infoset_graph` and tags the result with the bookkeeping the
    solver needs (its infoset key, which actions are legal there)."""

    node_feature_dim = NODE_FEATURE_DIM

    def encode(self, game, state, player: int):
        """Graph for `player`'s view of `state`, with `.infoset_key` and
        `.legal_actions` attached (plain Python attributes; PyG collates them
        into lists when batching)."""
        data = infoset_graph(game, state, player)
        data.infoset_key = state.infoset_key(player)
        data.legal_actions = tuple(game.legal_actions(state))
        return data
