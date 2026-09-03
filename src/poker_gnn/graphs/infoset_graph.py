"""Build a heterogeneous graph for one infoset (cards + public state)."""


def infoset_graph(game, state, player: int):
    """Encode what `player` sees as a graph the GNN can consume."""
    raise NotImplementedError
