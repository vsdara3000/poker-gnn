"""Turn a game state / infoset into GNN input tensors."""


class InfosetEncoder:
    def encode(self, game, state, player: int):
        raise NotImplementedError
