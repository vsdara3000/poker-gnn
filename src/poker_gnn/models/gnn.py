"""Policy / value GNN over poker infoset graphs."""


class PokerGNN:
    def forward(self, batch):
        """Return action logits and/or values."""
        raise NotImplementedError
