"""Leduc poker — phase 2 (after Kuhn CFR + GNN work).

6 cards, 2 rounds, 2 players. Leave as a stub until then.
"""

from poker_gnn.games.base import Game, State


class LeducPoker(Game):
    name = "leduc"

    def root(self) -> State:
        raise NotImplementedError

    def legal_actions(self, state: State):
        raise NotImplementedError

    def chance_outcomes(self, state: State):
        raise NotImplementedError

    def step(self, state: State, action: int) -> State:
        raise NotImplementedError

    def returns(self, state: State) -> tuple[float, float]:
        raise NotImplementedError

    def card_names(self):
        return ("Js", "Jh", "Qs", "Qh", "Ks", "Kh")

    def num_cards(self) -> int:
        return 6
