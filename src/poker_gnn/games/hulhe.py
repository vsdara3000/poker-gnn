"""Heads-up limit hold'em — phase 3 (last).

Huge game tree. Do not start until Leduc works.
"""

from poker_gnn.games.base import Game, State


class HeadsUpLimitHoldem(Game):
    name = "hulhe"

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
        raise NotImplementedError

    def num_cards(self) -> int:
        return 52
