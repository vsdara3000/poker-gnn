from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class Player(IntEnum):
    CHANCE = -1
    P0 = 0
    P1 = 1


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2


@dataclass(frozen=True)
class State:
    """Immutable snapshot of a public+private game state."""

    player: int
    hole_cards: tuple[int | None, int | None]
    board: tuple[int, ...]
    history: tuple[int, ...]
    pot: int
    stacks: tuple[int, int]
    round: int
    terminal: bool
    chance: bool
    # Index into `history` where the *current* betting round's actions start.
    # `history` itself always spans the whole hand (needed for infoset_key:
    # a player can observe every round's public betting, not just the
    # current round's), so multi-round games use this to recover the
    # current round's local sub-sequence. Always 0 for single-round games.
    round_start: int = 0

    def infoset_key(self, player: int) -> str:
        """Player's information set: own cards + public history, not opponent cards."""
        hole = self.hole_cards[player]
        return f"{player}|h{hole}|b{self.board}|r{self.round}|a{self.history}"


class Game(ABC):
    """Two-player zero-sum poker interface used by CFR and GNN encoders."""

    name: str
    num_players: int = 2

    @abstractmethod
    def root(self) -> State:
        ...

    @abstractmethod
    def legal_actions(self, state: State) -> Sequence[int]:
        ...

    @abstractmethod
    def chance_outcomes(self, state: State) -> Sequence[tuple[int, float]]:
        """(action_or_card, probability) pairs at chance nodes."""

    @abstractmethod
    def step(self, state: State, action: int) -> State:
        ...

    @abstractmethod
    def returns(self, state: State) -> tuple[float, float]:
        """Payoffs for (P0, P1) at terminal nodes. Zero-sum."""

    @abstractmethod
    def card_names(self) -> Sequence[str]:
        ...

    @abstractmethod
    def num_cards(self) -> int:
        ...

    def num_suits(self) -> int:
        """Cards are grouped into contiguous rank-blocks of this size, i.e.
        card id `i` has rank `i // num_suits()`. 1 means every card is its
        own rank (Kuhn); Leduc's 6-card deck (3 ranks x 2 suits) is 2."""
        return 1

    def is_terminal(self, state: State) -> bool:
        return state.terminal

    def current_player(self, state: State) -> int:
        return state.player
