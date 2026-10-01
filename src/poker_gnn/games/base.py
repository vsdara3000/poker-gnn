"""Shared game interface: `Player`, `Action`, the immutable `State`, and the
abstract `Game` every rule set implements.

Everything downstream (tabular CFR, MCCFR, Deep CFR, the graph encoders,
exploitability / baseline / LBR evaluation) is written against this file
only, never against a specific game.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class Player(IntEnum):
    """Whose turn it is. CHANCE marks card-dealing nodes."""

    CHANCE = -1
    P0 = 0
    P1 = 1


class Action(IntEnum):
    """The three limit-poker actions, shared by every game. CHECK_CALL and
    BET_RAISE each cover two moves (which one depends on whether the actor
    faces a bet), which keeps the action space a fixed size of 3 for the
    networks' output heads."""

    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2


@dataclass(frozen=True)
class State:
    """Immutable snapshot of a game state, including *both* players' private
    cards. Solvers must go through `infoset_key` / the encoders to get what
    one player is allowed to see.

    `pot` and `stacks` are in chips; `round` is the betting round (0-based);
    `player` is the actor (or `Player.CHANCE`, with `chance=True`)."""

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
    # A second hole card per player, for games that deal more than one (only
    # HULHE so far). Kuhn/Leduc deal exactly one and leave this (None, None),
    # so infoset_key's formatting of it is a no-op for them.
    hole_cards2: tuple[int | None, int | None] = (None, None)

    def infoset_key(self, player: int) -> str:
        """Player's information set: own cards + public history, not opponent cards."""
        hole = self.hole_cards[player]
        hole2 = self.hole_cards2[player]
        extra = f"|h2{hole2}" if hole2 is not None else ""
        return f"{player}|h{hole}{extra}|b{self.board}|r{self.round}|a{self.history}"


class Game(ABC):
    """Two-player zero-sum poker interface used by CFR and GNN encoders."""

    name: str
    num_players: int = 2

    @abstractmethod
    def root(self) -> State:
        """Start of a hand: antes/blinds posted, no cards dealt (a chance node)."""

    @abstractmethod
    def legal_actions(self, state: State) -> Sequence[int]:
        """`Action`s available to the player to act; empty at chance/terminal nodes."""

    @abstractmethod
    def chance_outcomes(self, state: State) -> Sequence[tuple[int, float]]:
        """(action_or_card, probability) pairs at chance nodes."""

    @abstractmethod
    def step(self, state: State, action: int) -> State:
        """Return the successor state. `action` is a card id at chance nodes,
        an `Action` otherwise. Never mutates `state`."""

    @abstractmethod
    def returns(self, state: State) -> tuple[float, float]:
        """Net chip payoffs for (P0, P1) at a terminal node (winner gets the
        loser's total contribution). Zero-sum."""

    @abstractmethod
    def card_names(self) -> Sequence[str]:
        """Human-readable name for each card id, e.g. ("J", "Q", "K")."""

    @abstractmethod
    def num_cards(self) -> int:
        """Deck size. Card ids are 0..num_cards()-1; also the card graph's node count."""

    def num_suits(self) -> int:
        """Cards are grouped into contiguous rank-blocks of this size, i.e.
        card id `i` has rank `i // num_suits()`. 1 means every card is its
        own rank (Kuhn); Leduc's 6-card deck (3 ranks x 2 suits) is 2;
        HULHE's 52-card deck is 4."""
        return 1

    def hand_strength(self, hole_cards: tuple[int, ...], board: tuple[int, ...]) -> float | None:
        """Optional [0, 1] made-hand-strength scalar for `hole_cards` given
        `board`, so the GNN gets an explicit hint instead of having to
        rediscover this game's hand rankings from raw graph structure alone
        -- worth little in Kuhn/Leduc (rank comparison is nearly the whole
        game already, and the card graph's rank/pair edges cover it), but
        HULHE's flush/straight/full-house rankings are a lot to ask a small
        GNN to learn purely from message passing over a 52-card graph.
        None if not applicable (e.g. too few cards revealed yet, or the
        game doesn't define one)."""
        return None

    def sample_showdown_equity(self, state: State, player: int, rng) -> float | None:
        """One unbiased sample of `player`'s showdown result against the
        opponent's *actual* hole cards (1 win, 0.5 split, 0 loss), dealing
        any missing board cards at random from `rng`. Uses information
        `player` can't see, so it's only for training targets (DeepCFR's
        range network), never for play. None if the game doesn't define it."""
        return None

    def is_terminal(self, state: State) -> bool:
        return state.terminal

    def current_player(self, state: State) -> int:
        return state.player
