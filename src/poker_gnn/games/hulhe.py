"""Heads-up limit hold'em — phase 3 (last).

52 cards, 2 hole cards each, 4 betting rounds (preflop / flop / turn /
river), community board grows 0 -> 3 -> 4 -> 5. Small bet is 2 (preflop,
flop), big bet is 4 (turn, river); standard limit cap of 4 bet/raise
actions per round. Blinds: P0 (button) posts the small blind and acts
first preflop; P1 (big blind) acts first in every later round -- the
"big blind option" (P0 completes the blind, P1 still gets to check or
raise before preflop can close) falls out of the same round-closing rule
Leduc uses, generalized to a round that can *start* with a nonzero level
(see `_round_ends`).

Huge game tree (~10^14 information sets): unlike Kuhn/Leduc, nothing here
can be enumerated exhaustively. `chance_outcomes` only ever needs to
enumerate the *current* chance node's own outcomes (at most 52 remaining
cards), never the whole tree, so it stays cheap -- but `TabularCFR` /
`DeepCFR`'s full-width traversal, and `eval/exploitability`'s exact best
response, both need a sampling-based replacement before they can run here.
"""

from __future__ import annotations

from poker_gnn.games.base import Action, Player, Game, State
from poker_gnn.utils.cards import STRAIGHT_FLUSH, deck_card_names, evaluate_hand

SMALL_BLIND = 1
BIG_BLIND = 2
BET_SIZES = (2, 2, 4, 4)  # preflop, flop, turn, river
MAX_RAISES = 4  # standard limit cap: bet + 3 raises
BOARD_TARGET_LEN = {1: 3, 2: 4, 3: 5}  # community cards once round `r` begins
# max a player can ever contribute: preflop (blind + capped raises) plus
# flop/turn/river each capped at their own bet size
STARTING_STACK = (
    BIG_BLIND
    + MAX_RAISES * BET_SIZES[0]
    + MAX_RAISES * BET_SIZES[1]
    + MAX_RAISES * BET_SIZES[2]
    + MAX_RAISES * BET_SIZES[3]
)


def _first_actor(round_: int) -> int:
    """P0 (button/small blind) acts first preflop; P1 (big blind) acts
    first in every later round."""
    return 0 if round_ == 0 else 1


def _round_opening(round_: int) -> tuple[list[int], int]:
    """(contrib_per_player, level) before either player has voluntarily
    acted this round. Preflop starts at the blinds; every later round
    starts at zero, same as Leduc."""
    if round_ == 0:
        return [SMALL_BLIND, BIG_BLIND], BIG_BLIND
    return [0, 0], 0


def _replay_round(local_history: tuple[int, ...], round_: int) -> tuple[list[int], int, int]:
    bet_size = BET_SIZES[round_]
    contrib, level = _round_opening(round_)
    raises = 0
    first_actor = _first_actor(round_)
    for i, action in enumerate(local_history):
        actor = (first_actor + i) % 2
        if action == Action.BET_RAISE:
            raises += 1
            level += bet_size
            contrib[actor] = level
        elif action == Action.CHECK_CALL:
            contrib[actor] = level
    return contrib, level, raises


def _round_ends(raises_before: int, local_history_before: tuple[int, ...], action: int) -> bool:
    if action != Action.CHECK_CALL:
        return False
    if raises_before > 0:
        return True  # a call closes out the last raise
    # nobody has voluntarily bet/raised yet this round (the level, if any,
    # is just the opening blinds): closes on the second such check/call --
    # this is what gives the big blind its preflop option.
    return len(local_history_before) >= 1


class HeadsUpLimitHoldem(Game):
    name = "hulhe"

    def root(self) -> State:
        return State(
            player=Player.CHANCE,
            hole_cards=(None, None),
            hole_cards2=(None, None),
            board=(),
            history=(),
            pot=SMALL_BLIND + BIG_BLIND,
            stacks=(STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND),
            round=0,
            terminal=False,
            chance=True,
            round_start=0,
        )

    def legal_actions(self, state: State):
        if state.chance or state.terminal:
            return ()
        local_history = state.history[state.round_start :]
        contrib, level, raises = _replay_round(local_history, state.round)
        actor = state.player
        actions = []
        if contrib[actor] < level:
            actions.append(Action.FOLD)
        actions.append(Action.CHECK_CALL)
        if raises < MAX_RAISES:
            actions.append(Action.BET_RAISE)
        return tuple(actions)

    def _remaining_cards(self, state: State) -> list[int]:
        dealt = set(state.board)
        dealt.update(c for c in state.hole_cards if c is not None)
        dealt.update(c for c in state.hole_cards2 if c is not None)
        return [c for c in range(self.num_cards()) if c not in dealt]

    def chance_outcomes(self, state: State):
        if not state.chance:
            return ()
        remaining = self._remaining_cards(state)
        return tuple((card, 1 / len(remaining)) for card in remaining)

    def step(self, state: State, action: int) -> State:
        if state.chance:
            if state.hole_cards[0] is None:
                return State(
                    player=Player.CHANCE,
                    hole_cards=(action, state.hole_cards[1]),
                    hole_cards2=state.hole_cards2,
                    board=state.board,
                    history=state.history,
                    pot=state.pot,
                    stacks=state.stacks,
                    round=state.round,
                    terminal=False,
                    chance=True,
                    round_start=state.round_start,
                )
            if state.hole_cards2[0] is None:
                return State(
                    player=Player.CHANCE,
                    hole_cards=state.hole_cards,
                    hole_cards2=(action, state.hole_cards2[1]),
                    board=state.board,
                    history=state.history,
                    pot=state.pot,
                    stacks=state.stacks,
                    round=state.round,
                    terminal=False,
                    chance=True,
                    round_start=state.round_start,
                )
            if state.hole_cards[1] is None:
                return State(
                    player=Player.CHANCE,
                    hole_cards=(state.hole_cards[0], action),
                    hole_cards2=state.hole_cards2,
                    board=state.board,
                    history=state.history,
                    pot=state.pot,
                    stacks=state.stacks,
                    round=state.round,
                    terminal=False,
                    chance=True,
                    round_start=state.round_start,
                )
            if state.hole_cards2[1] is None:
                return State(
                    player=Player(_first_actor(0)),
                    hole_cards=state.hole_cards,
                    hole_cards2=(state.hole_cards2[0], action),
                    board=state.board,
                    history=(),
                    pot=state.pot,
                    stacks=state.stacks,
                    round=0,
                    terminal=False,
                    chance=False,
                    round_start=0,
                )
            # community card, for the round already recorded in state.round
            new_board = state.board + (action,)
            target = BOARD_TARGET_LEN[state.round]
            if len(new_board) < target:
                return State(
                    player=Player.CHANCE,
                    hole_cards=state.hole_cards,
                    hole_cards2=state.hole_cards2,
                    board=new_board,
                    history=state.history,
                    pot=state.pot,
                    stacks=state.stacks,
                    round=state.round,
                    terminal=False,
                    chance=True,
                    round_start=state.round_start,
                )
            return State(
                player=Player(_first_actor(state.round)),
                hole_cards=state.hole_cards,
                hole_cards2=state.hole_cards2,
                board=new_board,
                history=state.history,
                pot=state.pot,
                stacks=state.stacks,
                round=state.round,
                terminal=False,
                chance=False,
                round_start=len(state.history),
            )

        actor = state.player
        local_history = state.history[state.round_start :]
        contrib, level, raises = _replay_round(local_history, state.round)
        bet_size = BET_SIZES[state.round]
        if action == Action.BET_RAISE:
            cost = level + bet_size - contrib[actor]
        elif action == Action.CHECK_CALL:
            cost = level - contrib[actor]
        else:
            cost = 0

        stacks = list(state.stacks)
        stacks[actor] -= cost
        pot = state.pot + cost
        new_history = state.history + (action,)

        if action == Action.FOLD:
            return State(
                player=Player.P0,
                hole_cards=state.hole_cards,
                hole_cards2=state.hole_cards2,
                board=state.board,
                history=new_history,
                pot=pot,
                stacks=tuple(stacks),
                round=state.round,
                terminal=True,
                chance=False,
                round_start=state.round_start,
            )

        if not _round_ends(raises, local_history, action):
            new_local_len = len(local_history) + 1
            next_actor = (_first_actor(state.round) + new_local_len) % 2
            return State(
                player=Player(next_actor),
                hole_cards=state.hole_cards,
                hole_cards2=state.hole_cards2,
                board=state.board,
                history=new_history,
                pot=pot,
                stacks=tuple(stacks),
                round=state.round,
                terminal=False,
                chance=False,
                round_start=state.round_start,
            )

        if state.round == 3:
            return State(
                player=Player.P0,
                hole_cards=state.hole_cards,
                hole_cards2=state.hole_cards2,
                board=state.board,
                history=new_history,
                pot=pot,
                stacks=tuple(stacks),
                round=3,
                terminal=True,
                chance=False,
                round_start=state.round_start,
            )

        return State(
            player=Player.CHANCE,
            hole_cards=state.hole_cards,
            hole_cards2=state.hole_cards2,
            board=state.board,
            history=new_history,
            pot=pot,
            stacks=tuple(stacks),
            round=state.round + 1,
            terminal=False,
            chance=True,
            round_start=state.round_start,  # unused until the new round's community cards land
        )

    def returns(self, state: State) -> tuple[float, float]:
        if not state.terminal:
            raise ValueError("returns() called on a non-terminal state")

        contributions = (
            STARTING_STACK - state.stacks[0],
            STARTING_STACK - state.stacks[1],
        )

        history = state.history
        if history and history[-1] == Action.FOLD:
            local_history = history[state.round_start :]
            folder = (_first_actor(state.round) + len(local_history) - 1) % 2
            winner = 1 - folder
        else:
            p0_hand = [state.hole_cards[0], state.hole_cards2[0], *state.board]
            p1_hand = [state.hole_cards[1], state.hole_cards2[1], *state.board]
            score0 = evaluate_hand(p0_hand)
            score1 = evaluate_hand(p1_hand)
            if score0 > score1:
                winner = 0
            elif score1 > score0:
                winner = 1
            else:
                return (0.0, 0.0)  # split pot

        loser = 1 - winner
        payoff = float(contributions[loser])
        result = [0.0, 0.0]
        result[winner] = payoff
        result[loser] = -payoff
        return (result[0], result[1])

    def card_names(self):
        return deck_card_names()

    def num_cards(self) -> int:
        return 52

    def num_suits(self) -> int:
        return 4

    def hand_strength(self, hole_cards: tuple[int, ...], board: tuple[int, ...]) -> float | None:
        cards = [*hole_cards, *board]
        if len(cards) < 5:
            return None  # preflop: no made-hand ranking exists yet
        category = evaluate_hand(cards)[0]
        return category / STRAIGHT_FLUSH
