"""Leduc poker — phase 2 (after Kuhn CFR + GNN work).

6 cards (J, Q, K x 2 suits, ids 0-5, rank = card // 2), 2 betting rounds, 2
players. Each player antes 1, is dealt one private card, and there is a
public board card revealed between rounds.

Betting: round 0 (preflop) bet size is 2, round 1 (postflop, after the board
card) bet size is 4. Betting only ever uses check/call or bet/raise (no
re-raising past a fixed cap): at most 2 bet/raise actions per round. Player 0
acts first in both rounds. Showdown: pairing the board beats everything
else; otherwise higher rank hole card wins; equal, unpaired ranks split the
pot.

`state.history` spans the *whole hand* (both rounds), never resets: which
round-0 sequence led into round 1 is public information a real player can
see, and collapsing it away would merge genuinely distinct information sets
(e.g. "checked through" vs. "bet-called" preflop leave a different pot size
entering round 1, so a player is *not* indifferent between them). `round_
start` is the index into `history` where the current round's actions begin,
used internally to compute legal actions / costs / turn order per round.
"""

from __future__ import annotations

from poker_gnn.games.base import Action, Player, Game, State

ANTE = 1
BET_SIZES = (2, 4)  # by round
MAX_RAISES = 2
# ante + max round-0 contribution (bet + 1 raise = 2*2) + max round-1 (4*2)
STARTING_STACK = ANTE + BET_SIZES[0] * MAX_RAISES + BET_SIZES[1] * MAX_RAISES


def _replay_round(local_history: tuple[int, ...], round_: int) -> tuple[list[int], int, int]:
    """Replay the current round's action so far; return (contrib_per_player, level, raises)."""
    bet_size = BET_SIZES[round_]
    contrib = [0, 0]
    level = 0
    raises = 0
    for i, action in enumerate(local_history):
        actor = i % 2
        if action == Action.BET_RAISE:
            raises += 1
            level += bet_size
            contrib[actor] = level
        elif action == Action.CHECK_CALL:
            contrib[actor] = level
    return contrib, level, raises


def _round_ends(local_history_before: tuple[int, ...], action: int, level_before: int) -> bool:
    if action != Action.CHECK_CALL:
        return False
    if level_before > 0:
        return True  # a call closes out a bet
    return len(local_history_before) + 1 == 2  # check-check


class LeducPoker(Game):
    name = "leduc"

    def root(self) -> State:
        return State(
            player=Player.CHANCE,
            hole_cards=(None, None),
            board=(),
            history=(),
            pot=2 * ANTE,
            stacks=(STARTING_STACK - ANTE, STARTING_STACK - ANTE),
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

    def chance_outcomes(self, state: State):
        if not state.chance:
            return ()
        if state.hole_cards[0] is None:
            return tuple((card, 1 / self.num_cards()) for card in range(self.num_cards()))
        if state.hole_cards[1] is None:
            remaining = [c for c in range(self.num_cards()) if c != state.hole_cards[0]]
            return tuple((card, 1 / len(remaining)) for card in remaining)
        dealt = {state.hole_cards[0], state.hole_cards[1], *state.board}
        remaining = [c for c in range(self.num_cards()) if c not in dealt]
        return tuple((card, 1 / len(remaining)) for card in remaining)

    def step(self, state: State, action: int) -> State:
        if state.chance:
            if state.hole_cards[0] is None:
                return State(
                    player=Player.CHANCE,
                    hole_cards=(action, None),
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
                    player=Player.P0,
                    hole_cards=(state.hole_cards[0], action),
                    board=state.board,
                    history=(),
                    pot=state.pot,
                    stacks=state.stacks,
                    round=0,
                    terminal=False,
                    chance=False,
                    round_start=0,
                )
            # dealing the board card between round 0 and round 1
            return State(
                player=Player.P0,
                hole_cards=state.hole_cards,
                board=(action,),
                history=state.history,
                pot=state.pot,
                stacks=state.stacks,
                round=1,
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
                board=state.board,
                history=new_history,
                pot=pot,
                stacks=tuple(stacks),
                round=state.round,
                terminal=True,
                chance=False,
                round_start=state.round_start,
            )

        if not _round_ends(local_history, action, level):
            new_local_len = len(local_history) + 1
            return State(
                player=Player(new_local_len % 2),
                hole_cards=state.hole_cards,
                board=state.board,
                history=new_history,
                pot=pot,
                stacks=tuple(stacks),
                round=state.round,
                terminal=False,
                chance=False,
                round_start=state.round_start,
            )

        if state.round == 1:
            return State(
                player=Player.P0,
                hole_cards=state.hole_cards,
                board=state.board,
                history=new_history,
                pot=pot,
                stacks=tuple(stacks),
                round=1,
                terminal=True,
                chance=False,
                round_start=state.round_start,
            )

        return State(
            player=Player.CHANCE,
            hole_cards=state.hole_cards,
            board=state.board,
            history=new_history,
            pot=pot,
            stacks=tuple(stacks),
            round=1,
            terminal=False,
            chance=True,
            round_start=state.round_start,  # unused pre-deal; reset once the board card lands
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
            folder = (len(local_history) - 1) % 2
            winner = 1 - folder
        else:
            rank0 = state.hole_cards[0] // 2
            rank1 = state.hole_cards[1] // 2
            board_rank = state.board[0] // 2
            paired0 = rank0 == board_rank
            paired1 = rank1 == board_rank
            if paired0 and not paired1:
                winner = 0
            elif paired1 and not paired0:
                winner = 1
            elif rank0 > rank1:
                winner = 0
            elif rank1 > rank0:
                winner = 1
            else:
                return (0.0, 0.0)  # split pot: unpaired, equal ranks

        loser = 1 - winner
        payoff = float(contributions[loser])
        result = [0.0, 0.0]
        result[winner] = payoff
        result[loser] = -payoff
        return (result[0], result[1])

    def card_names(self):
        return ("Js", "Jh", "Qs", "Qh", "Ks", "Kh")

    def num_cards(self) -> int:
        return 6

    def num_suits(self) -> int:
        return 2
