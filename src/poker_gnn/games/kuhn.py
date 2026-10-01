"""Kuhn poker: the smallest game in the repo (phase 1 of docs/PLAN.md).

3 cards, 1 betting round, 2 players, 12 information sets. Small enough to
check by hand and has a known closed-form Nash equilibrium, so it is the
sanity game every solver and the GNN pipeline are validated on first.

Rules: deck {J, Q, K} (card ids 0, 1, 2), each player is dealt one card,
each antes 1. Player 0 acts first and may check or bet 1; whichever player
faces a bet may only fold or call (no re-raising). Showdown pays the higher
card.
"""

from poker_gnn.games.base import Action, Player, Game, State

STARTING_STACK = 2  # ante 1 + one bet/call of 1
ANTE = 1

# Kuhn's whole betting tree is five lines, so terminal detection is a lookup.
_TERMINAL_HISTORIES = {
    (Action.CHECK_CALL, Action.CHECK_CALL),
    (Action.BET_RAISE, Action.FOLD),
    (Action.BET_RAISE, Action.CHECK_CALL),
    (Action.CHECK_CALL, Action.BET_RAISE, Action.FOLD),
    (Action.CHECK_CALL, Action.BET_RAISE, Action.CHECK_CALL),
}


class KuhnPoker(Game):
    """Two-player Kuhn poker. Card ids 0, 1, 2 = J, Q, K."""

    name = "kuhn"

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
        )

    def legal_actions(self, state: State):
        if state.chance or state.terminal:
            return ()
        history = state.history
        if len(history) == 0 or history == (Action.CHECK_CALL,):
            return (Action.CHECK_CALL, Action.BET_RAISE)
        # facing a bet: fold or call only, no re-raises in Kuhn
        return (Action.FOLD, Action.CHECK_CALL)

    def chance_outcomes(self, state: State):
        if not state.chance:
            return ()
        if state.hole_cards[0] is None:
            return tuple((card, 1 / 3) for card in range(self.num_cards()))
        remaining = [c for c in range(self.num_cards()) if c != state.hole_cards[0]]
        return tuple((card, 1 / len(remaining)) for card in remaining)

    def step(self, state: State, action: int) -> State:
        if state.chance:
            if state.hole_cards[0] is None:
                hole_cards = (action, None)
                return State(
                    player=Player.CHANCE,
                    hole_cards=hole_cards,
                    board=state.board,
                    history=state.history,
                    pot=state.pot,
                    stacks=state.stacks,
                    round=state.round,
                    terminal=False,
                    chance=True,
                )
            hole_cards = (state.hole_cards[0], action)
            return State(
                player=Player.P0,
                hole_cards=hole_cards,
                board=state.board,
                history=state.history,
                pot=state.pot,
                stacks=state.stacks,
                round=state.round,
                terminal=False,
                chance=False,
            )

        actor = state.player
        facing_bet = bool(state.history) and state.history[-1] == Action.BET_RAISE
        if action == Action.BET_RAISE:
            cost = 1
        elif action == Action.CHECK_CALL and facing_bet:
            cost = 1
        else:
            cost = 0

        stacks = list(state.stacks)
        stacks[actor] -= cost
        history = state.history + (action,)
        terminal = history in _TERMINAL_HISTORIES
        # players strictly alternate starting with P0; terminal states park on P0
        next_player = Player.P0 if terminal else Player((len(history)) % 2)

        return State(
            player=next_player,
            hole_cards=state.hole_cards,
            board=state.board,
            history=history,
            pot=state.pot + cost,
            stacks=tuple(stacks),
            round=state.round,
            terminal=terminal,
            chance=False,
        )

    def returns(self, state: State) -> tuple[float, float]:
        if not state.terminal:
            raise ValueError("returns() called on a non-terminal state")

        history = state.history
        if history[-1] == Action.FOLD:
            folder = (len(history) - 1) % 2
            winner = 1 - folder
        else:
            winner = 0 if state.hole_cards[0] > state.hole_cards[1] else 1
        loser = 1 - winner

        # the winner nets exactly what the loser put in (ante + any bet/call)
        contributions = (
            STARTING_STACK - state.stacks[0],
            STARTING_STACK - state.stacks[1],
        )
        payoff = float(contributions[loser])

        result = [0.0, 0.0]
        result[winner] = payoff
        result[loser] = -payoff
        return (result[0], result[1])

    def card_names(self):
        return ("J", "Q", "K")

    def num_cards(self) -> int:
        return 3

    def sample_showdown_equity(self, state: State, player: int, rng) -> float | None:
        # no board and no ties in Kuhn, so this is deterministic and `rng` is unused
        return 1.0 if state.hole_cards[player] > state.hole_cards[1 - player] else 0.0
