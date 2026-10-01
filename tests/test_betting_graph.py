"""Betting-graph action features must carry each action's true round.

Regression test: the encoder used to know only the current round's start,
so on the HULHE river every preflop/flop action was labeled as the turn.
"""

from poker_gnn.games.base import Action
from poker_gnn.games.hulhe import HeadsUpLimitHoldem
from poker_gnn.graphs.betting_graph import NUM_ACTION_TYPES, betting_graph

# columns after the action one-hot: is_start, position parity, round index, position
PARITY_COL = NUM_ACTION_TYPES + 1
ROUND_COL = NUM_ACTION_TYPES + 2


def _river_state():
    game = HeadsUpLimitHoldem()
    state = game.root()
    for card in (0, 1, 4, 5):
        state = game.step(state, card)
    # preflop: raise, call | flop: check, check | turn: bet, call | river: bet
    lines = [
        (Action.BET_RAISE, Action.CHECK_CALL),
        (Action.CHECK_CALL, Action.CHECK_CALL),
        (Action.BET_RAISE, Action.CHECK_CALL),
    ]
    for line in lines:
        for action in line:
            state = game.step(state, action)
        while state.chance:
            state = game.step(state, game.chance_outcomes(state)[0][0])
    state = game.step(state, Action.BET_RAISE)
    assert state.round == 3
    return state


def test_every_action_is_labeled_with_its_own_round():
    x, _ = betting_graph(_river_state())
    rounds = x[1:, ROUND_COL].tolist()  # skip the start node
    assert rounds == [0, 0, 1, 1, 2, 2, 3]


def test_parity_restarts_at_each_round():
    x, _ = betting_graph(_river_state())
    parity = x[1:, PARITY_COL].tolist()
    assert parity == [0, 1, 0, 1, 0, 1, 0]
