"""Tabular CFR should approach Nash on Kuhn.

Reference: Kuhn poker has a known family of Nash equilibria parameterized
by alpha in [0, 1/3] (see Kuhn 1950 / the "Kuhn poker" Wikipedia article).
Only player 0's opening-bet frequencies depend on alpha; everything else in
the equilibrium is pinned to a fixed value across the whole family. We
check both: exploitability converging to ~0, and the alpha-independent
pinned action probabilities.
"""

import pytest

from poker_gnn.games.base import Action
from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.eval.exploitability import exploitability

ITERATIONS = 20000


def _uniform_strategy(game):
    strategy = {}

    def walk(state):
        if state.terminal:
            return
        if state.chance:
            for outcome, _ in game.chance_outcomes(state):
                walk(game.step(state, outcome))
            return
        key = state.infoset_key(state.player)
        legal = game.legal_actions(state)
        strategy.setdefault(key, {a: 1 / len(legal) for a in legal})
        for action in legal:
            walk(game.step(state, action))

    walk(game.root())
    return strategy


@pytest.fixture(scope="module")
def kuhn_average_strategy():
    game = KuhnPoker()
    cfr = TabularCFR()
    cfr.iterate(game, ITERATIONS)
    return game, cfr.average_strategy()


def test_cfr_converges_far_below_uniform_random(kuhn_average_strategy):
    game, strategy = kuhn_average_strategy
    cfr_exploitability = exploitability(game, strategy)
    uniform_exploitability = exploitability(game, _uniform_strategy(game))

    assert cfr_exploitability < 0.01
    assert cfr_exploitability < uniform_exploitability / 10


def test_average_strategy_is_a_valid_distribution(kuhn_average_strategy):
    _, strategy = kuhn_average_strategy
    for probs in strategy.values():
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def _prob(strategy, player, card, history, action):
    key = f"{player}|h{card}|b()|r0|a{history}"
    return strategy[key][action]


def test_pinned_dominant_actions_match_known_nash(kuhn_average_strategy):
    game, strategy = kuhn_average_strategy
    J, Q, K = 0, 1, 2
    tol = 0.03

    # P0 with Jack, checked-then-bet: always folds (Jack never wins showdown).
    assert _prob(strategy, 0, J, (Action.CHECK_CALL, Action.BET_RAISE), Action.FOLD) == pytest.approx(1.0, abs=tol)
    # P0 with Queen: never bets as the opening action.
    assert _prob(strategy, 0, Q, (), Action.CHECK_CALL) == pytest.approx(1.0, abs=tol)
    # P0 with King, checked-then-bet: always calls (King always wins showdown).
    assert _prob(strategy, 0, K, (Action.CHECK_CALL, Action.BET_RAISE), Action.CHECK_CALL) == pytest.approx(1.0, abs=tol)

    # P1 with Jack, facing an opening bet: always folds.
    assert _prob(strategy, 1, J, (Action.BET_RAISE,), Action.FOLD) == pytest.approx(1.0, abs=tol)
    # P1 with Queen, checked to: always checks back (never bluffs with a middling hand).
    assert _prob(strategy, 1, Q, (Action.CHECK_CALL,), Action.CHECK_CALL) == pytest.approx(1.0, abs=tol)
    # P1 with King, checked to: always bets for value.
    assert _prob(strategy, 1, K, (Action.CHECK_CALL,), Action.BET_RAISE) == pytest.approx(1.0, abs=tol)
    # P1 with King, facing an opening bet: always calls.
    assert _prob(strategy, 1, K, (Action.BET_RAISE,), Action.CHECK_CALL) == pytest.approx(1.0, abs=tol)


def test_pinned_mixed_actions_match_known_nash(kuhn_average_strategy):
    game, strategy = kuhn_average_strategy
    J, Q = 0, 1
    tol = 0.06

    # P1 with Jack, checked to: bluff-bets exactly 1/3 of the time.
    assert _prob(strategy, 1, J, (Action.CHECK_CALL,), Action.BET_RAISE) == pytest.approx(1 / 3, abs=tol)
    # P1 with Queen, facing an opening bet: calls exactly 1/3 of the time (bluff-catching).
    assert _prob(strategy, 1, Q, (Action.BET_RAISE,), Action.CHECK_CALL) == pytest.approx(1 / 3, abs=tol)


def test_alpha_family_relationship_holds(kuhn_average_strategy):
    # P0's opening bet frequency with a Jack (alpha) and with a King (3*alpha)
    # are only pinned relative to each other, not to a fixed number.
    _, strategy = kuhn_average_strategy
    J, K = 0, 2
    alpha = _prob(strategy, 0, J, (), Action.BET_RAISE)
    king_bet = _prob(strategy, 0, K, (), Action.BET_RAISE)

    assert 0.0 <= alpha <= 1 / 3 + 0.03
    assert king_bet == pytest.approx(3 * alpha, abs=0.05)
