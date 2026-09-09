"""External-sampling MCCFR should approach the same Kuhn Nash equilibrium
as vanilla `TabularCFR` (test_cfr.py) -- just noisier per iteration, since
it samples opponent/chance branches instead of summing them exactly. This
validates the sampling algorithm against a game with a known closed-form
answer before trusting it on HULHE, where nothing can be checked exactly.
"""

import pytest

from poker_gnn.games.base import Action
from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.solver.mccfr import ExternalSamplingCFR
from poker_gnn.eval.exploitability import exploitability

ITERATIONS = 200000


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
def kuhn_mccfr_strategy():
    game = KuhnPoker()
    solver = ExternalSamplingCFR(seed=0)
    solver.iterate(game, ITERATIONS)
    return game, solver.average_strategy()


def test_mccfr_converges_far_below_uniform_random(kuhn_mccfr_strategy):
    game, strategy = kuhn_mccfr_strategy
    mccfr_exploitability = exploitability(game, strategy)
    uniform_exploitability = exploitability(game, _uniform_strategy(game))

    assert mccfr_exploitability < 0.01
    assert mccfr_exploitability < uniform_exploitability / 10


def test_average_strategy_is_a_valid_distribution(kuhn_mccfr_strategy):
    _, strategy = kuhn_mccfr_strategy
    for probs in strategy.values():
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def _prob(strategy, player, card, history, action):
    key = f"{player}|h{card}|b()|r0|a{history}"
    return strategy[key][action]


def test_pinned_dominant_actions_match_known_nash(kuhn_mccfr_strategy):
    game, strategy = kuhn_mccfr_strategy
    J, K = 0, 2
    tol = 0.05

    assert _prob(strategy, 0, J, (Action.CHECK_CALL, Action.BET_RAISE), Action.FOLD) == pytest.approx(1.0, abs=tol)
    assert _prob(strategy, 0, K, (Action.CHECK_CALL, Action.BET_RAISE), Action.CHECK_CALL) == pytest.approx(1.0, abs=tol)
    assert _prob(strategy, 1, J, (Action.BET_RAISE,), Action.FOLD) == pytest.approx(1.0, abs=tol)
    assert _prob(strategy, 1, K, (Action.CHECK_CALL,), Action.BET_RAISE) == pytest.approx(1.0, abs=tol)


def test_pinned_mixed_actions_match_known_nash(kuhn_mccfr_strategy):
    game, strategy = kuhn_mccfr_strategy
    J, Q = 0, 1
    tol = 0.08

    # P1 with Jack, checked to: bluff-bets exactly 1/3 of the time.
    assert _prob(strategy, 1, J, (Action.CHECK_CALL,), Action.BET_RAISE) == pytest.approx(1 / 3, abs=tol)
    # P1 with Queen, facing an opening bet: calls exactly 1/3 of the time.
    assert _prob(strategy, 1, Q, (Action.BET_RAISE,), Action.CHECK_CALL) == pytest.approx(1 / 3, abs=tol)
