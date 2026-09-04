"""Deep CFR (GNN-driven) should land far below uniform-random on Kuhn.

Phase 1 "done when" bar (docs/PLAN.md): not exact Nash, but far from
random. We don't hold it to tabular CFR's near-zero exploitability — a
dozen training iterations of a tiny neural net is noisier than an exact
regret table — just that it clearly beats uniform play by a wide margin.
"""

import pytest

from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.games.leduc import LeducPoker
from poker_gnn.solver.deep_cfr import DeepCFR
from poker_gnn.eval.exploitability import exploitability

ITERATIONS = 500
# Leduc's tree is ~450x bigger than Kuhn's (936 infosets vs 12), but
# `DeepCFR` batches one GNN forward pass per player per iteration over every
# infoset rather than one call per tree node visited, so iterations are
# still cheap enough to run plenty of them here.
LEDUC_ITERATIONS = 200


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
def kuhn_deep_cfr_strategy():
    game = KuhnPoker()
    solver = DeepCFR(seed=0)
    solver.train(game, ITERATIONS)
    return game, solver.average_strategy()


def test_deep_cfr_converges_far_below_uniform_random(kuhn_deep_cfr_strategy):
    game, strategy = kuhn_deep_cfr_strategy
    deep_cfr_exploitability = exploitability(game, strategy)
    uniform_exploitability = exploitability(game, _uniform_strategy(game))

    assert deep_cfr_exploitability < uniform_exploitability / 3


def test_average_strategy_is_a_valid_distribution(kuhn_deep_cfr_strategy):
    _, strategy = kuhn_deep_cfr_strategy
    for probs in strategy.values():
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.fixture(scope="module")
def leduc_deep_cfr_strategy():
    game = LeducPoker()
    solver = DeepCFR(seed=0)
    solver.train(game, LEDUC_ITERATIONS)
    return game, solver.average_strategy()


def test_leduc_deep_cfr_beats_uniform_random(leduc_deep_cfr_strategy):
    game, strategy = leduc_deep_cfr_strategy
    deep_cfr_exploitability = exploitability(game, strategy)
    uniform_exploitability = exploitability(game, _uniform_strategy(game))

    assert deep_cfr_exploitability < uniform_exploitability * 0.75


def test_leduc_average_strategy_is_a_valid_distribution(leduc_deep_cfr_strategy):
    _, strategy = leduc_deep_cfr_strategy
    for probs in strategy.values():
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
