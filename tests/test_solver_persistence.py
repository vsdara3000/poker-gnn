"""save()/load() round-trips for all three solvers -- without these, every
training run is throwaway: nothing can be evaluated or resumed later
without re-solving from scratch."""

import os

import pytest

from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.solver.deep_cfr import DeepCFR
from poker_gnn.solver.mccfr import ExternalSamplingCFR


def test_tabular_cfr_save_load_round_trips(tmp_path):
    game = KuhnPoker()
    solver = TabularCFR()
    solver.iterate(game, 500)
    strategy_before = solver.average_strategy()

    path = os.path.join(tmp_path, "cfr.pkl")
    solver.save(path)
    loaded = TabularCFR.load(path)

    assert loaded.average_strategy() == strategy_before

    # a loaded solver can keep training, not just be read from
    loaded.iterate(game, 100)


def test_mccfr_save_load_round_trips(tmp_path):
    game = KuhnPoker()
    solver = ExternalSamplingCFR(seed=0)
    solver.iterate(game, 500)
    strategy_before = solver.average_strategy()

    path = os.path.join(tmp_path, "mccfr.pkl")
    solver.save(path)
    loaded = ExternalSamplingCFR.load(path, seed=1)

    assert loaded.average_strategy() == strategy_before

    loaded.iterate(game, 100)


@pytest.fixture(scope="module")
def trained_deep_cfr():
    game = KuhnPoker()
    solver = DeepCFR(seed=0, external_sampling=True, parallel_traversals=4)
    solver.train(game, 40)
    return game, solver


def _all_decision_states(game):
    states = []

    def walk(state):
        if state.terminal:
            return
        if state.chance:
            for outcome, _ in game.chance_outcomes(state):
                walk(game.step(state, outcome))
            return
        states.append(state)
        for action in game.legal_actions(state):
            walk(game.step(state, action))

    walk(game.root())
    return states


def test_deep_cfr_save_load_preserves_policy_everywhere(trained_deep_cfr, tmp_path):
    # Covers both the exact dict lookup (visited infosets) and the trained
    # strategy-network fallback (unvisited ones) -- a bug that only zeroed
    # out one of the two save/load paths wouldn't show up if we only
    # checked visited states.
    game, solver = trained_deep_cfr
    path = os.path.join(tmp_path, "deep_cfr.pt")
    solver.save(path)
    loaded = DeepCFR.load(path)

    assert loaded.average_strategy() == solver.average_strategy()

    for state in _all_decision_states(game):
        before = solver.policy(game, state)
        after = loaded.policy(game, state)
        assert before.keys() == after.keys()
        for action in before:
            assert after[action] == pytest.approx(before[action], abs=1e-6)


def test_deep_cfr_save_load_preserves_config(trained_deep_cfr, tmp_path):
    game, solver = trained_deep_cfr
    path = os.path.join(tmp_path, "deep_cfr.pt")
    solver.save(path)
    loaded = DeepCFR.load(path)

    assert loaded.hidden_dim == solver.hidden_dim
    assert loaded.external_sampling == solver.external_sampling
    assert loaded.parallel_traversals == solver.parallel_traversals

    # a loaded solver can keep training, not just be read from
    loaded.train(game, 20)
