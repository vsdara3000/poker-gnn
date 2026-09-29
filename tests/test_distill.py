"""`DistilledPolicy` should reproduce a dict strategy where it has one, and
give a real (non-uniform, normalized) answer where it doesn't."""

import random

import pytest

from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.solver.distill import DistilledPolicy
from poker_gnn.solver.mccfr import ExternalSamplingCFR


def _decision_states(game):
    out = []

    def walk(state):
        if state.terminal:
            return
        if state.chance:
            for outcome, _ in game.chance_outcomes(state):
                walk(game.step(state, outcome))
            return
        out.append(state)
        for action in game.legal_actions(state):
            walk(game.step(state, action))

    walk(game.root())
    return out


def test_distilled_policy_uses_dict_where_trusted_and_network_elsewhere():
    game = KuhnPoker()
    solver = ExternalSamplingCFR(seed=0).iterate(game, 3000)
    strategy = solver.average_strategy()
    # Hide one infoset from the dict so only the network can answer there.
    states = _decision_states(game)
    hidden = states[0]
    hidden_key = hidden.infoset_key(hidden.player)
    visible = {k: v for k, v in strategy.items() if k != hidden_key}

    distilled = DistilledPolicy.fit(game, visible, random.Random(0), num_hands=500, train_steps=200)
    for state in states:
        key = state.infoset_key(state.player)
        probs = distilled.policy(game, state)
        assert sum(probs.values()) == pytest.approx(1.0)
        if key != hidden_key:
            assert probs == pytest.approx(strategy[key])
    batch = distilled.policy_batch(game, states)
    assert batch == [pytest.approx(distilled.policy(game, s)) for s in states]


def test_min_visits_excludes_rarely_visited_entries_from_lookup():
    game = KuhnPoker()
    solver = ExternalSamplingCFR(seed=0).iterate(game, 200)
    visits = solver.visit_counts()
    threshold = sorted(visits.values())[len(visits) // 2]
    distilled = DistilledPolicy.fit(
        game, solver.average_strategy(), random.Random(0), visits=visits,
        min_visits=threshold, num_hands=200, train_steps=10,
    )
    assert set(distilled.strategy) == {k for k, v in visits.items() if v >= threshold}
