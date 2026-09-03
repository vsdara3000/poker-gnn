"""Tabular CFR — ground-truth solver for Kuhn / Leduc.

Vanilla (full-width) counterfactual regret minimization: chance nodes are
summed over exactly via `Game.chance_outcomes`, so no sampling is needed at
the scale of Kuhn/Leduc. Regret matching + averaging follow Zinkevich et al.
"Regret Minimization in Games with Incomplete Information" (2007).
"""

from __future__ import annotations


class _InfosetNode:
    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions):
        self.actions = tuple(actions)
        self.regret_sum = {a: 0.0 for a in self.actions}
        self.strategy_sum = {a: 0.0 for a in self.actions}

    def current_strategy(self) -> dict:
        positive = {a: max(r, 0.0) for a, r in self.regret_sum.items()}
        total = sum(positive.values())
        if total > 0:
            return {a: r / total for a, r in positive.items()}
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}

    def accumulate_strategy(self, reach: float, strategy: dict) -> None:
        for a in self.actions:
            self.strategy_sum[a] += reach * strategy[a]

    def average_strategy(self) -> dict:
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: s / total for a, s in self.strategy_sum.items()}
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}


class TabularCFR:
    """Vanilla CFR over any two-player zero-sum `Game`."""

    def __init__(self):
        self._nodes: dict[str, _InfosetNode] = {}

    def iterate(self, game, iterations: int):
        for _ in range(iterations):
            self._cfr(game, game.root(), reach0=1.0, reach1=1.0)
        return self

    def average_strategy(self) -> dict:
        return {key: node.average_strategy() for key, node in self._nodes.items()}

    def _cfr(self, game, state, reach0: float, reach1: float) -> tuple[float, float]:
        if state.terminal:
            return game.returns(state)

        if state.chance:
            value0 = 0.0
            value1 = 0.0
            for outcome, prob in game.chance_outcomes(state):
                u0, u1 = self._cfr(game, game.step(state, outcome), reach0, reach1)
                value0 += prob * u0
                value1 += prob * u1
            return (value0, value1)

        player = state.player
        key = state.infoset_key(player)
        legal = game.legal_actions(state)
        node = self._nodes.setdefault(key, _InfosetNode(legal))
        strategy = node.current_strategy()

        action_values0 = {}
        action_values1 = {}
        node_value0 = 0.0
        node_value1 = 0.0
        for action in legal:
            child = game.step(state, action)
            if player == 0:
                u0, u1 = self._cfr(game, child, reach0 * strategy[action], reach1)
            else:
                u0, u1 = self._cfr(game, child, reach0, reach1 * strategy[action])
            action_values0[action] = u0
            action_values1[action] = u1
            node_value0 += strategy[action] * u0
            node_value1 += strategy[action] * u1

        opponent_reach = reach1 if player == 0 else reach0
        own_reach = reach0 if player == 0 else reach1
        node_value = node_value0 if player == 0 else node_value1
        action_values = action_values0 if player == 0 else action_values1
        for action in legal:
            node.regret_sum[action] += opponent_reach * (action_values[action] - node_value)
        node.accumulate_strategy(own_reach, strategy)

        return (node_value0, node_value1)
