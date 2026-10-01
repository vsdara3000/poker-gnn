"""Tabular CFR — ground-truth solver for Kuhn / Leduc.

Vanilla (full-width) counterfactual regret minimization: every chance
outcome and every action of both players is walked on every iteration, with
chance summed exactly via `Game.chance_outcomes`. That is affordable at
Kuhn/Leduc scale and hopeless at HULHE's (~10^14 infosets), where
`mccfr.ExternalSamplingCFR` takes over. Regret matching + averaging follow
Zinkevich et al., "Regret Minimization in Games with Incomplete
Information" (2007): the *average* strategy converges to a Nash equilibrium
in two-player zero-sum games; the current strategy need not.

This is the reference the other solvers are checked against: its average
strategy reaches Kuhn's known Nash family and exploitability -> 0
(`eval.exploitability`). Both players are updated in the same pass
(simultaneous updates).
"""

from __future__ import annotations

import pickle


class _InfosetNode:
    """Cumulative regret and strategy sums for one information set. Also
    reused unchanged by `mccfr.ExternalSamplingCFR`."""

    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions):
        self.actions = tuple(actions)
        self.regret_sum = {a: 0.0 for a in self.actions}
        self.strategy_sum = {a: 0.0 for a in self.actions}

    def current_strategy(self) -> dict:
        """Regret matching: play each action in proportion to its positive
        cumulative regret; uniform if no action has positive regret."""
        positive = {a: max(r, 0.0) for a, r in self.regret_sum.items()}
        total = sum(positive.values())
        if total > 0:
            return {a: r / total for a, r in positive.items()}
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}

    def accumulate_strategy(self, reach: float, strategy: dict) -> None:
        """Add this iteration's strategy to the running sum, weighted by the
        acting player's own reach probability, so infosets the player rarely
        steers into count for less in the average."""
        for a in self.actions:
            self.strategy_sum[a] += reach * strategy[a]

    def average_strategy(self) -> dict:
        """Normalized strategy sum -- the quantity that converges to Nash."""
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
        """Run `iterations` full-width CFR passes; returns self for chaining."""
        for _ in range(iterations):
            self._cfr(game, game.root(), reach0=1.0, reach1=1.0)
        return self

    def average_strategy(self) -> dict:
        """{infoset_key: {action: prob}} for every infoset visited so far."""
        return {key: node.average_strategy() for key, node in self._nodes.items()}

    def save(self, path: str) -> None:
        """Persist the trained regret table so `average_strategy()` (or
        more training via `iterate()`) doesn't require re-solving from
        scratch. `_InfosetNode` pickles directly despite `__slots__` --
        no custom (de)serialization needed."""
        with open(path, "wb") as f:
            pickle.dump(self._nodes, f)

    @classmethod
    def load(cls, path: str) -> "TabularCFR":
        with open(path, "rb") as f:
            nodes = pickle.load(f)
        solver = cls()
        solver._nodes = nodes
        return solver

    def _cfr(self, game, state, reach0: float, reach1: float) -> tuple[float, float]:
        """One CFR pass below `state`. `reach0`/`reach1` are each player's own
        contribution to the probability of reaching `state`. Chance reach is
        left out: every chance event in Kuhn/Leduc/HULHE deals one card
        uniformly from the remaining deck, so it is the same for every state
        in an infoset and only rescales that infoset's regrets. (A game with
        non-uniform chance would need it folded into `opponent_reach`.)
        Returns both players' expected values under the current profile."""
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

        # Counterfactual regret: how much better each action would have done
        # than the current mix, weighted by the probability that the
        # *opponent* plays into this state (the acting player's own reach is
        # deliberately excluded -- that is what makes it counterfactual).
        # The average strategy, in contrast, is weighted by own reach.
        opponent_reach = reach1 if player == 0 else reach0
        own_reach = reach0 if player == 0 else reach1
        node_value = node_value0 if player == 0 else node_value1
        action_values = action_values0 if player == 0 else action_values1
        for action in legal:
            node.regret_sum[action] += opponent_reach * (action_values[action] - node_value)
        node.accumulate_strategy(own_reach, strategy)

        return (node_value0, node_value1)
