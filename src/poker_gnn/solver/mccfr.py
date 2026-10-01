"""External-sampling Monte Carlo CFR (Lanctot et al. 2009), for games too
big to traverse exhaustively -- HULHE's ~10^14 information sets rule out
`TabularCFR`'s full-width recursion.

Each iteration picks one player as the *traverser* (alternating): only their
own decisions branch over every legal action, same as vanilla CFR. Every
opponent decision and every chance event is instead sampled once (one action
drawn from the current strategy / one card drawn from the deck) and only
that one branch is recursed into. So the number of nodes visited per
iteration scales with the traverser's own decision depth and branching, not
with the size of the whole tree.

This also removes the need for vanilla CFR's explicit opponent-reach weight
in the regret update: opponent and chance branches are reached in
proportion to their probability, and that sampling probability cancels the
importance weight, so the plain sampled regret is an unbiased estimate of
the true (reach-weighted) counterfactual regret (Lanctot et al., "Monte
Carlo Sampling for Regret Minimization in Extensive Games", NIPS 2009).
Reuses `TabularCFR`'s `_InfosetNode` bookkeeping unchanged, just always
calling `accumulate_strategy` with reach=1.0.

Used as the cheap tabular baseline on HULHE. Its dict covers only part of
what a fresh deal reaches, so for evaluation it is paired with a distilled
strategy network (`solver.distill`); on Kuhn it is checked against the
closed-form Nash mix (`tests/test_mccfr.py`).
"""

from __future__ import annotations

import pickle
import random

from poker_gnn.solver.cfr import _InfosetNode


class ExternalSamplingCFR:
    """External-sampling MCCFR over any two-player zero-sum `Game`."""

    def __init__(self, seed: int | None = None):
        self._nodes: dict[str, _InfosetNode] = {}
        self._rng = random.Random(seed)

    def iterate(self, game, iterations: int):
        """Run `iterations` sampled traversals, alternating the traverser
        between players; returns self for chaining."""
        for i in range(iterations):
            traverser = i % 2
            self._traverse(game, game.root(), traverser)
        return self

    def average_strategy(self) -> dict:
        """{infoset_key: {action: prob}} for every infoset visited so far."""
        return {key: node.average_strategy() for key, node in self._nodes.items()}

    def visit_counts(self) -> dict[str, float]:
        """Per-infoset number of times its strategy was accumulated (each
        accumulation adds a distribution summing to 1 at reach 1.0) -- how
        much to trust `average_strategy()` there (a once-visited infoset's
        "average" is just one noisy current strategy)."""
        return {key: sum(node.strategy_sum.values()) for key, node in self._nodes.items()}

    def save(self, path: str) -> None:
        """Persist the trained regret table. Doesn't preserve `_rng`'s exact
        state (`load` starts a fresh one) -- future sampling won't be
        bit-identical to an uninterrupted run, but training/averaging is
        correct regardless, same as any other CFR checkpoint restart."""
        with open(path, "wb") as f:
            pickle.dump(self._nodes, f)

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "ExternalSamplingCFR":
        with open(path, "rb") as f:
            nodes = pickle.load(f)
        solver = cls(seed=seed)
        solver._nodes = nodes
        return solver

    def _sample(self, outcomes):
        """Draw one item from a sequence of (item, probability) pairs."""
        items = [item for item, _ in outcomes]
        weights = [weight for _, weight in outcomes]
        return self._rng.choices(items, weights=weights, k=1)[0]

    def _traverse(self, game, state, traverser: int) -> float:
        """Sampled value of `state` for `traverser` under the current
        strategy profile, updating regrets and strategy sums on the way."""
        if state.terminal:
            return game.returns(state)[traverser]

        if state.chance:
            outcome = self._sample(game.chance_outcomes(state))
            return self._traverse(game, game.step(state, outcome), traverser)

        player = state.player
        key = state.infoset_key(player)
        legal = game.legal_actions(state)
        node = self._nodes.setdefault(key, _InfosetNode(legal))
        strategy = node.current_strategy()

        if player == traverser:
            # Traverser: branch over every action and update regrets, with no
            # reach weighting (see module docstring).
            action_values = {}
            node_value = 0.0
            for action in legal:
                v = self._traverse(game, game.step(state, action), traverser)
                action_values[action] = v
                node_value += strategy[action] * v
            for action in legal:
                node.regret_sum[action] += action_values[action] - node_value
            return node_value

        # Opponent: this is where the opponent's average strategy is
        # accumulated. Its own reach is accounted for by how often sampling
        # leads here, so the update is unweighted. Then follow one sampled
        # action.
        node.accumulate_strategy(1.0, strategy)
        action = self._sample([(a, strategy[a]) for a in legal])
        return self._traverse(game, game.step(state, action), traverser)
