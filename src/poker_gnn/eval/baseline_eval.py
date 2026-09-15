"""Approximate strategy evaluation for games too big for `exploitability()`'s
exact best response (HULHE's ~10^14 information sets can't be enumerated,
unlike Kuhn/Leduc).

Not a Nash-distance metric: it plays a strategy against simple, fixed
baseline opponents (always fold, always check/call, uniform random) over
many simulated hands and reports average chip EV per hand. Beating these
baselines by a wide margin doesn't prove a strategy is close to Nash --
only that it has learned something better than nonsense. Fully game-
agnostic (no enumeration), so it works the same way for Kuhn/Leduc too.
"""

from __future__ import annotations

import random
from typing import Callable

from poker_gnn.games.base import Action

Policy = Callable[[object, object], int]  # (game, state) -> action


def fold_policy(game, state) -> int:
    legal = game.legal_actions(state)
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def call_policy(game, state) -> int:
    return Action.CHECK_CALL


def make_random_policy(rng: random.Random) -> Policy:
    def policy(game, state) -> int:
        return rng.choice(game.legal_actions(state))

    return policy


def make_strategy_policy(strategy: dict, rng: random.Random) -> Policy:
    """Samples from a trained `average_strategy()` dict; falls back to
    uniform random at any infoset the strategy never visited during
    training (inevitable in HULHE, where the trained dict only covers a
    tiny sampled fraction of all information sets)."""

    def policy(game, state) -> int:
        legal = game.legal_actions(state)
        probs = strategy.get(state.infoset_key(state.player))
        if probs is None:
            return rng.choice(legal)
        weights = [probs.get(a, 0.0) for a in legal]
        if sum(weights) <= 0:
            return rng.choice(legal)
        return rng.choices(legal, weights=weights, k=1)[0]

    return policy


def make_solver_policy(solver, rng: random.Random) -> Policy:
    """Samples from `solver.policy(game, state)` instead of a plain dict
    (`make_strategy_policy`). Use this for a solver exposing a hybrid
    exact-dict/network-fallback lookup (`DeepCFR.policy` in `external_
    sampling` mode) so evaluation reflects what the network generalized
    to, not just the infosets it happened to visit during training."""

    def policy(game, state) -> int:
        legal = game.legal_actions(state)
        probs = solver.policy(game, state)
        weights = [probs.get(a, 0.0) for a in legal]
        if sum(weights) <= 0:
            return rng.choice(legal)
        return rng.choices(legal, weights=weights, k=1)[0]

    return policy


def average_payoff(
    game, policies: dict[int, Policy], num_hands: int, rng: random.Random
) -> tuple[float, float]:
    """Average (P0, P1) payoff per hand, playing `policies[0]` against
    `policies[1]` over `num_hands` simulated hands with random deals."""
    total = [0.0, 0.0]
    for _ in range(num_hands):
        state = game.root()
        while not state.terminal:
            if state.chance:
                outcomes = game.chance_outcomes(state)
                items = [item for item, _ in outcomes]
                weights = [w for _, w in outcomes]
                action = rng.choices(items, weights=weights, k=1)[0]
            else:
                action = policies[state.player](game, state)
            state = game.step(state, action)
        p0, p1 = game.returns(state)
        total[0] += p0
        total[1] += p1
    return (total[0] / num_hands, total[1] / num_hands)
