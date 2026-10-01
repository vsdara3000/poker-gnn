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

import math
import random
from dataclasses import dataclass
from typing import Callable

from poker_gnn.games.base import Action

Policy = Callable[[object, object], int]  # (game, state) -> action


def fold_policy(game, state) -> int:
    """Check/fold: folds to any bet, checks when checking is free."""
    legal = game.legal_actions(state)
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def call_policy(game, state) -> int:
    """Calling station: always checks or calls, never folds or raises."""
    return Action.CHECK_CALL


def make_random_policy(rng: random.Random) -> Policy:
    """Uniform random over legal actions."""

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
    sampling` mode, or `distill.DistilledPolicy`) so evaluation reflects
    what the network generalized to, not just the infosets visited during
    training."""

    def policy(game, state) -> int:
        legal = game.legal_actions(state)
        probs = solver.policy(game, state)
        weights = [probs.get(a, 0.0) for a in legal]
        if sum(weights) <= 0:
            return rng.choice(legal)
        return rng.choices(legal, weights=weights, k=1)[0]

    return policy


def _play_hand(game, policies: dict[int, Policy], chance_rng: random.Random) -> tuple[float, float]:
    """Play one hand to the end, drawing chance outcomes from `chance_rng`
    (policies keep their own rngs, so reseeding `chance_rng` replays the
    exact same deal regardless of what the players do)."""
    state = game.root()
    while not state.terminal:
        if state.chance:
            outcomes = game.chance_outcomes(state)
            items = [item for item, _ in outcomes]
            weights = [w for _, w in outcomes]
            action = chance_rng.choices(items, weights=weights, k=1)[0]
        else:
            action = policies[state.player](game, state)
        state = game.step(state, action)
    return game.returns(state)


def average_payoff(
    game, policies: dict[int, Policy], num_hands: int, rng: random.Random
) -> tuple[float, float]:
    """Average (P0, P1) payoff per hand, playing `policies[0]` against
    `policies[1]` over `num_hands` simulated hands with random deals."""
    total = [0.0, 0.0]
    for _ in range(num_hands):
        p0, p1 = _play_hand(game, policies, rng)
        total[0] += p0
        total[1] += p1
    return (total[0] / num_hands, total[1] / num_hands)


def _mean_and_stderr(samples: list[float]) -> tuple[float, float]:
    """Sample mean and its standard error (NaN with fewer than 2 samples)."""
    n = len(samples)
    mean = sum(samples) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    return mean, math.sqrt(var / n)


@dataclass(frozen=True)
class MatchupResult:
    """Hero's chip EV per hand against one opponent, with standard errors.

    `as_p0`/`as_p1` are hero's EV in each seat; `duplicate` is their
    per-deal average (hero plays the same deal once from each seat), which
    cancels most of the card luck and is the number to compare runs by.
    """

    num_deals: int
    as_p0: float
    as_p0_se: float
    as_p1: float
    as_p1_se: float
    duplicate: float
    duplicate_se: float


def evaluate_matchup(
    game, hero: Policy, opponent: Policy, num_deals: int, rng: random.Random
) -> MatchupResult:
    """Duplicate evaluation: every deal is played twice with identical
    chance outcomes, once with hero as P0 and once as P1, so a lucky or
    unlucky run of cards affects both seats equally instead of showing up
    as noise. A chance rng is reseeded per deal from `rng`; the policies
    keep drawing from their own rngs."""
    as_p0, as_p1, duplicate = [], [], []
    for _ in range(num_deals):
        deal_seed = rng.getrandbits(64)
        p0, _ = _play_hand(game, {0: hero, 1: opponent}, random.Random(deal_seed))
        _, p1 = _play_hand(game, {0: opponent, 1: hero}, random.Random(deal_seed))
        as_p0.append(p0)
        as_p1.append(p1)
        duplicate.append((p0 + p1) / 2)
    return MatchupResult(
        num_deals,
        *_mean_and_stderr(as_p0),
        *_mean_and_stderr(as_p1),
        *_mean_and_stderr(duplicate),
    )
