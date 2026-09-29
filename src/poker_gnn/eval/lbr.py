"""Local Best Response (Lisý & Bowling, 2017) for HULHE: an estimated lower
bound on a strategy's exploitability, for a game too big for
`exploitability()`'s exact best response.

Unlike `baseline_eval`'s fixed opponents, LBR adapts to the strategy it's
playing against. It keeps a *range*, a weighted set of hole-card pairs the
opponent might hold, and updates it by Bayes' rule after every opponent
action: a hand's weight is multiplied by the probability the strategy
gives that action when holding that hand. At each of its own decisions LBR
picks the action with the best one-step greedy value, assuming the hand is
then checked/called down to showdown:

    fold:  0
    call:  wp * pot - (1 - wp) * to_call
    raise: fp * pot + (1 - fp) * (wp' * (pot + bet) - (1 - wp') * (to_call + bet))

`wp` is LBR's showdown equity against the range, `fp` is the probability
the range folds to a raise, and `wp'` is the equity against the part of the
range that doesn't fold. All values are relative to folding now. Whatever
LBR wins on average is a valid lower bound on exploitability, up to
sampling noise. It's *loose* (LBR only looks one action ahead), so a small
number doesn't prove a strategy is close to Nash, but a large one proves
it isn't.

Two approximations keep this affordable in pure Python: the range is a
random sample of `range_size` hole-card pairs, drawn at the start of each
hand rather than all 1,326, and equity is estimated from `rollouts` random
board completions per range hand.
"""

from __future__ import annotations

import random
from dataclasses import replace
from itertools import combinations

from poker_gnn.eval.baseline_eval import MatchupResult, _mean_and_stderr
from poker_gnn.games.base import Action
from poker_gnn.games.hulhe import BET_SIZES, STARTING_STACK
from poker_gnn.utils.cards import evaluate_hand


def _probs_batch(strategy, game, states) -> list[dict[int, float]]:
    if hasattr(strategy, "policy_batch"):
        return strategy.policy_batch(game, states)
    return [strategy.policy(game, s) for s in states]


def _with_hole_cards(state, player: int, hand: tuple[int, int]):
    h1 = list(state.hole_cards)
    h2 = list(state.hole_cards2)
    h1[player], h2[player] = hand
    return replace(state, hole_cards=tuple(h1), hole_cards2=tuple(h2))


class _LBRPlayer:
    """LBR's state for one hand: its seat, and the opponent range."""

    def __init__(self, seat: int, strategy, rng: random.Random, range_size: int, rollouts: int):
        self.seat = seat
        self.strategy = strategy
        self.rng = rng
        self.range_size = range_size
        self.rollouts = rollouts
        self.range: dict[tuple[int, int], float] | None = None

    def _init_range(self, state) -> None:
        mine = {state.hole_cards[self.seat], state.hole_cards2[self.seat]}
        combos = list(combinations([c for c in range(52) if c not in mine], 2))
        sample = self.rng.sample(combos, min(self.range_size, len(combos)))
        self.range = {hand: 1.0 for hand in sample}

    def _live_range(self, state) -> dict[tuple[int, int], float]:
        board = set(state.board)
        return {h: w for h, w in self.range.items() if w > 0 and not (board & set(h))}

    def observe_opponent(self, game, state, action: int) -> None:
        """Bayes update after the opponent took `action` at `state`."""
        if self.range is None:
            self._init_range(state)
        live = self._live_range(state)
        hands = list(live)
        opp = 1 - self.seat
        probs = _probs_batch(self.strategy, game, [_with_hole_cards(state, opp, h) for h in hands])
        self.range = {h: live[h] * p.get(action, 0.0) for h, p in zip(hands, probs)}

    def _equity(self, state, hand: tuple[int, int]) -> float:
        mine = [state.hole_cards[self.seat], state.hole_cards2[self.seat]]
        dead = set(mine) | set(hand) | set(state.board)
        deck = [c for c in range(52) if c not in dead]
        missing = 5 - len(state.board)
        n = 1 if missing == 0 else self.rollouts
        total = 0.0
        for _ in range(n):
            board = list(state.board) + self.rng.sample(deck, missing)
            ours = evaluate_hand(mine + board)
            theirs = evaluate_hand(list(hand) + board)
            total += 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0
        return total / n

    def act(self, game, state) -> int:
        if self.range is None:
            self._init_range(state)
        legal = game.legal_actions(state)
        live = self._live_range(state)
        if not live:
            # Every sampled hand has been ruled out. The strategy did
            # something it gives ~0 probability to with every hand in our
            # sample, so there's no range left to exploit; just call.
            return Action.CHECK_CALL

        hands = list(live)
        weights = [live[h] for h in hands]
        total_w = sum(weights)
        equities = [self._equity(state, h) for h in hands]
        wp = sum(w * e for w, e in zip(weights, equities)) / total_w

        contrib = [STARTING_STACK - s for s in state.stacks]
        to_call = abs(contrib[0] - contrib[1])
        pot = state.pot
        bet = BET_SIZES[state.round]

        values = {Action.CHECK_CALL: wp * pot - (1 - wp) * to_call}
        if Action.FOLD in legal:
            values[Action.FOLD] = 0.0
        if Action.BET_RAISE in legal:
            after = game.step(state, Action.BET_RAISE)
            opp = 1 - self.seat
            probs = _probs_batch(self.strategy, game, [_with_hole_cards(after, opp, h) for h in hands])
            fold_w = [w * p.get(Action.FOLD, 0.0) for w, p in zip(weights, probs)]
            fp = sum(fold_w) / total_w
            call_w = [w - f for w, f in zip(weights, fold_w)]
            if sum(call_w) > 0:
                wp_call = sum(c * e for c, e in zip(call_w, equities)) / sum(call_w)
            else:
                wp_call = wp
            values[Action.BET_RAISE] = fp * pot + (1 - fp) * (
                wp_call * (pot + bet) - (1 - wp_call) * (to_call + bet)
            )
        return max(values, key=values.get)


def _play_lbr_hand(game, strategy, lbr_seat: int, chance_rng, lbr_rng, strategy_rng, range_size, rollouts):
    lbr = _LBRPlayer(lbr_seat, strategy, lbr_rng, range_size, rollouts)
    state = game.root()
    while not state.terminal:
        if state.chance:
            outcomes = game.chance_outcomes(state)
            action = chance_rng.choices(
                [o for o, _ in outcomes], weights=[w for _, w in outcomes], k=1
            )[0]
        elif state.player == lbr_seat:
            action = lbr.act(game, state)
        else:
            legal = game.legal_actions(state)
            probs = strategy.policy(game, state)
            action = strategy_rng.choices(legal, weights=[probs.get(a, 0.0) for a in legal], k=1)[0]
            lbr.observe_opponent(game, state, action)
        state = game.step(state, action)
    return game.returns(state)[lbr_seat]


def local_best_response(
    game, strategy, num_deals: int, rng: random.Random, range_size: int = 100, rollouts: int = 8
) -> MatchupResult:
    """LBR's chip EV per hand against `strategy` (anything with
    `policy(game, state) -> {action: prob}`, and optionally `policy_batch`),
    duplicate-dealt like `baseline_eval.evaluate_matchup`: every deal is
    played with LBR in each seat. `as_p0`/`as_p1` are *LBR's* seats, and
    `duplicate` is the exploitability lower-bound estimate."""
    as_p0, as_p1, duplicate = [], [], []
    for _ in range(num_deals):
        deal_seed = rng.getrandbits(64)
        v0 = _play_lbr_hand(game, strategy, 0, random.Random(deal_seed), rng, rng, range_size, rollouts)
        v1 = _play_lbr_hand(game, strategy, 1, random.Random(deal_seed), rng, rng, range_size, rollouts)
        as_p0.append(v0)
        as_p1.append(v1)
        duplicate.append((v0 + v1) / 2)
    return MatchupResult(
        num_deals, *_mean_and_stderr(as_p0), *_mean_and_stderr(as_p1), *_mean_and_stderr(duplicate)
    )
