"""Local best response on HULHE: its range bookkeeping must follow Bayes'
rule, and it must beat strategies that are trivially exploitable."""

import random

from poker_gnn.eval.lbr import _LBRPlayer, local_best_response
from poker_gnn.games.base import Action
from poker_gnn.games.hulhe import HeadsUpLimitHoldem
from poker_gnn.utils.cards import card_rank


class _Fixed:
    """A strategy given as a function of the acting player's hole cards."""

    def __init__(self, fn):
        self.fn = fn

    def policy(self, game, state):
        legal = game.legal_actions(state)
        hand = (state.hole_cards[state.player], state.hole_cards2[state.player])
        probs = self.fn(hand, legal)
        return {a: probs.get(a, 0.0) for a in legal}


def _raise_with_pairs(hand, legal):
    is_pair = card_rank(hand[0]) == card_rank(hand[1])
    if is_pair and Action.BET_RAISE in legal:
        return {Action.BET_RAISE: 1.0}
    return {Action.CHECK_CALL: 1.0}


def _deal(game, seed):
    rng = random.Random(seed)
    state = game.root()
    while state.chance:
        outcomes = game.chance_outcomes(state)
        state = game.step(state, rng.choice(outcomes)[0])
    return state


def test_range_update_keeps_only_hands_consistent_with_the_action():
    game = HeadsUpLimitHoldem()
    state = _deal(game, 0)  # P0 (button) to act first preflop
    lbr = _LBRPlayer(seat=1, strategy=_Fixed(_raise_with_pairs), rng=random.Random(0),
                     range_size=1326, rollouts=1)
    lbr.observe_opponent(game, state, Action.BET_RAISE)
    live = {h for h, w in lbr.range.items() if w > 0}
    assert live, "some pairs must survive"
    assert all(card_rank(a) == card_rank(b) for a, b in live)
    mine = {state.hole_cards[1], state.hole_cards2[1]}
    assert not any(set(h) & mine for h in lbr.range)


def test_lbr_exploits_always_call_and_always_fold():
    game = HeadsUpLimitHoldem()
    always_call = _Fixed(lambda hand, legal: {Action.CHECK_CALL: 1.0})
    r = local_best_response(game, always_call, num_deals=40, rng=random.Random(0),
                            range_size=30, rollouts=4)
    assert r.duplicate > 0.5

    def fold_to_bets(hand, legal):
        return {Action.FOLD: 1.0} if Action.FOLD in legal else {Action.CHECK_CALL: 1.0}

    r = local_best_response(game, _Fixed(fold_to_bets), num_deals=20, rng=random.Random(0),
                            range_size=30, rollouts=4)
    # It should raise every time and win the blinds: at least the small blind per hand.
    assert r.duplicate >= 1.0
