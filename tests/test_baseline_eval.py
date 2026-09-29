"""Sanity-check `baseline_eval` against Kuhn, where the right answer is
known, before trusting it as a signal on HULHE."""

import random

import pytest

from poker_gnn.eval.baseline_eval import (
    average_payoff,
    call_policy,
    evaluate_matchup,
    fold_policy,
    make_random_policy,
    make_strategy_policy,
)
from poker_gnn.games.base import Action
from poker_gnn.games.hulhe import HeadsUpLimitHoldem
from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.solver.cfr import TabularCFR


def _always_bet_policy(game, state):
    legal = game.legal_actions(state)
    return Action.BET_RAISE if Action.BET_RAISE in legal else Action.CHECK_CALL


def test_fold_policy_folds_to_every_bet_it_faces():
    # fold_policy can't fold on an unopened action (not legal in Kuhn), so
    # pit it against an opponent that always bets when possible: whichever
    # player faces that bet with fold_policy should give up the pot every
    # time, losing exactly their ante.
    game = KuhnPoker()
    rng = random.Random(0)
    p0, p1 = average_payoff(
        game, {0: _always_bet_policy, 1: fold_policy}, num_hands=2000, rng=rng
    )
    assert p1 == pytest.approx(-1.0)  # folds to the opening bet every time
    assert p0 == pytest.approx(1.0)
    assert p0 + p1 == pytest.approx(0.0)


def test_call_policy_always_checks_down_to_showdown():
    # call_policy never bets/raises, so two call-policy players always
    # check straight to showdown -- decided purely by the higher card.
    game = KuhnPoker()
    rng = random.Random(0)
    p0, p1 = average_payoff(
        game, {0: call_policy, 1: call_policy}, num_hands=2000, rng=rng
    )
    assert p0 + p1 == pytest.approx(0.0)


def test_near_nash_strategy_beats_always_fold_and_always_call():
    game = KuhnPoker()
    cfr = TabularCFR()
    cfr.iterate(game, 20000)
    strategy = cfr.average_strategy()
    rng = random.Random(0)

    ev_vs_fold = average_payoff(
        game, {0: make_strategy_policy(strategy, rng), 1: fold_policy}, 3000, rng
    )[0]
    ev_vs_call = average_payoff(
        game, {0: make_strategy_policy(strategy, rng), 1: call_policy}, 3000, rng
    )[0]
    ev_vs_random = average_payoff(
        game, {0: make_strategy_policy(strategy, rng), 1: make_random_policy(rng)}, 3000, rng
    )[0]

    assert ev_vs_fold > 0
    assert ev_vs_call > 0
    assert ev_vs_random > 0


def test_payoffs_always_sum_to_zero():
    game = KuhnPoker()
    rng = random.Random(1)
    p0, p1 = average_payoff(
        game, {0: make_random_policy(rng), 1: make_random_policy(rng)}, 2000, rng
    )
    assert p0 + p1 == pytest.approx(0.0)


@pytest.mark.parametrize("game", [KuhnPoker(), HeadsUpLimitHoldem()], ids=["kuhn", "hulhe"])
def test_duplicate_eval_cancels_card_luck_exactly_for_mirrored_check_down(game):
    # Two call-policy players check every hand down, so the result depends
    # only on who holds the better cards. Duplicate play gives hero both
    # hands of every deal, so hero's combined result must be exactly zero
    # with zero spread -- any leak here means the two replays of a deal
    # didn't actually see the same cards.
    r = evaluate_matchup(game, call_policy, call_policy, num_deals=300, rng=random.Random(0))
    assert r.duplicate == pytest.approx(0.0)
    assert r.duplicate_se == pytest.approx(0.0)
    assert r.as_p0_se > 0  # each seat on its own still sees card luck


def test_duplicate_eval_agrees_with_plain_average_payoff():
    game = KuhnPoker()
    cfr = TabularCFR()
    cfr.iterate(game, 20000)
    hero = make_strategy_policy(cfr.average_strategy(), random.Random(1))
    r = evaluate_matchup(game, hero, call_policy, num_deals=4000, rng=random.Random(0))
    plain_p0 = average_payoff(game, {0: hero, 1: call_policy}, 4000, random.Random(2))[0]
    # Same quantity, independent samples: must agree within a few SEs.
    assert abs(r.as_p0 - plain_p0) < 4 * r.as_p0_se * 2**0.5
    assert r.duplicate == pytest.approx((r.as_p0 + r.as_p1) / 2)
    assert r.duplicate > 0
