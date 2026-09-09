"""HULHE game tests.

The full tree is far too large to enumerate (unlike Kuhn/Leduc's tests),
so these check known scripted lines plus statistical properties (zero-sum,
no negative stacks) over many random playouts instead of exhaustive walks.
"""

import random

import pytest

from poker_gnn.games.base import Action
from poker_gnn.games.hulhe import HeadsUpLimitHoldem, STARTING_STACK
from poker_gnn.utils.cards import card_name


def _deal(game, four_cards):
    """Step through the four hole-card chance nodes: P0 card1, P0 card2,
    P1 card1, P1 card2."""
    state = game.root()
    for card in four_cards:
        state = game.step(state, card)
    return state


def _deal_street(game, state, rng=None):
    """Deal whatever community cards the current chance node needs
    (deterministically to the lowest remaining card id unless `rng` given)."""
    while state.chance:
        outcomes = game.chance_outcomes(state)
        if rng is None:
            card = outcomes[0][0]
        else:
            cards, probs = zip(*outcomes)
            card = rng.choices(cards, weights=probs, k=1)[0]
        state = game.step(state, card)
    return state


def _play_randomly(game, rng):
    state = game.root()
    while not state.terminal:
        if state.chance:
            state = _deal_street(game, state, rng)
        else:
            action = rng.choice(game.legal_actions(state))
            state = game.step(state, action)
    return state


def test_chance_deals_four_distinct_hole_cards_uniformly():
    game = HeadsUpLimitHoldem()
    state = game.root()
    outcomes = game.chance_outcomes(state)
    assert len(outcomes) == 52
    assert all(p == pytest.approx(1 / 52) for _, p in outcomes)

    state = game.step(state, 0)
    outcomes = game.chance_outcomes(state)
    assert len(outcomes) == 51
    assert 0 not in [c for c, _ in outcomes]


def test_root_after_deal_is_preflop_with_blinds_posted():
    game = HeadsUpLimitHoldem()
    state = _deal(game, [0, 1, 4, 5])
    assert not state.chance
    assert not state.terminal
    assert state.round == 0
    assert state.player == 0  # button/small blind acts first preflop
    assert state.pot == 3  # small blind 1 + big blind 2
    assert state.stacks == (STARTING_STACK - 1, STARTING_STACK - 2)
    assert set(game.legal_actions(state)) == {Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE}


def test_big_blind_gets_the_option_and_cannot_fold_facing_no_bet():
    game = HeadsUpLimitHoldem()
    state = _deal(game, [0, 1, 4, 5])
    state = game.step(state, Action.CHECK_CALL)  # P0 completes to match the BB
    assert state.player == 1
    assert Action.FOLD not in game.legal_actions(state)

    state_after_raise = game.step(state, Action.BET_RAISE)
    assert state_after_raise.player == 0
    assert Action.FOLD in game.legal_actions(state_after_raise)  # now P0 faces a real bet

    state_after_check = game.step(state, Action.CHECK_CALL)
    assert state_after_check.chance  # BB checking back closes preflop
    assert state_after_check.round == 1


def test_big_blind_acts_first_postflop():
    game = HeadsUpLimitHoldem()
    state = _deal(game, [0, 1, 4, 5])
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    state = _deal_street(game, state)
    assert not state.chance
    assert state.round == 1
    assert len(state.board) == 3
    assert state.player == 1


def test_board_grows_zero_three_four_five():
    game = HeadsUpLimitHoldem()
    state = _deal(game, [0, 1, 4, 5])
    assert state.board == ()
    for expected_len in (3, 4, 5):
        state = game.step(state, Action.CHECK_CALL)
        state = game.step(state, Action.CHECK_CALL)
        state = _deal_street(game, state)
        assert len(state.board) == expected_len


def test_raise_cap_is_four_per_round():
    game = HeadsUpLimitHoldem()
    state = _deal(game, [0, 1, 4, 5])
    for _ in range(4):
        assert Action.BET_RAISE in game.legal_actions(state)
        state = game.step(state, Action.BET_RAISE)
    assert Action.BET_RAISE not in game.legal_actions(state)
    assert set(game.legal_actions(state)) == {Action.FOLD, Action.CHECK_CALL}
    # level = big blind(2) + 4 raises of the preflop bet size(2) = 10
    state = game.step(state, Action.CHECK_CALL)
    assert state.stacks == (STARTING_STACK - 10, STARTING_STACK - 10)


def test_fold_awards_the_pot_without_showdown():
    game = HeadsUpLimitHoldem()
    state = _deal(game, [0, 1, 4, 5])
    state = game.step(state, Action.CHECK_CALL)  # P0 completes (pays 1 more)
    state = game.step(state, Action.CHECK_CALL)  # P1 checks, closes preflop
    state = _deal_street(game, state)
    state = game.step(state, Action.BET_RAISE)  # P1 bets the flop
    state = game.step(state, Action.FOLD)  # P0 folds
    assert state.terminal
    returns = game.returns(state)
    assert sum(returns) == pytest.approx(0.0)
    assert returns[1] > 0 and returns[0] < 0
    assert returns[1] == pytest.approx(2.0)  # P0's total preflop contribution


def test_showdown_quads_beat_board_only_full_house():
    game = HeadsUpLimitHoldem()
    # P0 = As Kh, P1 = 2s 2h; board 2d 2c 3s 3h 3d gives P1 quad deuces,
    # while P0 only has the board's own full house (333/22).
    ace_spades, king_hearts = 12 * 4 + 0, 11 * 4 + 1
    two_spades, two_hearts = 0 * 4 + 0, 0 * 4 + 1
    state = _deal(game, [ace_spades, king_hearts, two_spades, two_hearts])
    board = [0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 0, 1 * 4 + 1, 1 * 4 + 2]  # 2d 2c 3s 3h 3d
    for _ in range(4):
        state = game.step(state, Action.CHECK_CALL)
        state = game.step(state, Action.CHECK_CALL)
        while state.chance:
            state = game.step(state, board[len(state.board)])
    assert state.terminal
    assert [card_name(c) for c in state.board] == ["2d", "2c", "3s", "3h", "3d"]
    returns = game.returns(state)
    assert returns[1] > 0 and returns[0] < 0  # P1's quads win


def test_split_pot_on_identical_board_playing_hands():
    game = HeadsUpLimitHoldem()
    # Both hole cards are lower than every board card and irrelevant to the
    # best hand either player can make: the board's own straight plays for
    # both, so it's an exact split.
    state = _deal(game, [0, 1, 4, 5])  # 2s 2h 3s 3h -- pockets, unused by the board straight
    board = [2 * 4 + 0, 3 * 4 + 0, 4 * 4 + 0, 5 * 4 + 0, 6 * 4 + 0]  # 4s 5s 6s 7s 8s
    for _ in range(4):
        state = game.step(state, Action.CHECK_CALL)
        state = game.step(state, Action.CHECK_CALL)
        while state.chance:
            state = game.step(state, board[len(state.board)])
    assert state.terminal
    assert game.returns(state) == (0.0, 0.0)


def test_different_preflop_lines_reaching_flop_are_distinct_infosets():
    game = HeadsUpLimitHoldem()

    checked = _deal(game, [0, 1, 4, 5])
    checked = game.step(checked, Action.CHECK_CALL)
    checked = game.step(checked, Action.CHECK_CALL)
    checked = _deal_street(game, checked)

    raised = _deal(game, [0, 1, 4, 5])
    raised = game.step(raised, Action.BET_RAISE)
    raised = game.step(raised, Action.CHECK_CALL)
    raised = _deal_street(game, raised)

    assert checked.player == raised.player == 1
    assert checked.board == raised.board
    assert checked.pot != raised.pot
    assert checked.infoset_key(1) != raised.infoset_key(1)


def test_random_playouts_are_zero_sum_and_never_go_negative():
    game = HeadsUpLimitHoldem()
    rng = random.Random(0)
    for _ in range(2000):
        state = _play_randomly(game, rng)
        returns = game.returns(state)
        assert sum(returns) == pytest.approx(0.0)
        assert state.stacks[0] >= 0
        assert state.stacks[1] >= 0
