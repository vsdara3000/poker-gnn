"""Hand evaluator tests: category detection, tie-breaks, and the wheel."""

import pytest

from poker_gnn.utils.cards import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    card_name,
    evaluate_hand,
)


def c(name: str) -> int:
    """'Ah' -> card id. Rank char from RANK_CHARS, suit char from SUIT_CHARS."""
    rank_chars = "23456789TJQKA"
    suit_chars = "shdc"
    rank = rank_chars.index(name[0])
    suit = suit_chars.index(name[1])
    return rank * 4 + suit


def cards(names: str) -> list[int]:
    return [c(n) for n in names.split()]


def test_card_name_roundtrips():
    assert card_name(c("Ah")) == "Ah"
    assert card_name(c("2s")) == "2s"
    assert card_name(c("Tc")) == "Tc"


@pytest.mark.parametrize(
    "hand,expected_category",
    [
        ("As Ks Qs Js Ts", STRAIGHT_FLUSH),
        ("Ah Ad Ac As Kh", FOUR_OF_A_KIND),
        ("Ah Ad Ac Kh Ks", FULL_HOUSE),
        ("2s 5s 7s 9s Ks", FLUSH),
        ("5h 6d 7c 8s 9h", STRAIGHT),
        ("Ah 2d 3c 4s 5h", STRAIGHT),  # the wheel
        ("Ah Ad Ac 5s 9h", THREE_OF_A_KIND),
        ("Ah Ad Kc Ks 9h", TWO_PAIR),
        ("Ah Ad Kc Qs 9h", PAIR),
        ("Ah Kd Qc Js 9h", HIGH_CARD),
    ],
)
def test_hand_category(hand, expected_category):
    score = evaluate_hand(cards(hand))
    assert score[0] == expected_category


def test_wheel_is_the_lowest_straight():
    wheel = evaluate_hand(cards("Ah 2d 3c 4s 5h"))
    six_high = evaluate_hand(cards("2h 3d 4c 5s 6h"))
    assert wheel[0] == STRAIGHT
    assert wheel < six_high


def test_ace_high_straight_beats_king_high_straight():
    ace_high = evaluate_hand(cards("Ts Js Qd Kc Ah"))
    king_high = evaluate_hand(cards("9s Ts Jd Qc Kh"))
    assert ace_high[0] == STRAIGHT
    assert ace_high > king_high


def test_full_house_ranked_by_trips_not_pair():
    # 222 KK beats 999 AA: the three-of-a-kind decides it, not the pair.
    low_trips_high_pair = evaluate_hand(cards("2h 2d 2c Kh Ks"))
    high_trips_low_pair = evaluate_hand(cards("9h 9d 9c Ah As"))
    assert high_trips_low_pair[0] == FULL_HOUSE == low_trips_high_pair[0]
    assert high_trips_low_pair > low_trips_high_pair


def test_two_pair_ranked_by_higher_pair_then_lower_pair_then_kicker():
    aces_and_twos = evaluate_hand(cards("Ah Ad 2c 2s Kh"))
    aces_and_threes = evaluate_hand(cards("Ah Ad 3c 3s 4h"))
    kings_and_queens = evaluate_hand(cards("Kh Kd Qc Qs Ah"))
    assert aces_and_threes > aces_and_twos  # same top pair, better second pair
    assert aces_and_twos > kings_and_queens  # better top pair wins outright


def test_flush_ranked_by_all_five_cards_descending():
    ace_high_flush = evaluate_hand(cards("As 2s 3s 4s 6s"))
    king_high_flush = evaluate_hand(cards("Ks Qs Js 9s 8s"))
    assert ace_high_flush[0] == FLUSH == king_high_flush[0]
    assert ace_high_flush > king_high_flush


def test_kickers_break_ties_on_pair_and_high_card():
    pair_better_kicker = evaluate_hand(cards("Ah Ad Kc Qs 2h"))
    pair_worse_kicker = evaluate_hand(cards("Ah Ad Kc Js 2h"))
    assert pair_better_kicker[0] == PAIR == pair_worse_kicker[0]
    assert pair_better_kicker > pair_worse_kicker


def test_evaluate_hand_picks_the_best_five_of_seven():
    # 7 cards: a made flush should be found even with 2 extra unrelated cards.
    seven = cards("As 2s 3s 4s 6s 9h Kd")
    assert evaluate_hand(seven)[0] == FLUSH


def test_evaluate_hand_requires_at_least_five_cards():
    with pytest.raises(ValueError):
        evaluate_hand(cards("Ah Kd"))


def test_straight_flush_beats_four_of_a_kind():
    assert evaluate_hand(cards("9s Ts Js Qs Ks")) > evaluate_hand(cards("2h 2d 2c 2s Ah"))


def test_identical_hands_tie_exactly():
    a = evaluate_hand(cards("Ah Kd Qc Js 9h"))
    b = evaluate_hand(cards("Ad Kh Qs Jc 9d"))  # same ranks, different suits
    assert a == b
