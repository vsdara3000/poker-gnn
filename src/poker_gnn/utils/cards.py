"""Card ids/ranks/suits and a 5-7 card poker hand evaluator, for HULHE.

Card id convention matches `card_graph.py`'s generic `rank = card //
num_suits`: card `i` has rank `i // 4` (0=Two ... 12=Ace) and suit `i % 4`
(0=spades, 1=hearts, 2=diamonds, 3=clubs), so ranks come in contiguous
4-card blocks and 52 = 13 ranks x 4 suits.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Sequence

NUM_SUITS = 4
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "shdc"

# hand categories, low to high (used as the leading element of the score
# tuple returned by `evaluate_hand` / `_evaluate_five`)
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


def card_rank(card: int) -> int:
    return card // NUM_SUITS


def card_suit(card: int) -> int:
    return card % NUM_SUITS


def card_name(card: int) -> str:
    return RANK_CHARS[card_rank(card)] + SUIT_CHARS[card_suit(card)]


def deck_card_names() -> tuple[str, ...]:
    return tuple(card_name(c) for c in range(13 * NUM_SUITS))


def _straight_high(unique_ranks_desc: Sequence[int]) -> int | None:
    """`unique_ranks_desc` must have exactly 5 distinct ranks, sorted
    descending. Returns the straight's high rank, or None if not a straight.
    Handles the wheel (A-2-3-4-5, i.e. ranks [12,3,2,1,0]) as 5-high."""
    if len(unique_ranks_desc) != 5:
        return None
    if list(unique_ranks_desc) == [12, 3, 2, 1, 0]:
        return 3
    if unique_ranks_desc[0] - unique_ranks_desc[-1] == 4:
        return unique_ranks_desc[0]
    return None


def _evaluate_five(cards: Sequence[int]) -> tuple[int, ...]:
    """Score a single 5-card hand as a tuple, comparable lexicographically:
    (category, tiebreak-ranks...) with higher always meaning a better hand."""
    ranks = sorted((card_rank(c) for c in cards), reverse=True)
    suits = [card_suit(c) for c in cards]
    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    straight_high = _straight_high(unique_ranks)

    if straight_high is not None and is_flush:
        return (STRAIGHT_FLUSH, straight_high)

    # groups of same-rank cards, ordered by (count desc, rank desc) so the
    # more significant group (e.g. trips before the pair in a full house)
    # always compares first
    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    group_ranks = [rank for rank, _ in groups]
    group_sizes = [size for _, size in groups]

    if group_sizes == [4, 1]:
        return (FOUR_OF_A_KIND, *group_ranks)
    if group_sizes == [3, 2]:
        return (FULL_HOUSE, *group_ranks)
    if is_flush:
        return (FLUSH, *ranks)
    if straight_high is not None:
        return (STRAIGHT, straight_high)
    if group_sizes == [3, 1, 1]:
        return (THREE_OF_A_KIND, *group_ranks)
    if group_sizes == [2, 2, 1]:
        return (TWO_PAIR, *group_ranks)
    if group_sizes == [2, 1, 1, 1]:
        return (PAIR, *group_ranks)
    return (HIGH_CARD, *ranks)


def evaluate_hand(cards: Sequence[int]) -> tuple[int, ...]:
    """Best 5-card score among all 5-card subsets of `cards` (5, 6, or 7 of
    them). Higher tuple (lexicographic) always beats lower -- ties are
    exactly equal tuples."""
    if len(cards) < 5:
        raise ValueError("evaluate_hand needs at least 5 cards")
    return max(_evaluate_five(combo) for combo in combinations(cards, 5))
