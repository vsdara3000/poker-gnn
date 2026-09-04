"""Leduc game tests: rules, terminal payoffs, and full-tree sanity checks."""

import pytest

from poker_gnn.games.base import Action
from poker_gnn.games.leduc import LeducPoker


def _deal(game, card0, card1):
    """Step through the two hole-card chance nodes to reach a betting state."""
    state = game.root()
    state = game.step(state, card0)
    state = game.step(state, card1)
    return state


def _collect_tree(game, state, prob=1.0):
    """Recursively walk the whole tree; return (infoset_keys, terminal_probs_and_returns)."""
    infosets = set()
    terminals = []

    def walk(state, prob):
        if state.terminal:
            terminals.append((prob, game.returns(state)))
            return
        if state.chance:
            for outcome, p in game.chance_outcomes(state):
                walk(game.step(state, outcome), prob * p)
            return
        infosets.add(state.infoset_key(state.player))
        for action in game.legal_actions(state):
            walk(game.step(state, action), prob)

    walk(state, prob)
    return infosets, terminals


def test_chance_deals_distinct_cards_with_uniform_probability():
    game = LeducPoker()
    root = game.root()
    outcomes0 = game.chance_outcomes(root)
    assert sorted(c for c, _ in outcomes0) == list(range(6))
    assert all(p == pytest.approx(1 / 6) for _, p in outcomes0)

    state_after_p0 = game.step(root, 0)
    outcomes1 = game.chance_outcomes(state_after_p0)
    assert sorted(c for c, _ in outcomes1) == [1, 2, 3, 4, 5]
    assert all(p == pytest.approx(1 / 5) for _, p in outcomes1)


def test_root_betting_state_after_deal():
    game = LeducPoker()
    state = _deal(game, 0, 4)  # Js vs Ks
    assert not state.chance
    assert not state.terminal
    assert state.hole_cards == (0, 4)
    assert state.round == 0
    assert state.player == 0
    assert set(game.legal_actions(state)) == {Action.CHECK_CALL, Action.BET_RAISE}


def test_legal_actions_facing_a_bet_include_fold():
    game = LeducPoker()
    state = _deal(game, 0, 4)
    state = game.step(state, Action.BET_RAISE)
    assert set(game.legal_actions(state)) == {Action.FOLD, Action.CHECK_CALL, Action.BET_RAISE}


def test_raise_cap_of_two_removes_bet_raise():
    game = LeducPoker()
    state = _deal(game, 0, 4)
    state = game.step(state, Action.BET_RAISE)
    state = game.step(state, Action.BET_RAISE)
    assert set(game.legal_actions(state)) == {Action.FOLD, Action.CHECK_CALL}


def test_fold_ends_the_hand_immediately():
    game = LeducPoker()
    state = _deal(game, 0, 2)  # Js vs Qs
    state = game.step(state, Action.BET_RAISE)
    state = game.step(state, Action.FOLD)
    assert state.terminal
    assert state.round == 0
    returns = game.returns(state)
    assert sum(returns) == pytest.approx(0.0)
    assert returns[0] == pytest.approx(1.0)  # won P1's ante
    assert returns[1] == pytest.approx(-1.0)


def test_checked_round_deals_a_board_card_and_keeps_full_history():
    game = LeducPoker()
    state = _deal(game, 0, 4)
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    assert state.chance
    assert not state.terminal
    assert state.round == 1
    assert state.board == ()

    outcomes = game.chance_outcomes(state)
    assert sorted(c for c, _ in outcomes) == [1, 2, 3, 5]  # excludes both hole cards

    state = game.step(state, outcomes[0][0])
    assert not state.chance
    # history spans the whole hand: round 0's checks are still visible, since
    # a real player can see the full public betting, not just this round's.
    assert state.history == (Action.CHECK_CALL, Action.CHECK_CALL)
    assert state.round_start == 2
    assert state.player == 0
    assert state.board == (outcomes[0][0],)
    assert set(game.legal_actions(state)) == {Action.CHECK_CALL, Action.BET_RAISE}


def test_different_preflop_lines_reaching_round_one_are_distinct_infosets():
    # Checked-through and bet-called preflop both leave P0 to act with a
    # fresh round-1 history, but at different pot sizes -- a real player can
    # see which happened, so these must NOT collapse into one infoset.
    game = LeducPoker()

    checked = _deal(game, 0, 4)
    checked = game.step(checked, Action.CHECK_CALL)
    checked = game.step(checked, Action.CHECK_CALL)
    checked = game.step(checked, 2)  # board Qs

    bet_called = _deal(game, 0, 4)
    bet_called = game.step(bet_called, Action.BET_RAISE)
    bet_called = game.step(bet_called, Action.CHECK_CALL)
    bet_called = game.step(bet_called, 2)  # board Qs

    assert checked.player == bet_called.player == 0
    assert checked.board == bet_called.board
    assert checked.pot != bet_called.pot
    assert checked.infoset_key(0) != bet_called.infoset_key(0)


def test_showdown_higher_rank_wins_when_no_pair():
    game = LeducPoker()
    state = _deal(game, 0, 4)  # Js vs Ks
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, 2)  # board Qs, pairs neither
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    assert state.terminal
    returns = game.returns(state)
    assert sum(returns) == pytest.approx(0.0)
    assert returns[1] > 0  # King beats Jack
    assert returns[0] < 0


def test_showdown_pair_beats_higher_unpaired():
    game = LeducPoker()
    state = _deal(game, 0, 4)  # Js vs Ks
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, 1)  # board Jh: pairs player 0's Jack
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    assert state.terminal
    returns = game.returns(state)
    assert sum(returns) == pytest.approx(0.0)
    assert returns[0] > 0  # pair of Jacks beats an unpaired King


def test_showdown_unpaired_equal_ranks_split_the_pot():
    game = LeducPoker()
    state = _deal(game, 0, 1)  # Js vs Jh
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, 4)  # board Ks, pairs neither
    state = game.step(state, Action.CHECK_CALL)
    state = game.step(state, Action.CHECK_CALL)
    assert state.terminal
    returns = game.returns(state)
    assert returns == (0.0, 0.0)


def test_full_raise_war_costs_match_bet_sizes():
    game = LeducPoker()
    state = _deal(game, 0, 4)
    state = game.step(state, Action.BET_RAISE)  # P0 -> pays 2
    state = game.step(state, Action.BET_RAISE)  # P1 -> pays 4 (call 2 + raise 2)
    state = game.step(state, Action.CHECK_CALL)  # P0 calls -> pays 2 more
    assert state.round == 1
    assert state.pot == 2 + 2 + 4 + 2  # ante*2 + the round-0 action costs
    assert state.stacks == (8, 8)


def test_full_tree_is_zero_sum():
    game = LeducPoker()
    _, terminals = _collect_tree(game, game.root())
    assert len(terminals) > 0
    for _, returns in terminals:
        assert sum(returns) == pytest.approx(0.0)


def test_leduc_infoset_count():
    # Round 0: 6 hole cards x 6 decision histories = 36.
    # Round 1: 6 hole cards x 5 boards (excludes hero's own card) x 5 distinct
    # non-fold round-0 endings (cc, bc, cbc, bbc, cbbc -- each a different
    # pot size / public history a player can see) x 6 round-1 decision
    # histories = 900. Total 936, matching the standard Leduc Hold'em info
    # state count. Regression check on this implementation's tree shape.
    game = LeducPoker()
    infosets, _ = _collect_tree(game, game.root())
    assert len(infosets) == 936
