"""Kuhn game + known Nash checks — implement when the game works."""

import pytest

from poker_gnn.games.base import Action
from poker_gnn.games.kuhn import KuhnPoker


def _deal(game, card0, card1):
    """Step through the two chance nodes to reach a betting state with fixed hole cards."""
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
    game = KuhnPoker()
    root = game.root()
    outcomes0 = game.chance_outcomes(root)
    assert sorted(c for c, _ in outcomes0) == [0, 1, 2]
    assert all(p == pytest.approx(1 / 3) for _, p in outcomes0)

    state_after_p0 = game.step(root, 0)
    outcomes1 = game.chance_outcomes(state_after_p0)
    assert sorted(c for c, _ in outcomes1) == [1, 2]
    assert all(p == pytest.approx(1 / 2) for _, p in outcomes1)


def test_root_betting_state_after_deal():
    game = KuhnPoker()
    state = _deal(game, 0, 2)  # J vs K
    assert not state.chance
    assert not state.terminal
    assert state.hole_cards == (0, 2)
    assert state.player == 0
    assert set(game.legal_actions(state)) == {Action.CHECK_CALL, Action.BET_RAISE}


def test_legal_actions_facing_a_bet_are_fold_or_call():
    game = KuhnPoker()
    state = _deal(game, 0, 2)
    state = game.step(state, Action.BET_RAISE)
    assert set(game.legal_actions(state)) == {Action.FOLD, Action.CHECK_CALL}


@pytest.mark.parametrize(
    "actions,winner,expected_payoff",
    [
        ((Action.CHECK_CALL, Action.CHECK_CALL), 1, 1),  # showdown, K beats J, ante only
        ((Action.BET_RAISE, Action.FOLD), 0, 1),  # P0 bets, P1 folds, wins P1's ante
        ((Action.BET_RAISE, Action.CHECK_CALL), 1, 2),  # showdown after a called bet
        ((Action.CHECK_CALL, Action.BET_RAISE, Action.FOLD), 1, 1),  # P0 folds to a bet
        ((Action.CHECK_CALL, Action.BET_RAISE, Action.CHECK_CALL), 1, 2),  # showdown
    ],
)
def test_known_terminal_payoffs(actions, winner, expected_payoff):
    game = KuhnPoker()
    state = _deal(game, 0, 2)  # P0 has J, P1 has K
    for action in actions:
        state = game.step(state, action)
    assert state.terminal
    returns = game.returns(state)
    assert sum(returns) == pytest.approx(0.0)
    loser = 1 - winner
    assert returns[winner] == pytest.approx(expected_payoff)
    assert returns[loser] == pytest.approx(-expected_payoff)


def test_full_tree_is_zero_sum_with_thirty_terminal_nodes():
    # 6 deals (3 x 2 distinct hole cards) x 5 terminal histories each (cc, bf, bc, cbf, cbc).
    game = KuhnPoker()
    _, terminals = _collect_tree(game, game.root())
    assert len(terminals) == 30
    for _, returns in terminals:
        assert sum(returns) == pytest.approx(0.0)


def test_kuhn_has_twelve_infosets():
    game = KuhnPoker()
    infosets, _ = _collect_tree(game, game.root())
    assert len(infosets) == 12
