"""Play a hand of Kuhn poker by hand from the terminal.

Dev tool for poking at the game logic before CFR exists. You play both
seats; the deal is random each hand.

Run: PYTHONPATH=src python3 scripts/play_kuhn.py
"""

import random

from poker_gnn.games.base import Action
from poker_gnn.games.kuhn import KuhnPoker

ACTION_NAMES = {Action.FOLD: "fold", Action.CHECK_CALL: "check/call", Action.BET_RAISE: "bet/raise"}


def prompt_action(game, state):
    legal = game.legal_actions(state)
    options = ", ".join(f"{int(a)}={ACTION_NAMES[a]}" for a in legal)
    while True:
        raw = input(f"  Player {state.player} to act [{options}]: ").strip()
        try:
            action = Action(int(raw))
        except (ValueError, KeyError):
            print("  not a number, try again")
            continue
        if action in legal:
            return action
        print("  illegal action, try again")


def main():
    game = KuhnPoker()
    state = game.root()

    print("Dealing...")
    while state.chance:
        outcomes = game.chance_outcomes(state)
        card, _ = random.choices(outcomes, weights=[p for _, p in outcomes])[0]
        state = game.step(state, card)
    names = game.card_names()
    print(f"P0 dealt {names[state.hole_cards[0]]}, P1 dealt {names[state.hole_cards[1]]}")
    print(f"Pot after antes: {state.pot}")

    while not state.terminal:
        action = prompt_action(game, state)
        state = game.step(state, action)
        print(f"  history so far: {[ACTION_NAMES[a] for a in state.history]}, pot={state.pot}")

    returns = game.returns(state)
    print(f"\nHand over. History: {[ACTION_NAMES[a] for a in state.history]}")
    print(f"P0 ({names[state.hole_cards[0]]}) return: {returns[0]:+.1f}")
    print(f"P1 ({names[state.hole_cards[1]]}) return: {returns[1]:+.1f}")


if __name__ == "__main__":
    main()
