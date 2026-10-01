"""Game rules, one module per game, all behind the `Game` interface in `base.py`.

The three games form a difficulty ladder (see docs/PLAN.md): Kuhn to check
the pipeline by hand, Leduc for multi-round play with a board card, then
heads-up limit hold'em at full scale. Solvers and encoders only see `Game`
and `State`, so the same code runs on all three.
"""

from poker_gnn.games.base import Action, Game, Player, State
from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.games.leduc import LeducPoker
from poker_gnn.games.hulhe import HeadsUpLimitHoldem

# Name -> class registry used by the CLI scripts (`--game kuhn|leduc|hulhe`).
GAMES = {
    "kuhn": KuhnPoker,
    "leduc": LeducPoker,
    "hulhe": HeadsUpLimitHoldem,
}


def make_game(name: str, **kwargs) -> Game:
    """Instantiate a game by its registry name (case-insensitive)."""
    try:
        return GAMES[name.lower()](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown game {name!r}. Choose from {sorted(GAMES)}") from exc
