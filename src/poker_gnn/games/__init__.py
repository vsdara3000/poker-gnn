from poker_gnn.games.base import Action, Game, Player, State
from poker_gnn.games.kuhn import KuhnPoker
from poker_gnn.games.leduc import LeducPoker
from poker_gnn.games.hulhe import HeadsUpLimitHoldem

# Implement in order: kuhn → leduc → hulhe
GAMES = {
    "kuhn": KuhnPoker,
    "leduc": LeducPoker,
    "hulhe": HeadsUpLimitHoldem,
}


def make_game(name: str, **kwargs) -> Game:
    try:
        return GAMES[name.lower()](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown game {name!r}. Choose from {sorted(GAMES)}") from exc
