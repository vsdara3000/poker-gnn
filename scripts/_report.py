"""Shared progress/evaluation reporting for scripts/train.py and evaluate.py.

Small games (kuhn, leduc) can be enumerated exactly, so this reports exact
exploitability there. HULHE can't be enumerated (~10^14 infosets), so it
reports baseline-eval chip EV against fixed opponents instead -- see
`poker_gnn.eval.baseline_eval` for why that's a weaker signal than
exploitability. Kept as CLI glue rather than part of the `poker_gnn` package
itself, since it couples `eval` to a specific solver (`ExternalSamplingCFR`)
purely for this reporting convenience.
"""

from __future__ import annotations

import random

from poker_gnn.eval.baseline_eval import (
    average_payoff,
    call_policy,
    fold_policy,
    make_random_policy,
    make_solver_policy,
    make_strategy_policy,
)
from poker_gnn.eval.exploitability import exploitability
from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.solver.mccfr import ExternalSamplingCFR

SMALL_GAMES = {"kuhn", "leduc"}
# Solvers with no generalizing network -- their policy is a plain dict
# (average_strategy()), unlike DeepCFR's policy(), which falls back to a
# trained strategy network for infosets never exactly visited.
DICT_ONLY_SOLVERS = (TabularCFR, ExternalSamplingCFR)


def can_enumerate(game_name: str, solver) -> bool:
    if game_name not in SMALL_GAMES:
        return False
    if isinstance(solver, DICT_ONLY_SOLVERS):
        return True
    return not solver.external_sampling  # DeepCFR, full-width mode


def report_solver(game, game_name: str, solver, rng: random.Random, eval_hands: int = 2000) -> None:
    """Print exploitability (small games) or baseline-eval chip EV (HULHE,
    or a large-scale solver on any game) for the given trained solver."""
    if can_enumerate(game_name, solver):
        exp = exploitability(game, solver.average_strategy())
        print(f"    exploitability={exp:.6f}")
        return

    if isinstance(solver, DICT_ONLY_SOLVERS):
        policy = make_strategy_policy(solver.average_strategy(), rng)
    else:
        policy = make_solver_policy(solver, rng)

    baselines = {"fold": fold_policy, "call": call_policy, "random": make_random_policy(rng)}
    for name, opponent in baselines.items():
        p0, _ = average_payoff(game, {0: policy, 1: opponent}, eval_hands, rng)
        _, p1 = average_payoff(game, {0: opponent, 1: policy}, eval_hands, rng)
        print(f"    vs_{name}: P0={p0:+.3f}  P1={p1:+.3f}")
