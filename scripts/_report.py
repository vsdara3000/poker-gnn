"""Shared progress/evaluation reporting for scripts/train.py and evaluate.py.

Small games (kuhn, leduc) can be enumerated exactly, so this reports exact
exploitability there. HULHE can't be enumerated (~10^14 infosets), so it
reports duplicate baseline-eval chip EV against fixed opponents instead --
see `poker_gnn.eval.baseline_eval` for why that's a weaker signal than
exploitability -- plus, optionally, an LBR exploitability lower bound.
Dict-only solvers get a distilled strategy network first so they are
evaluated like-for-like with DeepCFR. Kept as CLI glue rather than part of
the `poker_gnn` package, since it couples `eval` to specific solver classes
purely for this reporting convenience. Not run directly.
"""

from __future__ import annotations

import random

from poker_gnn.eval.baseline_eval import (
    call_policy,
    evaluate_matchup,
    fold_policy,
    make_random_policy,
    make_solver_policy,
    make_strategy_policy,
)
from poker_gnn.eval.exploitability import exploitability
from poker_gnn.eval.lbr import local_best_response
from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.solver.distill import DistilledPolicy
from poker_gnn.solver.mccfr import ExternalSamplingCFR

SMALL_GAMES = {"kuhn", "leduc"}
# Solvers with no generalizing network -- their policy is a plain dict
# (average_strategy()), unlike DeepCFR's policy(), which falls back to a
# trained strategy network for infosets never exactly visited.
DICT_ONLY_SOLVERS = (TabularCFR, ExternalSamplingCFR)


def can_enumerate(game_name: str, solver) -> bool:
    """True if exact exploitability is affordable and meaningful: a small
    game, and a solver whose average_strategy() dict covers it."""
    if game_name not in SMALL_GAMES:
        return False
    if isinstance(solver, DICT_ONLY_SOLVERS):
        return True
    return not solver.external_sampling  # DeepCFR, full-width mode


def playable_strategy(game, solver, rng: random.Random, distill: bool = True):
    """Something with `policy(game, state)` to evaluate. DeepCFR already has
    a network fallback. A dict-only solver gets one fit post hoc
    (`DistilledPolicy`) unless `distill=False`, in which case it keeps the
    old uniform-random fallback (returns None; callers use the dict)."""
    if not isinstance(solver, DICT_ONLY_SOLVERS):
        return solver
    if not distill:
        return None
    visits = solver.visit_counts() if hasattr(solver, "visit_counts") else None
    distilled = DistilledPolicy.fit(game, solver.average_strategy(), rng, visits=visits)
    print(f"    distilled strategy net from {distilled.num_examples} examples")
    return distilled


def report_solver(
    game,
    game_name: str,
    solver,
    rng: random.Random,
    eval_hands: int = 2000,
    lbr_hands: int = 0,
    distill: bool = True,
) -> None:
    """Print exploitability (small games) or baseline-eval chip EV (HULHE,
    or a large-scale solver on any game) for the given trained solver, plus
    an LBR exploitability lower bound on HULHE if `lbr_hands` > 0."""
    if can_enumerate(game_name, solver):
        exp = exploitability(game, solver.average_strategy())
        print(f"    exploitability={exp:.6f}")
        return

    strategy = playable_strategy(game, solver, rng, distill)
    if strategy is None:
        policy = make_strategy_policy(solver.average_strategy(), rng)
    else:
        policy = make_solver_policy(strategy, rng)

    # +- is a 95% confidence half-width (1.96 standard errors).
    baselines = {"fold": fold_policy, "call": call_policy, "random": make_random_policy(rng)}
    for name, opponent in baselines.items():
        r = evaluate_matchup(game, policy, opponent, eval_hands, rng)
        print(
            f"    vs_{name}: P0={r.as_p0:+.3f}±{1.96 * r.as_p0_se:.3f}  "
            f"P1={r.as_p1:+.3f}±{1.96 * r.as_p1_se:.3f}  "
            f"dup={r.duplicate:+.3f}±{1.96 * r.duplicate_se:.3f}"
        )

    if lbr_hands > 0 and game_name == "hulhe" and strategy is None:
        print("    lbr skipped: needs a generalizing strategy (drop --no-distill)")
    if lbr_hands > 0 and game_name == "hulhe" and strategy is not None:
        r = local_best_response(game, strategy, lbr_hands, rng)
        print(
            f"    lbr (exploitability lower bound, chips/hand): "
            f"{r.duplicate:+.3f}±{1.96 * r.duplicate_se:.3f}  "
            f"(LBR as P0 {r.as_p0:+.3f}, as P1 {r.as_p1:+.3f})"
        )
