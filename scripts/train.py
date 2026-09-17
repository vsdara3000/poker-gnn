"""Train Deep CFR (GNN) or tabular external-sampling CFR, with progress checkpoints.

Small games (kuhn, leduc) can be enumerated exactly, so checkpoints there
report exact exploitability. HULHE can't be enumerated (~10^14 infosets), so
checkpoints report baseline-eval chip EV against fixed opponents (always
fold / always call / uniform random) instead -- see
`poker_gnn.eval.baseline_eval` for why that's a weaker signal than
exploitability.

Examples:
  python scripts/train.py --game kuhn --solver deep_cfr --iterations 2000
  python scripts/train.py --game hulhe --solver deep_cfr --iterations 5000
  python scripts/train.py --game hulhe --solver mccfr --iterations 20000
  python scripts/train.py --game hulhe --solver deep_cfr --iterations 5000 --save checkpoints/hulhe.pt

`--save PATH` persists the trained solver at the end (see `<Solver>.save`/
`.load`) so a later `scripts/evaluate.py --load PATH` run, or more training,
doesn't need to start over.
"""

import argparse
import random

from _report import SMALL_GAMES, report_solver
from poker_gnn.games import make_game
from poker_gnn.solver.deep_cfr import DeepCFR
from poker_gnn.solver.mccfr import ExternalSamplingCFR


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--game", default="hulhe", help="kuhn, leduc, or hulhe")
    parser.add_argument("--solver", default="deep_cfr", choices=["deep_cfr", "mccfr"])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--checkpoints", type=int, default=10, help="progress evaluations during training")
    parser.add_argument("--eval-hands", type=int, default=2000, help="simulated hands per baseline matchup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--external-sampling",
        dest="external_sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="deep_cfr only; default: on for hulhe (required at that scale), off for kuhn/leduc",
    )
    # deep_cfr hyperparameters (ignored for --solver mccfr)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--buffer-capacity", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--parallel-traversals", type=int, default=16)
    parser.add_argument("--save", help="path to persist the trained solver to when done")
    return parser.parse_args()


def make_solver(args):
    if args.solver == "mccfr":
        return ExternalSamplingCFR(seed=args.seed)

    external_sampling = args.external_sampling
    if external_sampling is None:
        external_sampling = args.game not in SMALL_GAMES
    return DeepCFR(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        train_steps_per_iteration=args.train_steps,
        seed=args.seed,
        external_sampling=external_sampling,
        parallel_traversals=args.parallel_traversals,
    )


def run_iterations(solver, game, chunk: int) -> None:
    if isinstance(solver, ExternalSamplingCFR):
        solver.iterate(game, chunk)
    else:
        solver.train(game, chunk)


def main():
    args = parse_args()
    game = make_game(args.game)
    solver = make_solver(args)
    rng = random.Random(args.seed)

    step = max(args.iterations // args.checkpoints, 1)
    done = 0
    while done < args.iterations:
        chunk = min(step, args.iterations - done)
        run_iterations(solver, game, chunk)
        done += chunk
        print(f"iterations={done:>8d}")
        report_solver(game, args.game, solver, rng, args.eval_hands)

    if args.save:
        solver.save(args.save)
        print(f"saved solver to {args.save}")


if __name__ == "__main__":
    main()
