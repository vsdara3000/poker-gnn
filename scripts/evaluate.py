"""Load a solver saved by scripts/train.py (--save) and report on it,
without re-solving from scratch.

Reports exact exploitability for kuhn/leduc (full-width solvers), or
baseline-eval chip EV against fixed opponents (fold/call/random) otherwise
-- see `poker_gnn.eval.baseline_eval` for why that's a weaker signal than
exploitability. Loading is solver-type-specific (`TabularCFR`/`Deep
CFR`/`ExternalSamplingCFR` all serialize differently), so `--solver` must
match whatever `scripts/train.py --solver ...` produced the file.

Examples:
  python scripts/evaluate.py --game hulhe --solver deep_cfr --load checkpoints/hulhe.pt
  python scripts/evaluate.py --game kuhn --solver cfr --load checkpoints/kuhn_cfr.pkl
"""

import argparse
import random

from _report import report_solver
from poker_gnn.games import make_game
from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.solver.deep_cfr import DeepCFR
from poker_gnn.solver.mccfr import ExternalSamplingCFR

LOADERS = {
    "cfr": TabularCFR.load,
    "mccfr": ExternalSamplingCFR.load,
    "deep_cfr": DeepCFR.load,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--game", default="hulhe", help="kuhn, leduc, or hulhe")
    parser.add_argument("--solver", default="deep_cfr", choices=list(LOADERS))
    parser.add_argument("--load", required=True, help="path a scripts/train.py --save run wrote")
    parser.add_argument("--eval-hands", type=int, default=2000, help="simulated hands per baseline matchup")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    game = make_game(args.game)
    solver = LOADERS[args.solver](args.load)
    rng = random.Random(args.seed)

    print(f"loaded {args.solver} from {args.load}")
    report_solver(game, args.game, solver, rng, args.eval_hands)


if __name__ == "__main__":
    main()
