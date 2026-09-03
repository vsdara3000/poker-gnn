"""Run tabular CFR on a small game and report exploitability + the average strategy.

Example: python scripts/solve_tabular.py --game kuhn --iterations 20000
"""

import argparse

from poker_gnn.eval.exploitability import exploitability
from poker_gnn.games import make_game
from poker_gnn.solver.cfr import TabularCFR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="kuhn", help="kuhn, leduc, or hulhe")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--checkpoints", type=int, default=10, help="progress prints during training")
    return parser.parse_args()


def main():
    args = parse_args()
    game = make_game(args.game)
    cfr = TabularCFR()

    step = max(args.iterations // args.checkpoints, 1)
    done = 0
    while done < args.iterations:
        chunk = min(step, args.iterations - done)
        cfr.iterate(game, chunk)
        done += chunk
        exp = exploitability(game, cfr.average_strategy())
        print(f"iterations={done:>8d}  exploitability={exp:.6f}")

    print("\nAverage strategy:")
    strategy = cfr.average_strategy()
    for key in sorted(strategy):
        probs = ", ".join(f"{action.name}={p:.3f}" for action, p in strategy[key].items())
        print(f"  {key}: {probs}")


if __name__ == "__main__":
    main()
