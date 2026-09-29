"""Run several `scripts/train.py` configurations x seeds in parallel, one
process per run, then summarize their final checkpoints.

Each training run is effectively single-threaded (DeepCFR pins torch to one
thread, and the bottleneck is Python-side traversal), so N runs on N cores
finish in about the time of one. Memory, not CPU, is the limit: a long
mccfr run's dict grows to millions of infosets (several GB), so keep
`--jobs` modest when mccfr variants are included.

Each run writes `<out>/<variant>_s<seed>.log` and `.pt` (the saved solver,
for `scripts/evaluate.py --load` later). Runs whose log already ends with a
saved solver are skipped, so a sweep can be re-launched after an
interruption without redoing finished work.

Examples:
  python scripts/sweep.py --variants deep deep_range --seeds 0 1 2 --iterations 3000 --lbr-hands 200
  python scripts/sweep.py --variants mccfr --seeds 0 1 --mccfr-iterations 100000 --jobs 2
  python scripts/sweep.py --summarize-only --out runs/sweep
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# Extra train.py arguments per variant (on top of --game/--seed/--iterations).
VARIANTS = {
    "deep": ["--solver", "deep_cfr"],
    "deep_range": ["--solver", "deep_cfr", "--range-net"],
    "deep_big": ["--solver", "deep_cfr", "--hidden-dim", "64", "--buffer-capacity", "20000", "--train-steps", "8"],
    "deep_big_range": [
        "--solver", "deep_cfr", "--hidden-dim", "64", "--buffer-capacity", "20000", "--train-steps", "8", "--range-net",
    ],
    "mccfr": ["--solver", "mccfr"],
}

METRIC_RE = re.compile(r"^\s+vs_(\w+): P0=\S+\s+P1=\S+\s+dup=([+-][\d.]+)±([\d.]+)")
LBR_RE = re.compile(r"^\s+lbr .*?: ([+-][\d.]+)±([\d.]+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["deep", "deep_range"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--game", default="hulhe")
    parser.add_argument("--iterations", type=int, default=3000, help="deep_cfr variants")
    parser.add_argument("--mccfr-iterations", type=int, default=100000, help="mccfr variant")
    parser.add_argument("--checkpoints", type=int, default=5)
    parser.add_argument("--eval-hands", type=int, default=1000)
    parser.add_argument("--lbr-hands", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=max((os.cpu_count() or 2) // 2, 1))
    parser.add_argument("--out", default="runs/sweep")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def run_one(args, variant: str, seed: int) -> tuple[str, int, float]:
    name = f"{variant}_s{seed}"
    log_path = os.path.join(args.out, f"{name}.log")
    save_path = os.path.join(args.out, f"{name}.pt")
    if os.path.exists(save_path) and os.path.exists(log_path):
        with open(log_path) as f:
            if "saved solver to" in f.read():
                print(f"[skip] {name} (already finished)", flush=True)
                return name, 0, 0.0

    iterations = args.mccfr_iterations if variant == "mccfr" else args.iterations
    cmd = [
        sys.executable, os.path.join(HERE, "train.py"),
        "--game", args.game, "--seed", str(seed), "--iterations", str(iterations),
        "--checkpoints", str(args.checkpoints), "--eval-hands", str(args.eval_hands),
        "--lbr-hands", str(args.lbr_hands), "--save", save_path, *VARIANTS[variant],
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    print(f"[start] {name}", flush=True)
    start = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start
    print(f"[{'done' if code == 0 else 'FAILED'}] {name} in {elapsed / 60:.1f} min (exit {code})", flush=True)
    return name, code, elapsed


def final_metrics(log_path: str) -> dict[str, float]:
    """Duplicate EV per baseline (and LBR) at the log's last checkpoint."""
    metrics: dict[str, float] = {}
    with open(log_path) as f:
        for line in f:
            if line.startswith("iterations="):
                metrics = {}
            elif m := METRIC_RE.match(line):
                metrics[m.group(1)] = float(m.group(2))
            elif m := LBR_RE.match(line):
                metrics["lbr"] = float(m.group(1))
    return metrics


def summarize(args) -> None:
    """Per variant: mean and spread across seeds of each final metric."""
    print("\nfinal checkpoint, duplicate EV (chips/hand); lbr = exploitability lower bound, lower is better")
    columns = ["fold", "call", "random", "lbr"]
    print(f"{'variant':<16}{'seeds':>6}" + "".join(f"{c:>18}" for c in columns))
    for variant in args.variants:
        per_seed = []
        for seed in args.seeds:
            path = os.path.join(args.out, f"{variant}_s{seed}.log")
            if os.path.exists(path) and (m := final_metrics(path)):
                per_seed.append(m)
        if not per_seed:
            continue
        cells = []
        for c in columns:
            values = [m[c] for m in per_seed if c in m]
            if not values:
                cells.append(f"{'-':>18}")
            elif len(values) == 1:
                cells.append(f"{values[0]:>+18.3f}")
            else:
                # mean ± sample std across seeds (seed-to-seed spread, not a CI)
                cells.append(f"{statistics.mean(values):+.3f} ±{statistics.stdev(values):.3f}".rjust(18))
        print(f"{variant:<16}{len(per_seed):>6}" + "".join(cells))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.summarize_only:
        runs = [(v, s) for v in args.variants for s in args.seeds]
        print(f"{len(runs)} runs, {args.jobs} at a time, logs in {args.out}/", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda r: run_one(args, *r), runs))
        failed = [name for name, code, _ in results if code != 0]
        if failed:
            print(f"failed: {', '.join(failed)} (see their logs)")
    summarize(args)


if __name__ == "__main__":
    main()
