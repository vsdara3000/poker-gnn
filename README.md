# Poker GNN Solver

A poker solver that represents each decision point as a graph and learns a strategy with a graph neural network trained by Deep CFR (counterfactual regret minimization). It works up a ladder of games of increasing size:

| Game | Size | How it's solved | How it's checked |
|---|---|---|---|
| **Kuhn** | 3 cards, 1 betting round, 12 infosets | tabular CFR, Deep CFR + GNN | exact exploitability, plus the known closed-form Nash strategy |
| **Leduc** | 6 cards, 2 rounds, 936 infosets | tabular CFR, Deep CFR + GNN | exact exploitability |
| **Heads-up limit hold'em (HULHE)** | 52 cards, 4 streets, ~10^14 infosets | external-sampling MCCFR, sampled Deep CFR + GNN | duplicate chip EV against fixed baselines, and Local Best Response (an exploitability lower bound) |

Kuhn and Leduc are small enough to compute an exact best response, so every component (rules, CFR, graph encoding, GNN) was validated there before it was used on hold'em. HULHE is too big to enumerate, so it is evaluated with sampling-based metrics instead. HULHE is still in progress. [docs/PLAN.md](docs/PLAN.md) has the design, milestones, and a log of every experiment with its results and caveats.

## How it works

1. **Game rules** (`games/`): Kuhn, Leduc, and HULHE behind one `Game`/`State` interface (`root`, `legal_actions`, `chance_outcomes`, `step`, `returns`), so every solver and evaluator runs on all three.
2. **Infoset → graph** (`graphs/`): what one player can see becomes a graph with a node per card (hole and board cards flagged) and a path of nodes for the public betting history. The opponent's hole cards are never included.
3. **GNN** (`models/`): message passing over that graph. Informative card nodes and action nodes are pooled separately, so the ~45 empty card slots in hold'em don't wash out the signal. A hand-strength feature is added to the pooled vector. Separate heads predict per-action regrets (advantage net) and the average strategy (policy net).
4. **Solvers** (`solver/`):
   - `cfr.py`: vanilla full-width CFR, exact on Kuhn/Leduc.
   - `mccfr.py`: external-sampling MCCFR, a tabular baseline for HULHE.
   - `deep_cfr.py`: Deep CFR (Brown et al. 2019) with the GNN. On HULHE it uses external sampling, batching many concurrent traversals into one forward pass. An optional learned range net (`--range-net`) predicts the player's equity against the opponent's actual hand from the betting so far.
   - `distill.py`: fits a policy net to MCCFR's table, so that on HULHE MCCFR is compared like-for-like with Deep CFR.
5. **Evaluation** (`eval/`):
   - `exploitability.py`: exact best response (Kuhn/Leduc).
   - `baseline_eval.py`: duplicate dealing (every deal played from both seats) against always-fold, always-call, and uniform-random, with 95% confidence intervals.
   - `lbr.py`: Local Best Response (Lisý & Bowling 2017), an adaptive opponent whose winnings are a lower bound on HULHE exploitability.

## Reproducing it

### Setup

Requires Python 3.10+. CPU is enough: the bottleneck is Python-side tree traversal, not matrix math.

```bash
git clone https://github.com/vsdara3000/poker-gnn.git
cd poker-gnn

# with uv (uses the pinned uv.lock, CPU-only torch)
uv sync --extra dev
source .venv/bin/activate

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

On WSL, clone into the Linux filesystem (e.g. `~/poker-gnn`), not under `/mnt/c`. Reading files across the Windows boundary made startup about 7x slower.

### 1. Check that everything is correct

```bash
pytest
```

The tests check the rules of each game against hand-scripted lines and random playouts, check the hand evaluator against known 5-card category odds, check that tabular CFR and MCCFR recover Kuhn's known Nash equilibrium, check that Deep CFR lands far below uniform-random exploitability, and round-trip save/load for every solver. The HULHE tests take a few minutes.

### 2. Solve the small games exactly

```bash
python scripts/solve_tabular.py --game kuhn  --iterations 20000
python scripts/solve_tabular.py --game leduc --iterations 2000
```

This prints exploitability at each checkpoint. It should head toward 0. For Kuhn, compare the printed strategy with the published Nash family. For example, player 1 with a Jack facing a bet always folds.

### 3. Train the GNN on the small games

```bash
python scripts/train.py --game kuhn  --solver deep_cfr --iterations 500
python scripts/train.py --game leduc --solver deep_cfr --iterations 200
```

This reports exact exploitability. Compare it with the tabular run above and with uniform-random play.

### 4. Train on hold'em

```bash
# Deep CFR + GNN; --lbr-hands adds the Local Best Response bound (slower)
python scripts/train.py --game hulhe --solver deep_cfr --iterations 3000 \
    --eval-hands 1000 --lbr-hands 200 --save checkpoints/hulhe.pt

# Tabular MCCFR baseline (cheap per iteration, but its table only covers part of the game)
python scripts/train.py --game hulhe --solver mccfr --iterations 100000

# Re-evaluate a saved solver without retraining
python scripts/evaluate.py --game hulhe --solver deep_cfr --load checkpoints/hulhe.pt --lbr-hands 200
```

Each checkpoint prints chip EV per hand against each baseline, as `P0` / `P1` / `dup` (seat-averaged) with 95% intervals, and the LBR bound if enabled. Read `dup` and its interval rather than single-seat numbers, because the single-seat numbers are noisy.

### 5. Compare configurations across seeds

```bash
python scripts/sweep.py --variants deep deep_range deep_big deep_big_range \
    --seeds 0 1 2 --iterations 3000 --lbr-hands 200 --out runs/sweep
python scripts/sweep.py --summarize-only --out runs/sweep
```

This runs one process per (config, seed) in parallel and prints a mean ± spread table of the final checkpoints. Finished runs are skipped if you re-launch after an interruption. The variants are defined in `VARIANTS` at the top of `scripts/sweep.py`.

### Play Kuhn by hand

```bash
PYTHONPATH=src python scripts/play_kuhn.py
```

## Layout

```
src/poker_gnn/
  games/     base.py (Game/State/Action), kuhn.py, leduc.py, hulhe.py
  graphs/    card_graph.py, betting_graph.py, infoset_graph.py
  models/    encoder.py (infoset -> graph), gnn.py (PokerGNN)
  solver/    cfr.py, mccfr.py, deep_cfr.py, distill.py, policy_net.py
  eval/      exploitability.py, baseline_eval.py, lbr.py
  utils/     cards.py (5/6/7-card hand evaluator)
scripts/     train.py, evaluate.py, solve_tabular.py, sweep.py, play_kuhn.py
tests/       one test module per component
configs/     per-game hyperparameter notes
docs/PLAN.md design, milestones, and the full experiment log
```

## References

- Zinkevich et al., *Regret Minimization in Games with Incomplete Information*, NeurIPS 2007 (CFR)
- Lanctot et al., *Monte Carlo Sampling for Regret Minimization in Extensive Games*, NeurIPS 2009 (external-sampling MCCFR)
- Brown, Lerer, Gross, Sandholm, *Deep Counterfactual Regret Minimization*, ICML 2019
- Lisý & Bowling, *Equilibrium Approximation Quality of Current No-Limit Poker Bots*, AAAI-17 Workshop 2017 (Local Best Response)
