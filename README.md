# Poker GNN Solver

Research codebase for a graph-neural poker solver: **Kuhn** → **Leduc** → **heads-up limit hold'em (HULHE)**.

Kuhn and Leduc are solved (tabular CFR reaches known Nash, GNN/Deep CFR trained on top). HULHE is in progress: rules, hand evaluator, and external-sampling CFR (tabular + Deep CFR/GNN) are implemented and validated, but no exploitability/convergence claim exists yet at that scale — see [docs/PLAN.md](docs/PLAN.md) for the full roadmap and current status.

## Quick start

```bash
pip install -e ".[dev]"
pytest

# Tabular CFR on a small game (exact for Kuhn/Leduc)
python scripts/solve_tabular.py --game kuhn --iterations 20000

# Play a hand of Kuhn by hand, for poking at the rules
PYTHONPATH=src python3 scripts/play_kuhn.py
```

`scripts/train.py` and `scripts/evaluate.py` are not wired up yet; Deep CFR / external-sampling CFR and baseline evaluation on HULHE are currently driven directly from `poker_gnn.solver` / `poker_gnn.eval` (see `docs/PLAN.md` and `tests/test_deep_cfr.py`, `tests/test_mccfr.py`, `tests/test_baseline_eval.py` for usage examples), not yet from a CLI.

## Layout

- `src/poker_gnn/games` — `base.py` (shared `Game`/`State`/`Action`), `kuhn.py`, `leduc.py`, `hulhe.py`
- `src/poker_gnn/graphs` — card graph, betting graph, infoset graph builders
- `src/poker_gnn/models` — infoset encoder + `PokerGNN` policy/value heads
- `src/poker_gnn/solver` — `cfr.py` (tabular CFR), `mccfr.py` (external-sampling CFR for large games), `deep_cfr.py` (Deep CFR + GNN, with a batched sampled-traversal mode)
- `src/poker_gnn/eval` — `exploitability.py` (exact best-response, small games only) and `baseline_eval.py` (chip-EV vs. fixed baselines, for games too big to enumerate)
- `src/poker_gnn/utils/cards.py` — 5/6/7-card hand evaluator
- `configs/` — per-game hyperparameters
- `docs/PLAN.md` — design, milestones, and detailed run history
