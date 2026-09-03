# Poker GNN Solver

Research codebase for a graph-neural poker solver. **Start with Kuhn**, then Leduc, then heads-up limit hold'em.

See [docs/PLAN.md](docs/PLAN.md) for the full roadmap.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python scripts/solve_tabular.py --game kuhn --iterations 20000
```

## Layout

- `src/poker_gnn/games` — Kuhn, Leduc, HULHE skeleton
- `src/poker_gnn/graphs` — infoset / card / betting graphs
- `src/poker_gnn/models` — GNN policy and value heads
- `src/poker_gnn/solver` — tabular CFR and Deep CFR + GNN
- `src/poker_gnn/eval` — exploitability
- `configs/` — per-game hyperparameters
- `docs/PLAN.md` — design and milestones
