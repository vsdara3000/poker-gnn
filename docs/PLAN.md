# Poker GNN solver — plan

**Current focus: Kuhn only.** Do not implement Leduc or HULHE until Kuhn is solved both tabularly and with the GNN.

## Difficulty ladder

| Phase | Game | Size | Start when |
|-------|------|------|------------|
| **1 (now)** | Kuhn | 3 cards, 1 betting round, ~12 infosets | Immediately |
| **2** | Leduc | 6 cards, 2 rounds, flop, pairs | Kuhn CFR ≈ Nash and GNN is clearly better than random |
| **3** | HULHE | 52 cards, 4 streets | Leduc GNN is in the same ballpark as tabular CFR |

Kuhn is the easiest real poker-like game. Use it to get the whole pipeline working (rules → CFR → graphs → GNN → exploitability) on something you can check by hand.

## Phase 1 — Kuhn

Rules in one sentence: two players, deck `{J, Q, K}`, each gets one card, ante 1, one round of check/bet/call/fold.

Work in this order, still only on Kuhn:

1. Fill in `games/kuhn.py` (`root`, `legal_actions`, `chance_outcomes`, `step`, `returns`).
2. Fill in `solver/cfr.py` + `eval/exploitability.py`. Average strategy should match the known Kuhn Nash mix; exploitability → 0.
3. Fill in graphs + `InfosetEncoder` for Kuhn infosets only.
4. Fill in `PokerGNN` + `DeepCFR`. Train on Kuhn until exploitability is much better than random and close-ish to tabular.

Leave `leduc.py` and `hulhe.py` as stubs until this phase is done.

## Phase 2 — Leduc

Same interfaces, bigger tree: two betting rounds and a public card. Reuse the Kuhn encoder/GNN; only the game + graph features should grow.

1. Implement `games/leduc.py`.
2. Tabular CFR until exploitability trends down.
3. Train the same GNN/Deep CFR loop on Leduc.

## Phase 3 — HULHE

Same idea at hold'em scale. Do not try a full Cepheus-style solve. Abstraction, sampling, and compute come after Leduc works.

## Shared pieces (keep game-agnostic)

- `games/base.py` — `Game` / `State` / actions
- graph builders — card nodes, betting path, infoset merge
- GNN heads — policy ± value
- CFR vs Deep CFR loop
- exploitability

If something only works for Kuhn, that is fine in phase 1. Generalize when you move to Leduc.

## Graph sketch (fill in during phase 1)

- **Card graph**: nodes for the three Kuhn ranks; later, suits/board for Leduc.
- **Betting graph**: public action sequence as a path.
- **Infoset graph**: hero card + public history; never the opponent’s hole card.

## Folder map

- `src/poker_gnn/games/` — rules only. Implement Kuhn first.
- `src/poker_gnn/graphs/` — state → nodes/edges
- `src/poker_gnn/models/` — encoder + GNN
- `src/poker_gnn/solver/` — tabular CFR, then Deep CFR
- `src/poker_gnn/eval/` — exploitability
- `configs/kuhn.yaml` — the only config that matters until phase 2
- `tests/test_kuhn.py` / `test_cfr.py` — first tests to make real

## Done when (per phase)

- **Kuhn tabular:** exploitability ≈ 0; strategies look like published Nash.
- **Kuhn GNN:** not exact Nash, but far from random.
- **Leduc:** same two checks, then consider HULHE.
