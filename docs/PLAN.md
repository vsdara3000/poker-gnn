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

### Status (in progress, paused here)

Built and tested so far:

- `utils/cards.py` — 5/6/7-card hand evaluator (`evaluate_hand`), validated both by unit tests and by sampling 200k random 5-card hands and matching known category odds (`tests/test_cards.py`).
- `games/hulhe.py` — full rules: 2 hole cards/player, 4 rounds (preflop/flop/turn/river), blinds + the big-blind option, small/big bet sizing, a 4-raise cap per round. `base.State` gained `hole_cards2` (second hole card) and `round_start` was already there from Leduc. Validated via scripted lines (blind option, raise cap, fold payoffs, a real quads-vs-full-house showdown) plus 20k random Monte Carlo playouts checked for zero-sum / no negative stacks (`tests/test_hulhe.py`). The full tree is too big to enumerate (~10^14 infosets), so unlike Kuhn/Leduc there's no exhaustive tree test here.
- `solver/mccfr.py` (`ExternalSamplingCFR`) — tabular CFR replacement for games too big for full-width traversal: only the traverser's own decisions branch over every action; opponent + chance are sampled once. Validated against Kuhn's known closed-form Nash mix (`tests/test_mccfr.py`) before trusting it on HULHE.
- `solver/deep_cfr.py` gained `external_sampling=True` mode — same sampling algorithm, GNN instead of a regret table. Validated on Kuhn (`tests/test_deep_cfr.py::test_sampled_deep_cfr_*`).
- `graphs/infoset_graph.py` fixed to flag *both* hero hole cards (was only flagging one — a real gap for any 2-card-hand game, silent before HULHE existed to expose it).
- `eval/baseline_eval.py` — since exact best-response (`exploitability()`) also requires full enumeration and can't run on HULHE, this instead plays a strategy against fixed baselines (always-fold, always-call, uniform-random) over many simulated hands and reports average chip EV. **This is not a Nash-distance metric** — beating the baselines shows the strategy learned *something*, not how close it is to equilibrium. Validated against Kuhn (`tests/test_baseline_eval.py`).

Run so far (`DeepCFR(external_sampling=True)` directly on HULHE, not yet committed as a script):

- Timing: ~0.25-0.4s/iteration (varies run to run). 500 iterations took 125s and touched 49,427 distinct information sets.
- Baseline-eval @ 500 iterations, default hyperparameters (hidden_dim=32, buffer_capacity=4000, train_steps_per_iteration=4): crushes always-fold (+0.96 as P0, +1.00 as P1), but **loses to always-call** (-0.213 as P0, -0.086 as P1) and is mixed/losing to uniform-random as P0 (-0.543; +0.625 as P1). Reads as "pipeline works end-to-end on the real game" rather than "strategy is any good" — expected at this scale (10^14 infosets, only 500 iterations).
- Tried a "bigger" config (hidden_dim=64, buffer_capacity=20000, train_steps_per_iteration=8) at 300 iterations each, single seed: modestly better on 3/4 metrics vs the 32-unit default (e.g. vs-call as P0: -0.414 vs -0.607), worse on 1, small effect size, not a rigorous comparison (one seed, short run) -- inconclusive but no clear downside either.
- Re-ran the same 5,000-iteration, "bigger"-config comparison *after* the architecture changes below (strategy network + `make_solver_policy`, masked pooling, hand-strength feature), same seed. Dramatic difference:

  | iterations | vs_call (P0/P1) before | vs_call (P0/P1) after | vs_random (P0/P1) before | vs_random (P0/P1) after |
  |---|---|---|---|---|
  | 1,000 | -0.651 / +0.043 | +0.261 / -0.083 | -0.655 / +0.464 | +3.684 / +1.872 |
  | 2,000 | -0.560 / -0.024 | -0.019 / +0.288 | -0.500 / +0.470 | +3.160 / +1.762 |
  | 3,000 | -0.647 / -0.206 | +0.372 / +0.329 | -0.386 / +0.411 | +3.167 / +0.922 |
  | 4,000 | -0.427 / -0.163 | +0.247 / +0.383 | -0.358 / +0.398 | +2.967 / +1.552 |
  | 5,000 | -0.610 / +0.011 | +0.638 / +0.036 | -0.349 / +0.259 | +2.942 / +1.552 |

  Now mostly *beats* always-call (the metric that mattered most, and was solidly negative before) instead of losing to it, and crushes random by 3+ chips/hand instead of barely positive. Best guess at attribution (not isolated via an ablation run): probably mostly the strategy network -- only ~490k of ~10^14 infosets were ever exactly visited, so a fresh random-deal evaluation hand almost never lands on one, meaning the *old* numbers were mostly measuring uniform-random fallback behavior, not what had actually been learned.
- Not yet locked in a config or run long enough to call phase 3 "done" by any bar -- this is still an early, promising result, not a finished solve.

GPU / parallelism -- done (batching), CUDA install skipped:

- There's an NVIDIA RTX 4050 (6GB) visible from WSL, but the installed torch is still CPU-only (`torch==2.14.0+cpu`).
- Restructured `DeepCFR`'s sampled traversal (`_traverse_sampled`) into a *generator*: it now `yield`s a `(player, graph, legal_actions)` request instead of computing the prediction inline, pausing there. `_run_batch` drives `parallel_traversals` (default 16) of these concurrently via `.send()`, collecting every paused traversal's pending request each round and answering all of one player's requests with a single batched forward pass. `yield from` makes this transparent across the recursive tree calls.
- Measured on HULHE directly (200 iterations each, `parallel_traversals` swept): K=1 -> 429ms/iter, K=16 -> **245ms/iter (1.75x)**, K=32 -> 311ms/iter, K=64 -> 280ms/iter. Not monotonic -- 16 is the sweet spot; past that, the Python-level cost of juggling more concurrent generators outweighs the benefit of a bigger matmul, since these graphs/networks are small. This is also why we're skipping the CUDA install: the bottleneck is CPU-side orchestration, not floating-point throughput, so a GPU wouldn't address what's actually slow. Validated correctness against Kuhn (`tests/test_deep_cfr.py::test_sampled_deep_cfr_*` still pass unchanged) before trusting it on HULHE.

Architecture changes (done, from a "what else besides more iterations" discussion):

1. **Average-strategy network**, not just a growing dict. `DeepCFR`'s exact `_strategy_sum` dict only ever covers infosets actually visited (a few hundred thousand out of ~10^14 after a real run) -- `baseline_eval`'s old `make_strategy_policy` fell back to *uniform random* for anything else, meaning most of a random-deal evaluation was silently measuring untrained behavior, not the network's real quality. Added a second network per player (`_policy_networks`, trained via `_fit_policy` to imitate the sampled `strategy` seen at every infoset visited, matching Brown et al.'s original Deep CFR), and a hybrid `DeepCFR.policy(game, state)` method: exact dict lookup where visited, network fallback otherwise. `baseline_eval.make_solver_policy(solver, rng)` wraps this (only in `external_sampling` mode -- full-width mode's dict already covers Kuhn/Leduc exactly, so this network isn't trained there). Verified directly: trained a solver for only 20 iterations on Kuhn (so several of its 12 infosets are never visited), confirmed `policy()` returns a real, non-uniform, correctly-normalized distribution for those instead of the old uniform fallback (`tests/test_deep_cfr.py::test_policy_falls_back_to_the_strategy_network_for_unvisited_infosets`).
2. **Fixed pooling dilution + added an explicit hand-strength feature.** `PokerGNN` used to mean-pool over *every* node -- fine at Kuhn/Leduc scale, but at HULHE's 52-card graph as few as 7 of 52 card nodes are ever informative (2 hole + 5 board); the other ~45 are identical, contentless placeholders that were washing out the signal that matters. Added `Data.node_role` (per-node: other-card / informative-card / action) so `PokerGNN` now pools informative-card and action nodes separately and concatenates them, instead of one diluted mean. Also added `Game.hand_strength(hole_cards, board)` (base.py, default None; HULHE overrides it via `evaluate_hand`, normalized hand-category to [0,1]) as an explicit feature concatenated onto the pooled representation -- gives the network a shortcut on flush/straight/full-house recognition instead of expecting it to rediscover poker hand rankings from raw graph structure with this little training data. Verified the hand-strength values directly against known hands (quads vs. two pair on a real board) and that PyG batches the new per-graph attributes correctly.

Not implemented: explicit opponent range/belief-state estimation (Bayesian, DeepStack-style) -- discussed as a further idea, judged bigger than the above two and deferred pending whether they move the needle enough on their own.

`scripts/train.py` -- done. Was a stub; now a generic CLI over any game/solver combo: `--solver deep_cfr` (GNN, external-sampling on by default for hulhe, off for kuhn/leduc, overridable via `--external-sampling`/`--no-external-sampling`) or `--solver mccfr` (plain `ExternalSamplingCFR`, no network -- the cheap tabular baseline on HULHE that was missing). Reports exact `exploitability()` at each checkpoint for kuhn/leduc, `baseline_eval` chip EV vs. fold/call/random for hulhe (or for kuhn/leduc run in external-sampling mode). Smoke-tested on all four combinations (kuhn/leduc full-width, hulhe mccfr, hulhe deep_cfr external-sampling); full test suite (77 tests) unaffected.

Other not done yet:

- Haven't actually *used* the new `--solver mccfr` HULHE path for anything beyond the smoke test above -- next: a real run to see how tabular external-sampling CFR compares to Deep CFR per-iteration and per-wall-clock-second on HULHE.
- No exploitability trend or a "done when" call for phase 3 -- only baseline-eval numbers so far, and only at small iteration counts.
- No multi-seed hyperparameter comparison yet for the "bigger" Deep CFR config vs. default -- `scripts/train.py` now makes this easy to script, just hasn't been run.

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
