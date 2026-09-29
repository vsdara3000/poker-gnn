"""Give a dict-only solver's average strategy the same generalizing
network fallback `DeepCFR.policy()` has.

`ExternalSamplingCFR.average_strategy()` is an exact dict that, on HULHE,
covers only about half the decision points a fresh random deal reaches, and
`baseline_eval.make_strategy_policy` plays uniform random at the rest. So
comparing mccfr to deep_cfr by baseline eval mostly measured that coverage
gap, not what either one learned (see docs/PLAN.md). `DistilledPolicy` fixes
this after training: play self-play hands with the dict strategy, record
(infoset graph, average strategy) at every decision the dict covers
(optionally only ones visited at least `min_visits` times), and fit a
strategy network to them. `policy()` then works like `DeepCFR.policy()`:
exact dict lookup where the dict is trusted, and the network everywhere else.
"""

from __future__ import annotations

import random

import torch

from poker_gnn.models.encoder import InfosetEncoder
from poker_gnn.models.gnn import PokerGNN
from poker_gnn.solver.policy_net import fit_policy_step, network_policy_batch, policy_example


class DistilledPolicy:
    def __init__(self, strategy: dict, networks: dict[int, PokerGNN], encoder: InfosetEncoder):
        self.strategy = strategy
        self.networks = networks
        self.encoder = encoder
        self.num_examples = 0

    @classmethod
    def fit(
        cls,
        game,
        strategy: dict,
        rng: random.Random,
        visits: dict[str, float] | None = None,
        min_visits: float = 1.0,
        num_hands: int = 2000,
        hidden_dim: int = 32,
        lr: float = 5e-3,
        batch_size: int = 128,
        train_steps: int = 500,
        limit_threads: bool = True,
    ) -> "DistilledPolicy":
        if limit_threads:
            torch.set_num_threads(1)  # same reason as DeepCFR: tiny graphs
        encoder = InfosetEncoder()
        examples: dict[int, list] = {0: [], 1: []}
        for _ in range(num_hands):
            state = game.root()
            while not state.terminal:
                if state.chance:
                    outcomes = game.chance_outcomes(state)
                    state = game.step(state, rng.choices(
                        [o for o, _ in outcomes], weights=[w for _, w in outcomes], k=1)[0])
                    continue
                player = state.player
                legal = game.legal_actions(state)
                key = state.infoset_key(player)
                probs = strategy.get(key)
                if probs is not None and (visits is None or visits.get(key, 0.0) >= min_visits):
                    data = encoder.encode(game, state, player)
                    examples[player].append(policy_example(data, legal, probs))
                if probs is None:
                    action = rng.choice(legal)
                else:
                    action = rng.choices(legal, weights=[probs.get(a, 0.0) for a in legal], k=1)[0]
                state = game.step(state, action)

        networks = {}
        for player, items in examples.items():
            if not items:
                continue
            net = PokerGNN(encoder.node_feature_dim, hidden_dim)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            for _ in range(train_steps):
                fit_policy_step(net, optimizer, rng.sample(items, min(batch_size, len(items))))
            networks[player] = net
        # Lookups skip the same under-visited entries training skipped, so
        # they get the network's (smoothed) answer instead of a noisy one.
        trusted = strategy if visits is None else {
            k: v for k, v in strategy.items() if visits.get(k, 0.0) >= min_visits
        }
        result = cls(trusted, networks, encoder)
        result.num_examples = sum(len(v) for v in examples.values())
        return result

    def policy(self, game, state) -> dict[int, float]:
        return self.policy_batch(game, [state])[0]

    def policy_batch(self, game, states) -> list[dict[int, float]]:
        result: list = [None] * len(states)
        misses: dict[int, list[int]] = {0: [], 1: []}
        for i, state in enumerate(states):
            legal = game.legal_actions(state)
            probs = self.strategy.get(state.infoset_key(state.player))
            if probs is not None:
                result[i] = {a: probs.get(a, 0.0) for a in legal}
            elif state.player in self.networks:
                misses[state.player].append(i)
            else:
                result[i] = {a: 1.0 / len(legal) for a in legal}
        for player, idxs in misses.items():
            if idxs:
                probs = network_policy_batch(
                    self.networks[player], self.encoder, game, [states[i] for i in idxs]
                )
                for i, p in zip(idxs, probs):
                    result[i] = p
        return result
