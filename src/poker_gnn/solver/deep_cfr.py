"""Deep CFR with a GNN advantage network (Brown et al., 2019).

Full-width tree traversal (chance summed exactly, both players' branches
walked every iteration) rather than external sampling — Kuhn's and Leduc's
trees are small enough not to need it, and it keeps this a direct neural
analogue of `TabularCFR`. Per player, an advantage network regresses the
instantaneous counterfactual regret observed at each visited infoset;
regret matching over its predicted advantages gives the current-iteration
strategy in place of a tabular regret table. The average strategy is still
accumulated exactly like `TabularCFR` (a reach-weighted running sum keyed
by infoset), since a few hundred/thousand infosets make a second averaging
network unnecessary. Full-width traversal is the bottleneck at Leduc's
scale (~1000 infosets, thousands of tree nodes visited per iteration) --
HULHE's much larger tree will need external sampling instead.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from poker_gnn.games.base import Action
from poker_gnn.models.encoder import InfosetEncoder
from poker_gnn.models.gnn import PokerGNN

NUM_ACTIONS = len(Action)


class _ReservoirBuffer:
    """Fixed-capacity reservoir sample of (graph, target_regrets) pairs."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self._rng = rng
        self.items: list = []
        self._seen = 0

    def add(self, item) -> None:
        self._seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self._rng.randint(0, self._seen - 1)
            if j < self.capacity:
                self.items[j] = item

    def sample(self, batch_size: int) -> list:
        if not self.items:
            return []
        return self._rng.sample(self.items, min(batch_size, len(self.items)))

    def __len__(self) -> int:
        return len(self.items)


class DeepCFR:
    """Deep CFR over any two-player zero-sum `Game`, driven by `PokerGNN`."""

    def __init__(
        self,
        encoder: InfosetEncoder | None = None,
        hidden_dim: int = 32,
        lr: float = 5e-3,
        buffer_capacity: int = 4000,
        batch_size: int = 128,
        train_steps_per_iteration: int = 4,
        seed: int | None = None,
        limit_threads: bool = True,
    ):
        if limit_threads:
            # Kuhn/Leduc-sized graphs are a handful to a couple dozen nodes;
            # torch's intra-op thread pool spends far more time coordinating
            # than computing, so single-threaded is an order of magnitude
            # faster here.
            torch.set_num_threads(1)
        if seed is not None:
            # `seed` only covers `self._rng` (reservoir sampling) unless we
            # also seed torch's own RNG here -- network weight init draws
            # from that instead, so runs were silently non-reproducible.
            torch.manual_seed(seed)
        self.encoder = encoder or InfosetEncoder()
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.train_steps_per_iteration = train_steps_per_iteration
        self._rng = random.Random(seed)

        self._networks: dict[int, PokerGNN] = {}
        self._optimizers: dict[int, torch.optim.Optimizer] = {}
        self._buffers: dict[int, _ReservoirBuffer] = {
            0: _ReservoirBuffer(buffer_capacity, self._rng),
            1: _ReservoirBuffer(buffer_capacity, self._rng),
        }
        self._strategy_sum: dict[str, dict[int, float]] = {}
        # Every (player, infoset_key) this game can reach, mapped to its
        # encoded graph -- computed once per game (see `_enumerate_infosets`),
        # not per iteration: full-width traversal visits every infoset every
        # iteration regardless of strategy, so the *set* of infosets is
        # static and only the network's predictions over them change.
        self._infosets: dict[int, dict[str, object]] | None = None
        # Per-iteration cache of (player, infoset_key) -> (strategy, encoded
        # graph), filled by one batched forward pass per player per
        # iteration (`_predict_all`) instead of one GNN call per tree node.
        # Full-width CFR calls this at *every* visited state, and the same
        # infoset is reached from many different tree branches (worse for
        # Leduc's ~1000 infosets than Kuhn's dozen), so batching turns
        # thousands of single-graph forward passes into two per iteration.
        self._strategy_cache: dict[tuple[int, str], tuple[dict, object]] = {}

    def train(self, game, iterations: int):
        if self._infosets is None:
            self._infosets = self._enumerate_infosets(game)
        for _ in range(iterations):
            self._strategy_cache = self._predict_all()
            self._traverse(game, game.root(), reach0=1.0, reach1=1.0)
            for player in (0, 1):
                self._fit(player)
        return self

    def _enumerate_infosets(self, game) -> dict[int, dict[str, object]]:
        """Walk the whole game tree once, encoding every (player, infoset_
        key) it can reach. Safe to do once and reuse across all iterations:
        the encoding of an infoset depends only on the game's structure
        (own card, board, public history), never on network weights."""
        encoded: dict[int, dict[str, object]] = {0: {}, 1: {}}

        def walk(state) -> None:
            if state.terminal:
                return
            if state.chance:
                for outcome, _ in game.chance_outcomes(state):
                    walk(game.step(state, outcome))
                return
            player = state.player
            key = state.infoset_key(player)
            if key not in encoded[player]:
                encoded[player][key] = self.encoder.encode(game, state, player)
            for action in game.legal_actions(state):
                walk(game.step(state, action))

        walk(game.root())
        return encoded

    def _predict_all(self) -> dict[tuple[int, str], tuple[dict, object]]:
        """One batched GNN forward pass per player over every infoset that
        player can act at, replacing thousands of individual per-node calls
        during traversal with two batched ones."""
        cache: dict[tuple[int, str], tuple[dict, object]] = {}
        for player in (0, 1):
            infosets = self._infosets[player]
            if not infosets:
                continue
            keys = list(infosets.keys())
            graphs = [infosets[key] for key in keys]
            net = self._network(player)
            net.eval()
            with torch.no_grad():
                advantages, _ = net(Batch.from_data_list(graphs))
            for key, data, adv in zip(keys, graphs, advantages):
                legal = data.legal_actions
                positive = {a: max(adv[a].item(), 0.0) for a in legal}
                total = sum(positive.values())
                if total > 0:
                    strategy = {a: v / total for a, v in positive.items()}
                else:
                    strategy = {a: 1.0 / len(legal) for a in legal}
                cache[(player, key)] = (strategy, data)
        return cache

    def average_strategy(self) -> dict:
        result = {}
        for key, sums in self._strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                result[key] = {a: s / total for a, s in sums.items()}
            else:
                result[key] = {a: 1.0 / len(sums) for a in sums}
        return result

    def _network(self, player: int) -> PokerGNN:
        if player not in self._networks:
            net = PokerGNN(self.encoder.node_feature_dim, self.hidden_dim)
            self._networks[player] = net
            self._optimizers[player] = torch.optim.Adam(net.parameters(), lr=self.lr)
        return self._networks[player]

    def _predict_strategy(self, state, player: int) -> tuple[dict, object]:
        return self._strategy_cache[(player, state.infoset_key(player))]

    def _traverse(self, game, state, reach0: float, reach1: float) -> tuple[float, float]:
        if state.terminal:
            return game.returns(state)

        if state.chance:
            value0 = value1 = 0.0
            for outcome, prob in game.chance_outcomes(state):
                u0, u1 = self._traverse(game, game.step(state, outcome), reach0, reach1)
                value0 += prob * u0
                value1 += prob * u1
            return (value0, value1)

        player = state.player
        legal = game.legal_actions(state)
        strategy, data = self._predict_strategy(state, player)

        action_values0: dict = {}
        action_values1: dict = {}
        node_value0 = node_value1 = 0.0
        for action in legal:
            child = game.step(state, action)
            if player == 0:
                u0, u1 = self._traverse(game, child, reach0 * strategy[action], reach1)
            else:
                u0, u1 = self._traverse(game, child, reach0, reach1 * strategy[action])
            action_values0[action] = u0
            action_values1[action] = u1
            node_value0 += strategy[action] * u0
            node_value1 += strategy[action] * u1

        opponent_reach = reach1 if player == 0 else reach0
        own_reach = reach0 if player == 0 else reach1
        node_value = node_value0 if player == 0 else node_value1
        action_values = action_values0 if player == 0 else action_values1

        target = torch.zeros(NUM_ACTIONS)
        for action in legal:
            target[action] = opponent_reach * (action_values[action] - node_value)
        self._buffers[player].add((data, target))

        key = state.infoset_key(player)
        sums = self._strategy_sum.setdefault(key, {a: 0.0 for a in legal})
        for action in legal:
            sums[action] += own_reach * strategy[action]

        return (node_value0, node_value1)

    def _fit(self, player: int) -> None:
        buffer = self._buffers[player]
        if not buffer:
            return
        net = self._network(player)
        optimizer = self._optimizers[player]
        net.train()
        for _ in range(self.train_steps_per_iteration):
            batch_items = buffer.sample(self.batch_size)
            graphs = Batch.from_data_list([g for g, _ in batch_items])
            targets = torch.stack([t for _, t in batch_items])
            predicted, _ = net(graphs)
            loss = F.mse_loss(predicted, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
