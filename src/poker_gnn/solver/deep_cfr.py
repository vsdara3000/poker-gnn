"""Deep CFR with a GNN advantage network (Brown et al., 2019).

Two traversal modes, chosen by `external_sampling`:

- Full-width (default): chance summed exactly, both players' branches
  walked every iteration -- Kuhn's and Leduc's trees are small enough not
  to need sampling, and it keeps this a direct neural analogue of
  `TabularCFR`. Every infoset is enumerated once up front (`_enumerate_
  infosets`) and predicted in one batched forward pass per player per
  iteration (`_predict_all`).
- External sampling (`external_sampling=True`, required for HULHE): same
  algorithm as `solver.mccfr.ExternalSamplingCFR`, just with the GNN
  standing in for the regret table -- only the traverser's own decisions
  branch over every action; the opponent and chance are each sampled once.
  Nothing can be enumerated up front (HULHE's ~10^14 infosets rule that
  out). To still get batched network calls despite that,
  `parallel_traversals` sampled traversals run concurrently as generators
  (`_traverse_sampled`): each pauses at every network query, `_run_batch`
  collects every paused traversal's pending query, does one batched
  forward pass per player, and resumes them all with their strategies.
  (On CPU; the bottleneck is Python-side orchestration, not matmuls --
  see docs/PLAN.md.)

Either way, per player, an advantage network regresses the instantaneous
counterfactual regret observed at each visited infoset; regret matching
over its predicted advantages gives the current-iteration strategy in
place of a tabular regret table. The average strategy is still accumulated
exactly like `TabularCFR` (a running sum keyed by infoset, reach-weighted
in full-width mode; unweighted in sampling mode since the sampling itself
already reflects reach probabilities -- same reasoning as `mccfr.py`).

In `external_sampling` mode there's a second network per player, too: the
running-sum dict above is *exact* but only ever covers the infosets it
actually visited, which is a fine approximation of the average strategy
at Kuhn/Leduc scale (the dict ends up covering the whole game) but breaks
down at HULHE's scale (~10^14 infosets, a few hundred thousand visited
after a real training run -- a vanishing fraction). `policy()` looks the
dict up first and falls back to this strategy network -- trained to
imitate the current `strategy` seen at every opponent-side infoset visit
(the same visits that feed the dict), same as
Brown et al.'s original Deep CFR -- everywhere else, so a strategy built
from this training run actually generalizes past the exact infosets it
visited, rather than silently defaulting to uniform random outside them
the way a pure dict lookup (`average_strategy()`) does.

`range_net=True` (external-sampling mode only) adds a third network per
player that *ranges the opponent*. Traversal deals real cards to both
players, so at every decision a player makes we know what the opponent
actually holds. The range network learns to predict two things from the
acting player's infoset alone: their showdown equity against the
opponent's actual hand (`Game.sample_showdown_equity`, one random board
rollout per example), and the opponent's made-hand strength (`Game.hand_
strength` on the opponent's cards). Since the infoset shows the hero only
their own cards and the public betting, the one way to predict these beyond
the no-information average is to read the opponent's betting pattern. It
is a learned version of Bayesian range estimation: its target is exactly
the posterior under how the opponent currently plays, and it costs one
forward pass instead of a Bayes update over all 1,326 hands. Both
predictions are fed to the advantage and strategy networks as extra
per-graph input features (`PokerGNN`'s `extra`).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from poker_gnn.games.base import Action
from poker_gnn.models.encoder import InfosetEncoder
from poker_gnn.models.gnn import PokerGNN
from poker_gnn.solver.policy_net import fit_policy_step, network_policy_batch, policy_example

NUM_ACTIONS = len(Action)
# Range network outputs: [hero showdown equity, opponent made-hand strength]
RANGE_OUTPUTS = 2


class _ReservoirBuffer:
    """Fixed-capacity uniform sample over every item ever added (Vitter's
    algorithm R), so old iterations stay represented in proportion instead
    of being pushed out by recent ones. Items are (graph, target_regrets)
    for advantage buffers, policy_net.policy_example triples for strategy
    buffers, and (graph, target, mask) for range buffers."""

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
            # Keep the new item with probability capacity / seen, replacing
            # a uniformly chosen slot.
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
        external_sampling: bool = False,
        parallel_traversals: int = 16,
        range_net: bool = False,
    ):
        if range_net and not external_sampling:
            raise ValueError("range_net needs external_sampling=True (it learns from sampled deals)")
        if limit_threads:
            # Kuhn/Leduc graphs are a handful to a couple dozen nodes, and
            # even HULHE's are just 52 card nodes plus the betting path;
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
        self.external_sampling = external_sampling
        self.parallel_traversals = parallel_traversals
        self.range_net = range_net
        self._extra_dim = RANGE_OUTPUTS if range_net else 0
        self._rng = random.Random(seed)

        self._networks: dict[int, PokerGNN] = {}
        self._optimizers: dict[int, torch.optim.Optimizer] = {}
        self._buffers: dict[int, _ReservoirBuffer] = {
            0: _ReservoirBuffer(buffer_capacity, self._rng),
            1: _ReservoirBuffer(buffer_capacity, self._rng),
        }
        self._strategy_sum: dict[str, dict[int, float]] = {}
        # Strategy (average-policy) network per player: only trained/used in
        # external_sampling mode, where _strategy_sum's exact dict can't
        # cover the game -- see module docstring and `policy()`.
        self._policy_networks: dict[int, PokerGNN] = {}
        self._policy_optimizers: dict[int, torch.optim.Optimizer] = {}
        self._policy_buffers: dict[int, _ReservoirBuffer] = {
            0: _ReservoirBuffer(buffer_capacity, self._rng),
            1: _ReservoirBuffer(buffer_capacity, self._rng),
        }
        # Opponent-range network per player (only with range_net=True): see
        # the module docstring.
        self._range_networks: dict[int, PokerGNN] = {}
        self._range_optimizers: dict[int, torch.optim.Optimizer] = {}
        self._range_buffers: dict[int, _ReservoirBuffer] = {
            0: _ReservoirBuffer(buffer_capacity, self._rng),
            1: _ReservoirBuffer(buffer_capacity, self._rng),
        }
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
        """Run `iterations` CFR iterations, refitting the networks as it goes.

        Sampling mode counts one traversal as one iteration, runs them in
        batches of `parallel_traversals` (traverser alternating), and refits
        after each batch. Full-width mode refits both players' advantage
        networks after every full tree pass. Returns self for chaining."""
        if self.external_sampling:
            remaining = iterations
            traverser_counter = 0
            while remaining > 0:
                batch = min(self.parallel_traversals, remaining)
                traversers = [(traverser_counter + i) % 2 for i in range(batch)]
                traverser_counter += batch
                remaining -= batch
                self._run_batch(game, traversers)
                for player in set(traversers):
                    self._fit_range(player)
                    self._fit(player)
                    self._fit_policy(player)
            return self

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
        """Exact normalized `_strategy_sum`, {infoset_key: {action: prob}};
        covers only infosets visited during training (see `policy()`)."""
        result = {}
        for key, sums in self._strategy_sum.items():
            total = sum(sums.values())
            if total > 0:
                result[key] = {a: s / total for a, s in sums.items()}
            else:
                result[key] = {a: 1.0 / len(sums) for a in sums}
        return result

    def save(self, path: str) -> None:
        """Persist the trained networks and the exact `_strategy_sum` dict
        -- enough to call `policy()`/`average_strategy()` or resume training
        later without starting over. Does *not* persist the reservoir
        buffers or optimizer momentum: resuming restarts those from empty/
        fresh, a minor, self-correcting cost (a few thousand iterations'
        worth of buffer refill) against the size and complexity of also
        checkpointing every buffered graph."""
        state = {
            "config": {
                "hidden_dim": self.hidden_dim,
                "lr": self.lr,
                "buffer_capacity": self._buffers[0].capacity,
                "batch_size": self.batch_size,
                "train_steps_per_iteration": self.train_steps_per_iteration,
                "external_sampling": self.external_sampling,
                "parallel_traversals": self.parallel_traversals,
                "node_feature_dim": self.encoder.node_feature_dim,
                "range_net": self.range_net,
            },
            "strategy_sum": self._strategy_sum,
            "networks": {p: net.state_dict() for p, net in self._networks.items()},
            "policy_networks": {p: net.state_dict() for p, net in self._policy_networks.items()},
            "range_networks": {p: net.state_dict() for p, net in self._range_networks.items()},
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "DeepCFR":
        state = torch.load(path, weights_only=False)
        cfg = state["config"]
        solver = cls(
            hidden_dim=cfg["hidden_dim"],
            lr=cfg["lr"],
            buffer_capacity=cfg["buffer_capacity"],
            batch_size=cfg["batch_size"],
            train_steps_per_iteration=cfg["train_steps_per_iteration"],
            seed=seed,
            external_sampling=cfg["external_sampling"],
            parallel_traversals=cfg["parallel_traversals"],
            range_net=cfg.get("range_net", False),
        )
        solver._strategy_sum = state["strategy_sum"]
        for player, state_dict in state["networks"].items():
            solver._network(player).load_state_dict(state_dict)
        for player, state_dict in state["policy_networks"].items():
            solver._policy_network(player).load_state_dict(state_dict)
        for player, state_dict in state.get("range_networks", {}).items():
            solver._range_network(player).load_state_dict(state_dict)
        return solver

    def _network(self, player: int) -> PokerGNN:
        if player not in self._networks:
            net = PokerGNN(self.encoder.node_feature_dim, self.hidden_dim, self._extra_dim)
            self._networks[player] = net
            self._optimizers[player] = torch.optim.Adam(net.parameters(), lr=self.lr)
        return self._networks[player]

    def _policy_network(self, player: int) -> PokerGNN:
        if player not in self._policy_networks:
            net = PokerGNN(self.encoder.node_feature_dim, self.hidden_dim, self._extra_dim)
            self._policy_networks[player] = net
            self._policy_optimizers[player] = torch.optim.Adam(net.parameters(), lr=self.lr)
        return self._policy_networks[player]

    def _range_network(self, player: int) -> PokerGNN:
        if player not in self._range_networks:
            net = PokerGNN(self.encoder.node_feature_dim, self.hidden_dim, num_outputs=RANGE_OUTPUTS)
            self._range_networks[player] = net
            self._range_optimizers[player] = torch.optim.Adam(net.parameters(), lr=self.lr)
        return self._range_networks[player]

    def _range_features(self, player: int, batch):
        """The range network's [equity, opponent strength] predictions for
        `batch`, as extra inputs for `player`'s other networks. None when
        range_net is off."""
        if not self.range_net:
            return None
        net = self._range_network(player)
        net.eval()
        with torch.no_grad():
            out, _ = net(batch)
        return torch.sigmoid(out)

    def estimate_range(self, game, state) -> dict[str, float]:
        """The acting player's estimated showdown equity vs. the opponent's
        range, and the opponent's estimated made-hand strength. Requires
        range_net=True."""
        batch = Batch.from_data_list([self.encoder.encode(game, state, state.player)])
        equity, opp_strength = self._range_features(state.player, batch)[0].tolist()
        return {"equity": equity, "opp_strength": opp_strength}

    def policy(self, game, state) -> dict[int, float]:
        """The strategy to actually play with: `average_strategy()`'s exact
        dict where this infoset was visited during training, falling back
        to the trained strategy network elsewhere (only ever populated in
        `external_sampling` mode). Use this instead of `average_strategy()`
        for actual play on a game too big to have visited every infoset."""
        return self.policy_batch(game, [state])[0]

    def policy_batch(self, game, states) -> list[dict[int, float]]:
        """`policy()` for many states at once, with a single batched
        network forward pass per player for whichever of them miss the
        exact dict (e.g. LBR querying every hand in an opponent range)."""
        result: list = [None] * len(states)
        misses: dict[int, list[int]] = {0: [], 1: []}
        for i, state in enumerate(states):
            legal = game.legal_actions(state)
            sums = self._strategy_sum.get(state.infoset_key(state.player))
            total = sum(sums.values()) if sums is not None else 0.0
            if total > 0:
                result[i] = {a: sums[a] / total for a in legal}
            elif state.player in self._policy_networks:
                misses[state.player].append(i)
            else:
                result[i] = {a: 1.0 / len(legal) for a in legal}
        for player, idxs in misses.items():
            if not idxs:
                continue
            probs = network_policy_batch(
                self._policy_networks[player], self.encoder, game, [states[i] for i in idxs],
                extra_fn=lambda b, p=player: self._range_features(p, b),
            )
            for i, p in zip(idxs, probs):
                result[i] = p
        return result

    def _predict_strategy(self, state, player: int) -> tuple[dict, object]:
        return self._strategy_cache[(player, state.infoset_key(player))]

    def _sample(self, outcomes):
        """Draw one item from a sequence of (item, probability) pairs."""
        items = [item for item, _ in outcomes]
        weights = [weight for _, weight in outcomes]
        return self._rng.choices(items, weights=weights, k=1)[0]

    def _traverse_sampled(self, game, state, traverser: int):
        """External-sampling counterpart to `_traverse`: only `traverser`'s
        own decisions branch over every action; the opponent and chance are
        each sampled once. See `solver.mccfr.ExternalSamplingCFR`, which
        this mirrors -- same algorithm, GNN instead of a regret table.

        A *generator*, not a plain function: whenever it needs a network
        prediction it `yield`s the request instead of computing it inline,
        so `_run_batch` can run many of these concurrently and batch their
        pending requests into one forward pass per player. `yield from`
        makes this transparent across recursive calls -- a `yield` deep in
        the recursion still surfaces to whichever driver is stepping this
        generator, and its `.send(strategy)` reply lands right back there.
        Returns (via a generator's `return`, i.e. `StopIteration.value`)
        the node's value for `traverser`.
        """
        if state.terminal:
            return game.returns(state)[traverser]

        if state.chance:
            outcome = self._sample(game.chance_outcomes(state))
            result = yield from self._traverse_sampled(game, game.step(state, outcome), traverser)
            return result

        player = state.player
        legal = game.legal_actions(state)
        data = self.encoder.encode(game, state, player)
        if self.range_net:
            self._remember_range(game, state, player, data)
        # Pause here; `_run_batch` answers with the regret-matched strategy.
        strategy = yield (player, data, legal)

        if player == traverser:
            action_values = {}
            node_value = 0.0
            for action in legal:
                v = yield from self._traverse_sampled(game, game.step(state, action), traverser)
                action_values[action] = v
                node_value += strategy[action] * v
            # Sampled regret, unweighted (see mccfr.py). Illegal actions
            # keep a 0 target and are never read back.
            target = torch.zeros(NUM_ACTIONS)
            for action in legal:
                target[action] = action_values[action] - node_value
            self._buffers[player].add((data, target))
            return node_value

        key = state.infoset_key(player)
        sums = self._strategy_sum.setdefault(key, {a: 0.0 for a in legal})
        for action in legal:
            sums[action] += strategy[action]
        self._remember_policy(player, data, legal, strategy)
        action = self._sample([(a, strategy[a]) for a in legal])
        result = yield from self._traverse_sampled(game, game.step(state, action), traverser)
        return result

    def _remember_policy(self, player: int, data, legal, strategy: dict) -> None:
        """Add one (infoset, sampled strategy) example to that player's
        strategy-network training buffer -- same role as the `sums[action]
        += ...` line just above, except this one generalizes to infosets
        that never get visited again."""
        self._policy_buffers[player].add(policy_example(data, legal, strategy))

    def _remember_range(self, game, state, player: int, data) -> None:
        """Add one (infoset, what the opponent really held) example to the
        range network's buffer. The targets use the opponent's actual cards,
        which the network's input never contains."""
        equity = game.sample_showdown_equity(state, player, self._rng)
        if equity is None:
            return
        opp = 1 - player
        opp_cards = tuple(c for c in (state.hole_cards[opp], state.hole_cards2[opp]) if c is not None)
        strength = game.hand_strength(opp_cards, state.board)
        target = torch.tensor([equity, strength if strength is not None else 0.0])
        mask = torch.tensor([1.0, 1.0 if strength is not None else 0.0])
        self._range_buffers[player].add((data, target, mask))

    def _run_batch(self, game, traversers: list[int]) -> None:
        """Drive `len(traversers)` `_traverse_sampled` generators concurrently
        in lockstep: each round, collect every generator's pending network
        request, answer all of one player's requests with a single batched
        forward pass, then resume every generator with its own strategy."""
        gens = [self._traverse_sampled(game, game.root(), t) for t in traversers]
        to_send: list = [None] * len(gens)
        active = list(range(len(gens)))

        while active:
            requests = []  # (index, player, data, legal)
            still_active = []
            for idx in active:
                try:
                    item = gens[idx].send(to_send[idx])
                except StopIteration:
                    continue
                requests.append((idx, *item))
                still_active.append(idx)
            active = still_active
            if not requests:
                break

            for player in (0, 1):
                group = [(idx, data, legal) for idx, p, data, legal in requests if p == player]
                if not group:
                    continue
                batch = Batch.from_data_list([data for _, data, _ in group])
                extra = self._range_features(player, batch)
                net = self._network(player)
                net.eval()
                with torch.no_grad():
                    advantages, _ = net(batch, extra)
                # Regret matching over the predicted advantages.
                for (idx, _, legal), adv in zip(group, advantages):
                    positive = {a: max(adv[a].item(), 0.0) for a in legal}
                    total = sum(positive.values())
                    if total > 0:
                        strategy = {a: v / total for a, v in positive.items()}
                    else:
                        strategy = {a: 1.0 / len(legal) for a in legal}
                    to_send[idx] = strategy

    def _traverse(self, game, state, reach0: float, reach1: float) -> tuple[float, float]:
        """Full-width CFR pass, the same recursion as `TabularCFR._cfr`, but
        the strategy comes from `_strategy_cache` and the opponent-reach-
        weighted regrets go into the advantage buffer instead of a table."""
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
        """A few MSE steps of the advantage network on its reservoir.

        Targets are per-visit (instantaneous) regrets, not cumulative ones.
        Since the reservoir is a uniform sample over all iterations, the
        regression converges to their per-infoset mean, which is
        proportional to cumulative regret; regret matching is scale-
        invariant, so that gives the same strategy. Unlike Brown et al.,
        there is no linear-CFR iteration weighting, and the network is
        warm-started rather than retrained from scratch each iteration."""
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
            predicted, _ = net(graphs, self._range_features(player, graphs))
            loss = F.mse_loss(predicted, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _fit_policy(self, player: int) -> None:
        """A few steps of the strategy network toward the buffered strategies
        (their mean per infoset is the average strategy)."""
        buffer = self._policy_buffers[player]
        if not buffer:
            return
        net = self._policy_network(player)
        optimizer = self._policy_optimizers[player]
        for _ in range(self.train_steps_per_iteration):
            fit_policy_step(
                net, optimizer, buffer.sample(self.batch_size),
                extra_fn=lambda b: self._range_features(player, b),
            )

    def _fit_range(self, player: int) -> None:
        """A few MSE steps of the range network on sigmoid outputs; the mask
        drops the opponent-strength target where `hand_strength` is None."""
        buffer = self._range_buffers[player]
        if not self.range_net or not buffer:
            return
        net = self._range_network(player)
        optimizer = self._range_optimizers[player]
        net.train()
        for _ in range(self.train_steps_per_iteration):
            batch_items = buffer.sample(self.batch_size)
            graphs = Batch.from_data_list([g for g, _, _ in batch_items])
            targets = torch.stack([t for _, t, _ in batch_items])
            masks = torch.stack([m for _, _, m in batch_items])
            out, _ = net(graphs)
            loss = ((torch.sigmoid(out) - targets) ** 2 * masks).sum() / masks.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
