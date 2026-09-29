"""Shared helpers for strategy (average-policy) networks: batched lookup
and one fitting step, used by `DeepCFR` and by `distill.DistilledPolicy`
(the same kind of network, fit post hoc to a dict-only solver's average
strategy instead of during training)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from poker_gnn.games.base import Action

NUM_ACTIONS = len(Action)


def network_policy_batch(net, encoder, game, states, extra_fn=None) -> list[dict[int, float]]:
    """Softmax-over-legal-actions strategy for each of `states` (each at its
    own acting player), in one batched forward pass. `extra_fn`, if given,
    maps the batch to extra per-graph input features (see `PokerGNN`)."""
    if not states:
        return []
    graphs = [encoder.encode(game, s, s.player) for s in states]
    batch = Batch.from_data_list(graphs)
    net.eval()
    with torch.no_grad():
        extra = extra_fn(batch) if extra_fn is not None else None
        logits, _ = net(batch, extra)
    result = []
    for data, row in zip(graphs, logits):
        legal = list(data.legal_actions)
        probs = F.softmax(row[legal], dim=0)
        result.append({a: p.item() for a, p in zip(legal, probs)})
    return result


def policy_example(data, legal, strategy: dict):
    """One (graph, target, legal-mask) training example for a strategy net."""
    target = torch.zeros(NUM_ACTIONS)
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for action in legal:
        target[action] = strategy[action]
        mask[action] = True
    return (data, target, mask)


def fit_policy_step(net, optimizer, batch_items, extra_fn=None) -> float:
    """One gradient step of masked-softmax MSE toward the target strategies."""
    graphs = Batch.from_data_list([g for g, _, _ in batch_items])
    targets = torch.stack([t for _, t, _ in batch_items])
    legal_masks = torch.stack([m for _, _, m in batch_items])
    net.train()
    extra = None
    if extra_fn is not None:
        with torch.no_grad():
            extra = extra_fn(graphs)
    logits, _ = net(graphs, extra)
    # illegal actions get -inf logit so softmax puts ~0 mass there,
    # matching the 0s already in `targets` at those positions
    masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
    probs = F.softmax(masked_logits, dim=1)
    loss = F.mse_loss(probs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
