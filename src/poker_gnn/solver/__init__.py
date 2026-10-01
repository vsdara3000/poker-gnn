"""Solvers: `TabularCFR` (vanilla CFR, exact on Kuhn/Leduc), `DeepCFR` (GNN
function approximation, full-width or external-sampling), plus
`mccfr.ExternalSamplingCFR` (tabular, sampled) and `distill.DistilledPolicy`
(post-hoc strategy network for a dict-only solver), imported from their
modules directly."""

from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.solver.deep_cfr import DeepCFR
