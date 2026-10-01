"""Solvers: `TabularCFR` (vanilla CFR, exact on Kuhn/Leduc), `DeepCFR` (GNN
function approximation, full-width or external-sampling), plus
`mccfr.ExternalSamplingCFR` (tabular, sampled) and `distill.DistilledPolicy`
(post-hoc strategy network for a dict-only solver)."""

from poker_gnn.solver.cfr import TabularCFR
from poker_gnn.solver.deep_cfr import DeepCFR
from poker_gnn.solver.distill import DistilledPolicy
from poker_gnn.solver.mccfr import ExternalSamplingCFR
