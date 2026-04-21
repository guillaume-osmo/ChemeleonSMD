"""CheMeleon-SCORE: DMPNN + Set2Set + SCORE on Mordred descriptors."""

from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.utils import scatter

from chemeleon_smd import layers
from chemeleon_smd import mpnn


class CheMeleonSCORE(nn.Module):
    """CheMeleon with SCORE architecture for masked descriptor prediction."""

    def __init__(
        self,
        d_v: int = 72,
        d_e: int = 14,
        d_h: int = 2048,
        depth: int = 6,
        readout: Literal["mean", "set2set"] = "set2set",
        set2set_iters: int = 6,
        score_dim: int = 2048,
        score_steps: int = 6,
        skip_alpha: float = 0.5,
        n_descriptors: int = 1613,
        masking_ratio: float = 0.15,
        winsorization_factor: float = 6.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.n_descriptors = n_descriptors
        self.score_dim = score_dim

        self.bond_mpnn = mpnn.CheMeleonBondMPNN(
            d_v=d_v, d_e=d_e, d_h=d_h, depth=depth, dropout=dropout
        )

        if readout == "set2set":
            self.set2set = mpnn.Set2SetReadout(d_h, n_iters=set2set_iters)
            readout_dim = 2 * d_h
        else:
            self.set2set = None
            readout_dim = d_h

        self.score_encoder = layers.ScoreEncoder(
            input_dim=readout_dim,
            score_dim=score_dim,
            score_steps=score_steps,
            skip_alpha=skip_alpha,
            dropout=dropout,
        )
        self.decoder = nn.Linear(score_dim, n_descriptors)
        self.winsorize = layers.WinsorizeStdevN(winsorization_factor)

        self.feature_means = mx.zeros((n_descriptors,))
        self.feature_vars = mx.ones((n_descriptors,))

    def set_feature_stats(self, means: mx.array, vars: mx.array):
        self.feature_means = means
        self.feature_vars = vars

    def _scale_targets(self, targets: mx.array) -> mx.array:
        safe_std = mx.sqrt(mx.maximum(self.feature_vars, mx.array(1e-8)))
        scaled = (targets - self.feature_means) / safe_std
        return self.winsorize(scaled)

    def fingerprint(
        self,
        V: mx.array,
        E: mx.array,
        edge_index: mx.array,
        rev_edge_index: mx.array,
        batch: mx.array,
        num_graphs: int,
    ) -> mx.array:
        h_v = self.bond_mpnn(V, E, edge_index, rev_edge_index)
        if self.set2set is not None:
            fp = self.set2set(h_v, batch, num_graphs)
        else:
            fp = scatter(h_v, batch, out_size=num_graphs, aggr="mean")
        return self.score_encoder(fp)

    def __call__(
        self,
        V: mx.array,
        E: mx.array,
        edge_index: mx.array,
        rev_edge_index: mx.array,
        batch: mx.array,
        num_graphs: int,
    ) -> mx.array:
        fp = self.fingerprint(V, E, edge_index, rev_edge_index, batch, num_graphs)
        return self.decoder(fp)


def masked_mse_loss(
    preds: mx.array,
    targets: mx.array,
    masking_ratio: float,
) -> mx.array:
    """Masked MSE: random mask, loss only on masked positions."""
    mask = mx.random.uniform(shape=targets.shape) < masking_ratio
    diff = (preds - targets) ** 2
    masked_diff = mx.where(mask, diff, mx.zeros_like(diff))
    n_masked = mx.maximum(mx.sum(mask), mx.array(1.0))
    return mx.sum(masked_diff) / n_masked


def full_mse_loss(preds: mx.array, targets: mx.array) -> mx.array:
    """Full MSE for validation."""
    return mx.mean((preds - targets) ** 2)
