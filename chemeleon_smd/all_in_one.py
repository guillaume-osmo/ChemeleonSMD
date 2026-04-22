"""Utilities for fold-masked all-in-one regression ensembles.

This module keeps the blind-fold idea local to ChemeleonSMD:

- train ``K`` experts in parallel
- assign each training sample to one holdout fold
- never apply supervised loss from sample ``i`` to expert ``fold_id[i]``

The matching expert can then be used for blind out-of-fold predictions on the
training split, while a learned or mean ensemble can be applied on validation
and test splits that were unseen by all experts.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def fold_holdout_mask(fold_ids: mx.array, num_experts: int) -> mx.array:
    """Return ``True`` where an expert is allowed to see a sample."""
    if num_experts < 2:
        raise ValueError("All-in-one requires at least two experts.")
    fold_ids = fold_ids.astype(mx.int32).reshape(-1)
    expert_ids = mx.arange(num_experts, dtype=mx.int32)
    return fold_ids[:, None] != expert_ids[None, :]


def normalize_targets_per_expert(
    raw_targets: mx.array,
    target_means: mx.array,
    target_stds: mx.array,
) -> mx.array:
    """Expand raw targets to expert-specific normalized targets."""
    y = raw_targets.reshape(-1, 1)
    means = target_means.reshape(1, -1)
    stds = target_stds.reshape(1, -1)
    return (y - means) / stds


def denormalize_predictions(
    normalized_predictions: mx.array,
    target_means: mx.array,
    target_stds: mx.array,
) -> mx.array:
    """Convert expert-normalized predictions back to raw target units."""
    means = target_means.reshape(1, -1)
    stds = target_stds.reshape(1, -1)
    return normalized_predictions * stds + means


def masked_normalized_mse_loss(
    normalized_predictions: mx.array,
    raw_targets: mx.array,
    fold_ids: mx.array,
    target_means: mx.array,
    target_stds: mx.array,
) -> mx.array:
    """MSE over all non-held-out expert/sample pairs."""
    target_matrix = normalize_targets_per_expert(
        raw_targets,
        target_means=target_means,
        target_stds=target_stds,
    )
    weights = fold_holdout_mask(fold_ids, normalized_predictions.shape[1]).astype(
        normalized_predictions.dtype
    )
    sqerr = (normalized_predictions - target_matrix) ** 2
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=normalized_predictions.dtype))
    return mx.sum(sqerr * weights) / denom


def blind_holdout_predictions(
    raw_expert_predictions: mx.array,
    fold_ids: mx.array,
) -> mx.array:
    """Pick the matching blind expert for each sample."""
    pick = (~fold_holdout_mask(fold_ids, raw_expert_predictions.shape[1])).astype(
        raw_expert_predictions.dtype
    )
    return mx.sum(raw_expert_predictions * pick, axis=1)


def mean_ensemble_prediction(raw_expert_predictions: mx.array) -> mx.array:
    """Simple test-time mean ensemble."""
    return mx.mean(raw_expert_predictions, axis=1)


class GlobalSoftmaxBlend(nn.Module):
    """A tiny learned ensemble over expert predictions.

    The weights are global rather than sample-dependent, which keeps the blend
    stable on small validation sets while still allowing non-uniform expert
    weighting on test-time predictions.
    """

    def __init__(self, num_experts: int):
        super().__init__()
        if num_experts < 2:
            raise ValueError("GlobalSoftmaxBlend requires at least two experts.")
        self.logits = mx.zeros((num_experts,))

    def weights(self) -> mx.array:
        return mx.softmax(self.logits, axis=0)

    def __call__(self, raw_expert_predictions: mx.array) -> mx.array:
        weights = self.weights()
        return mx.sum(raw_expert_predictions * weights[None, :], axis=1)
