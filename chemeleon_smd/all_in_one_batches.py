"""Lightweight batch helpers for ChemeleonSMD-AllinOne.

This module intentionally stays free of the heavier training imports so unit
tests can validate fold-aware batch routing without importing the full
benchmark stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from chemeleon_smd import graph_cache


def _mx_int32_array(x) -> mx.array:
    """Convert NumPy- or MLX-backed arrays to MLX int32 arrays."""
    if isinstance(x, mx.array):
        return x if x.dtype == mx.int32 else x.astype(mx.int32)
    return mx.array(np.asarray(x, dtype=np.int32))


def _index_values(values, graph_indices_np: np.ndarray, graph_indices_mx: mx.array) -> mx.array:
    """Index either NumPy- or MLX-backed arrays without an unnecessary bounce."""
    if isinstance(values, mx.array):
        return values[graph_indices_mx]
    return mx.array(np.asarray(values)[graph_indices_np])


def iter_cached_batches(
    cache: "graph_cache.MolGraphCache",
    targets,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    fold_lookup=None,
):
    """Yield cached graph batches with aligned targets and optional fold ids."""
    for batch in cache.iter_batches_from_indices(
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        ):
        graph_indices_np = np.asarray(batch.graph_indices, dtype=np.int64)
        graph_indices_mx = _mx_int32_array(batch.graph_indices)
        out = [
            batch.V,
            batch.E,
            _mx_int32_array(batch.edge_index),
            _mx_int32_array(batch.rev_edge_index),
            _mx_int32_array(batch.batch),
            batch.num_graphs,
            _index_values(targets, graph_indices_np, graph_indices_mx),
            graph_indices_mx,
        ]
        if fold_lookup is not None:
            out.append(_index_values(fold_lookup, graph_indices_np, graph_indices_mx))
        yield tuple(out)
