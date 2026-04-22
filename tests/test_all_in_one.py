import mlx.core as mx
from chemeleon_smd import all_in_one
from chemeleon_smd.all_in_one_batches import iter_cached_batches


def test_mask_and_blind_pick():
    fold_ids = mx.array([0, 2], dtype=mx.int32)
    mask = all_in_one.fold_holdout_mask(fold_ids, 3)
    expected = mx.array([[False, True, True], [True, True, False]])
    assert mx.all(mask == expected).item()

    preds = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    blind = all_in_one.blind_holdout_predictions(preds, fold_ids)
    assert mx.allclose(blind, mx.array([1.0, 6.0])).item()


def test_normalize_denormalize_and_masked_loss():
    preds_norm = mx.array([[0.0, 1.0], [1.0, 2.0]])
    raw_targets = mx.array([10.0, 20.0])
    means = mx.array([8.0, 12.0])
    stds = mx.array([2.0, 4.0])
    fold_ids = mx.array([0, 1], dtype=mx.int32)

    raw_preds = all_in_one.denormalize_predictions(preds_norm, means, stds)
    assert mx.allclose(raw_preds, mx.array([[8.0, 16.0], [10.0, 20.0]])).item()

    norm_targets = all_in_one.normalize_targets_per_expert(raw_targets, means, stds)
    assert mx.allclose(norm_targets, mx.array([[1.0, -0.5], [6.0, 2.0]])).item()

    loss = all_in_one.masked_normalized_mse_loss(
        preds_norm,
        raw_targets,
        fold_ids,
        means,
        stds,
    )
    # Active pairs: sample0/expert1 => (1 - (-0.5))^2 = 2.25
    #               sample1/expert0 => (1 - 6)^2 = 25
    assert abs(float(loss.item()) - 13.625) < 1e-6


def test_global_softmax_blend_prefers_higher_logit():
    blend = all_in_one.GlobalSoftmaxBlend(3)
    blend.logits = mx.array([0.0, 3.0, -3.0])
    preds = mx.array([[1.0, 5.0, 9.0]])
    out = blend(preds)
    weights = blend.weights()
    assert float(weights[1]) > float(weights[0]) > float(weights[2])
    assert 4.0 < float(out[0]) < 6.0


class _FakeBatch:
    def __init__(self, V, graph_indices):
        self.V = V
        self.E = mx.zeros((len(graph_indices), 1))
        self.edge_index = mx.zeros((2, len(graph_indices)), dtype=mx.int32)
        self.rev_edge_index = mx.arange(len(graph_indices), dtype=mx.int32)
        self.batch = mx.arange(len(graph_indices), dtype=mx.int32)
        self.num_graphs = len(graph_indices)
        self.graph_indices = graph_indices


class _FakeCache:
    def __init__(self, order):
        self.order = order

    def iter_batches_from_indices(self, indices, batch_size, shuffle, seed):
        del indices, batch_size, shuffle, seed
        features = mx.array(
            [
                [3.0, 7.0],  # graph 2, fold 0 => expert 0 should be selected
                [1.0, 9.0],  # graph 0, fold 0 => expert 0 should be selected
                [6.0, 4.0],  # graph 3, fold 1 => expert 1 should be selected
                [8.0, 2.0],  # graph 1, fold 1 => expert 1 should be selected
            ]
        )
        yield _FakeBatch(features, self.order)


def test_shuffled_batch_mask_routes_holdout_submodel_correctly():
    graph_indices = mx.array([2, 0, 3, 1], dtype=mx.int64)
    cache = _FakeCache(graph_indices)
    targets = mx.array([1.0, 2.0, 3.0, 4.0]).astype(mx.float32)
    fold_lookup = mx.array([0, 1, 0, 1], dtype=mx.int32)

    batch = next(
        iter_cached_batches(
            cache,
            targets=targets,
            indices=mx.array([0, 1, 2, 3], dtype=mx.int64),
            batch_size=4,
            shuffle=True,
            seed=123,
            fold_lookup=fold_lookup,
        )
    )
    V, _, _, _, _, _, y_raw, returned_indices, fold_ids = batch

    assert mx.all(returned_indices == graph_indices).item()
    assert mx.all(fold_ids == mx.array([0, 0, 1, 1], dtype=mx.int32)).item()
    assert mx.allclose(y_raw, mx.array([3.0, 1.0, 4.0, 2.0])).item()

    preds = V  # expert 0 = first feature, expert 1 = second feature
    blind = all_in_one.blind_holdout_predictions(preds, fold_ids)
    assert mx.allclose(blind, y_raw).item()

    mask = all_in_one.fold_holdout_mask(fold_ids, num_experts=2)
    expected_mask = mx.array(
        [
            [False, True],
            [False, True],
            [True, False],
            [True, False],
        ]
    )
    assert mx.all(mask == expected_mask).item()

    loss = all_in_one.masked_normalized_mse_loss(
        preds,
        raw_targets=y_raw,
        fold_ids=fold_ids,
        target_means=mx.array([0.0, 0.0]),
        target_stds=mx.array([1.0, 1.0]),
    )
    assert abs(float(loss.item()) - 30.0) < 1e-6
