"""ChemeleonSMD-AllinOne v6 lipophilicity benchmark with cached graphs.

This keeps the same held-out 80/10/10 split as ``finetuning_demo.py``:

- train split: internal CV5-style fold masking across five v6 experts
- val split: unseen-by-all-experts checkpoint selection and ensemble fitting
- test split: final ensemble evaluation

The graph path is cached-only: we reuse the persistent molecular graph cache
and never re-featurize inside the training loop.

Usage:
    cd ChemeleonSMD
    PYTHONPATH=. python finetuning_allinone_v6_cv5.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from mlx.utils import tree_flatten

from chemeleon_smd import all_in_one
from chemeleon_smd import all_in_one_batches
from chemeleon_smd import graph_cache
from chemeleon_smd import score_dmpnn
from chemeleon_smd._graph_ops import scatter

WEIGHTS_DIR = Path(__file__).parent / "chemeleon_smd" / "weights"
GRAPH_CACHE_DIR = WEIGHTS_DIR / "graph_cache_lipophilicity"
LIPOPHILICITY_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
)


class MPNN_FFN(nn.Module):
    """MPNN backbone + 2-layer FFN head with LeakyReLU for regression."""

    def __init__(self, backbone, hidden_dim=300, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.ffn1 = nn.Linear(backbone.d_h, hidden_dim)
        self.ffn2 = nn.Linear(hidden_dim, 1)

    def __call__(self, V, E, ei, rev, batch, ng):
        if self.freeze_backbone:
            H_v = mx.stop_gradient(self.backbone(V, E, ei, rev))
        else:
            H_v = self.backbone(V, E, ei, rev)
        fp = scatter(H_v, batch, out_size=ng, aggr="mean")
        h = nn.leaky_relu(self.ffn1(fp))
        return self.ffn2(h).reshape(-1)


class AllInOneV6Ensemble(nn.Module):
    """Five independent v6 experts trained with fold-masked supervision."""

    def __init__(self, experts: list[MPNN_FFN]):
        super().__init__()
        if len(experts) < 2:
            raise ValueError("All-in-one benchmark requires at least two experts.")
        self.experts = experts
        self.num_experts = len(experts)

    def expert_predictions(self, V, E, ei, rev, batch, ng):
        preds = [expert(V, E, ei, rev, batch, ng) for expert in self.experts]
        return mx.stack(preds, axis=1)


def single_expert_masked_normalized_mse_loss(
    expert: MPNN_FFN,
    expert_id: int,
    V,
    E,
    ei,
    rev,
    batch_arr,
    ng,
    y_raw: mx.array,
    fold_ids: mx.array,
    target_means: mx.array,
    target_stds: mx.array,
) -> mx.array:
    """Per-expert masked loss to keep peak training memory low."""
    pred_norm = expert(V, E, ei, rev, batch_arr, ng)
    target_norm = (y_raw - target_means[expert_id]) / target_stds[expert_id]
    weights = (fold_ids != expert_id).astype(pred_norm.dtype)
    sqerr = (pred_norm - target_norm) ** 2
    denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=pred_norm.dtype))
    return mx.sum(sqerr * weights) / denom


def align_smiles_to_cache(source_smiles, cached_smiles):
    positions = defaultdict(deque)
    for idx, smi in enumerate(source_smiles):
        positions[str(smi)].append(idx)

    aligned = []
    for smi in cached_smiles:
        if not positions[smi]:
            raise ValueError(f"Could not align cached SMILES back to targets: {smi}")
        aligned.append(positions[smi].popleft())
    return np.array(aligned, dtype=np.int64)


def build_fold_ids(num_items: int, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_items)
    fold_ids = np.empty(num_items, dtype=np.int32)
    for fold, chunk in enumerate(np.array_split(perm, n_folds)):
        fold_ids[chunk] = fold
    return fold_ids


def compute_per_expert_target_stats(
    raw_targets: np.ndarray,
    fold_ids: np.ndarray,
    n_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(n_experts, dtype=np.float32)
    stds = np.zeros(n_experts, dtype=np.float32)
    for expert_id in range(n_experts):
        train_mask = fold_ids != expert_id
        if not np.any(train_mask):
            raise ValueError(f"Expert {expert_id} has no training targets.")
        means[expert_id] = raw_targets[train_mask].mean(dtype=np.float64)
        std = raw_targets[train_mask].std(dtype=np.float64)
        stds[expert_id] = max(float(std), 1e-6)
    return means, stds


def rmse_and_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae


def collect_raw_expert_predictions(
    model: AllInOneV6Ensemble,
    cache: graph_cache.MolGraphCache,
    raw_targets_np: np.ndarray,
    raw_targets_mx: mx.array,
    indices: np.ndarray,
    batch_size: int,
    target_means: mx.array,
    target_stds: mx.array,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.full(cache.n_graphs, -1, dtype=np.int64)
    pos[indices] = np.arange(len(indices), dtype=np.int64)
    preds = np.zeros((len(indices), model.num_experts), dtype=np.float32)
    y = raw_targets_np[indices].astype(np.float32, copy=False)

    model.eval()
    for V, E, ei, rev, batch_arr, ng, _, graph_indices in all_in_one_batches.iter_cached_batches(
        cache,
        raw_targets_mx,
        indices,
        batch_size=batch_size,
        shuffle=False,
    ):
        pred_norm = model.expert_predictions(V, E, ei, rev, batch_arr, ng)
        pred_raw = all_in_one.denormalize_predictions(
            pred_norm,
            target_means=target_means,
            target_stds=target_stds,
        )
        mx.eval(pred_raw)
        graph_indices_np = np.array(graph_indices, dtype=np.int64)
        preds[pos[graph_indices_np]] = np.array(pred_raw)

    return preds, y


def stream_metrics(
    model: AllInOneV6Ensemble,
    cache: graph_cache.MolGraphCache,
    raw_targets_mx: mx.array,
    indices: np.ndarray,
    batch_size: int,
    target_means: mx.array,
    target_stds: mx.array,
    fold_lookup=None,
    blind_holdout: bool = False,
) -> tuple[float, float]:
    """Compute RMSE/MAE without materializing a full prediction matrix."""
    sqerr_sum = 0.0
    abserr_sum = 0.0
    count = 0

    model.eval()
    for batch in all_in_one_batches.iter_cached_batches(
        cache,
        raw_targets_mx,
        indices,
        batch_size=batch_size,
        shuffle=False,
        fold_lookup=fold_lookup,
    ):
        if blind_holdout:
            V, E, ei, rev, batch_arr, ng, y_raw, _, fold_ids = batch
        else:
            V, E, ei, rev, batch_arr, ng, y_raw, _ = batch
            fold_ids = None

        pred_norm = model.expert_predictions(V, E, ei, rev, batch_arr, ng)
        pred_raw = all_in_one.denormalize_predictions(
            pred_norm,
            target_means=target_means,
            target_stds=target_stds,
        )
        if blind_holdout:
            pred = all_in_one.blind_holdout_predictions(pred_raw, fold_ids)
        else:
            pred = all_in_one.mean_ensemble_prediction(pred_raw)

        diff = pred - y_raw
        sq = mx.sum(diff * diff)
        ab = mx.sum(mx.abs(diff))
        mx.eval(sq, ab)
        sqerr_sum += float(sq.item())
        abserr_sum += float(ab.item())
        count += int(y_raw.shape[0])

    if count == 0:
        raise ValueError("stream_metrics received an empty split.")
    return float(np.sqrt(sqerr_sum / count)), float(abserr_sum / count)


def train_global_blend(
    val_expert_predictions: np.ndarray,
    val_targets: np.ndarray,
    epochs: int,
    lr: float,
) -> all_in_one.GlobalSoftmaxBlend:
    blend = all_in_one.GlobalSoftmaxBlend(val_expert_predictions.shape[1])
    optimizer = optim.Adam(learning_rate=lr)
    x = mx.array(val_expert_predictions)
    y = mx.array(val_targets)

    def loss_fn(module, preds, target):
        out = module(preds)
        return mx.mean((out - target) ** 2)

    loss_and_grad = nn.value_and_grad(blend, loss_fn)

    for _ in range(epochs):
        loss, grads = loss_and_grad(blend, x, y)
        optimizer.update(blend, grads)
        mx.eval(blend.parameters(), optimizer.state, loss)

    return blend


def build_v6_expert(hidden_dim: int, freeze_backbone: bool) -> MPNN_FFN:
    w_path = WEIGHTS_DIR / "score_dmpnn_distilled_v6.npz"
    backbone = score_dmpnn.ScoreDMPNN(
        d_v=72,
        d_e=14,
        d_h=2048,
        n_steps=6,
        skip_alpha=0.5,
        dropout=0.0,
    )
    backbone.load_weights(list(mx.load(str(w_path)).items()))
    return MPNN_FFN(backbone, hidden_dim=hidden_dim, freeze_backbone=freeze_backbone)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChemeleonSMD-AllinOne v6 cached lipophilicity benchmark."
    )
    parser.add_argument("--data-csv", type=str, default=LIPOPHILICITY_URL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--blend-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--blend-lr", type=float, default=5e-2)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold-seed", type=int, default=13)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("ChemeleonSMD-AllinOne v6: Cached Lipophilicity Benchmark")
    print("=" * 78)

    print("\nLoading MoleculeNet Lipophilicity dataset...")
    df = pd.read_csv(args.data_csv)
    smiles = df["smiles"].values
    targets = df["exp"].values.astype(np.float32)
    print(f"  {len(smiles)} molecules, target: lipophilicity (exp)")

    mol_graph_cache = graph_cache.load_or_build_graph_cache(
        smiles.tolist(),
        str(GRAPH_CACHE_DIR),
        rebuild=args.rebuild_cache,
        log=print,
    )
    cached_smiles = mol_graph_cache.get_smiles()
    if mol_graph_cache.n_graphs != len(smiles):
        keep_idx = align_smiles_to_cache(smiles.tolist(), cached_smiles)
        smiles = np.array(cached_smiles)
        targets = targets[keep_idx]
        print(f"  Graph cache kept {len(smiles)} valid molecules")
    else:
        smiles = np.array(cached_smiles)
        print(f"  Graph cache contains all {len(smiles)} molecules")

    n = mol_graph_cache.n_graphs
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    print(f"  Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    train_fold_ids = build_fold_ids(len(train_idx), args.n_folds, args.fold_seed)
    train_fold_lookup = np.full(n, -1, dtype=np.int32)
    train_fold_lookup[train_idx] = train_fold_ids
    print(f"  Internal train CV folds: {args.n_folds}")

    targets_mx = mx.array(targets)
    train_fold_lookup_mx = mx.array(train_fold_lookup)

    train_targets = targets[train_idx]
    target_means_np, target_stds_np = compute_per_expert_target_stats(
        train_targets,
        train_fold_ids,
        args.n_folds,
    )
    target_means = mx.array(target_means_np)
    target_stds = mx.array(target_stds_np)

    experts = [
        build_v6_expert(
            hidden_dim=args.hidden_dim,
            freeze_backbone=args.freeze_backbone,
        )
        for _ in range(args.n_folds)
    ]
    model = AllInOneV6Ensemble(experts)
    expert_optimizers = [optim.Adam(learning_rate=args.lr) for _ in range(args.n_folds)]
    expert_loss_and_grad = []
    for expert_id, expert in enumerate(model.experts):
        def make_loss_fn(current_expert_id: int):
            def loss_fn(module, V, E, ei, rev, batch_arr, ng, y_raw, fold_ids):
                return single_expert_masked_normalized_mse_loss(
                    module,
                    expert_id=current_expert_id,
                    V=V,
                    E=E,
                    ei=ei,
                    rev=rev,
                    batch_arr=batch_arr,
                    ng=ng,
                    y_raw=y_raw,
                    fold_ids=fold_ids,
                    target_means=target_means,
                    target_stds=target_stds,
                )

            return loss_fn

        expert_loss_and_grad.append(nn.value_and_grad(expert, make_loss_fn(expert_id)))

    best_val_rmse = float("inf")
    best_weights = None
    best_epoch = 0

    print(
        "\nTraining five v6 experts with sequential fold-masked updates "
        "on the cached train split..."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for V, E, ei, rev, batch_arr, ng, y_raw, _, fold_ids in all_in_one_batches.iter_cached_batches(
            mol_graph_cache,
            targets_mx,
            train_idx,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            fold_lookup=train_fold_lookup_mx,
        ):
            for expert_id, expert in enumerate(model.experts):
                loss, grads = expert_loss_and_grad[expert_id](
                    expert,
                    V,
                    E,
                    ei,
                    rev,
                    batch_arr,
                    ng,
                    y_raw,
                    fold_ids,
                )
                expert_optimizers[expert_id].update(expert, grads)
                mx.eval(
                    expert.parameters(),
                    expert_optimizers[expert_id].state,
                    loss,
                )

        train_oof_rmse, train_oof_mae = stream_metrics(
            model,
            mol_graph_cache,
            targets_mx,
            train_idx,
            batch_size=args.batch_size,
            target_means=target_means,
            target_stds=target_stds,
            fold_lookup=train_fold_lookup_mx,
            blind_holdout=True,
        )

        val_rmse, val_mae = stream_metrics(
            model,
            mol_graph_cache,
            targets_mx,
            val_idx,
            batch_size=args.batch_size,
            target_means=target_means,
            target_stds=target_stds,
        )

        print(
            f"  epoch {epoch:02d}  "
            f"trainOOF RMSE={train_oof_rmse:.4f}  trainOOF MAE={train_oof_mae:.4f}  "
            f"valMean RMSE={val_rmse:.4f}  valMean MAE={val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_weights = [
                dict(tree_flatten(expert.parameters())) for expert in model.experts
            ]

    if best_weights is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    for expert, weights in zip(model.experts, best_weights):
        expert.load_weights(list(weights.items()))
    model.eval()

    train_oof_rmse, train_oof_mae = stream_metrics(
        model,
        mol_graph_cache,
        targets_mx,
        train_idx,
        batch_size=args.batch_size,
        target_means=target_means,
        target_stds=target_stds,
        fold_lookup=train_fold_lookup_mx,
        blind_holdout=True,
    )

    val_mean_rmse, val_mean_mae = stream_metrics(
        model,
        mol_graph_cache,
        targets_mx,
        val_idx,
        batch_size=args.batch_size,
        target_means=target_means,
        target_stds=target_stds,
    )

    test_mean_rmse, test_mean_mae = stream_metrics(
        model,
        mol_graph_cache,
        targets_mx,
        test_idx,
        batch_size=args.batch_size,
        target_means=target_means,
        target_stds=target_stds,
    )

    val_expert_preds, val_true = collect_raw_expert_predictions(
        model,
        mol_graph_cache,
        targets,
        targets_mx,
        val_idx,
        batch_size=args.batch_size,
        target_means=target_means,
        target_stds=target_stds,
    )
    test_expert_preds, test_true = collect_raw_expert_predictions(
        model,
        mol_graph_cache,
        targets,
        targets_mx,
        test_idx,
        batch_size=args.batch_size,
        target_means=target_means,
        target_stds=target_stds,
    )

    print("\nFitting a small validation-trained softmax blend for test-time ensembling...")
    blend = train_global_blend(
        val_expert_predictions=val_expert_preds,
        val_targets=val_true,
        epochs=args.blend_epochs,
        lr=args.blend_lr,
    )
    blend_weights = np.array(blend.weights())
    val_blend = np.array(blend(mx.array(val_expert_preds)))
    test_blend = np.array(blend(mx.array(test_expert_preds)))
    val_blend_rmse, val_blend_mae = rmse_and_mae(val_true, val_blend)
    test_blend_rmse, test_blend_mae = rmse_and_mae(test_true, test_blend)

    print("\n" + "=" * 78)
    print("RESULTS SUMMARY (Lipophilicity, cached graphs, v6 only)")
    print("=" * 78)
    mode_name = "frozen" if args.freeze_backbone else "finetuned"
    print(f"  Model: ChemeleonSMD-AllinOne v6 ({mode_name})")
    print(f"  Best epoch (by val mean RMSE): {best_epoch}")
    print(f"  Internal train CV5 OOF:   RMSE={train_oof_rmse:.4f}  MAE={train_oof_mae:.4f}")
    print(f"  Validation mean ensemble: RMSE={val_mean_rmse:.4f}  MAE={val_mean_mae:.4f}")
    print(f"  Validation softmax blend: RMSE={val_blend_rmse:.4f}  MAE={val_blend_mae:.4f}")
    print(f"  Test mean ensemble:       RMSE={test_mean_rmse:.4f}  MAE={test_mean_mae:.4f}")
    print(f"  Test softmax blend:       RMSE={test_blend_rmse:.4f}  MAE={test_blend_mae:.4f}")
    print(
        "  Blend weights (expert 0..4): "
        + ", ".join(f"{w:.3f}" for w in blend_weights.tolist())
    )


if __name__ == "__main__":
    main()
