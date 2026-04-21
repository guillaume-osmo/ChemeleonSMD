"""SCORE-DMPNN finetuning demo: lipophilicity prediction.

Replicates the CheMeleon finetuning_demo.ipynb but using SCORE-DMPNN on MLX.
Compares v3, v4, v5, v6 distilled weights + teacher (original CheMeleon).

Usage:
    cd ChemeleonSMD
    PYTHONPATH=. python finetuning_demo.py
"""

from collections import defaultdict, deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from mlx.utils import tree_flatten

from chemeleon_smd._graph_ops import scatter
from chemeleon_smd import convert_weights
from chemeleon_smd import graph_cache
from chemeleon_smd import score_dmpnn

WEIGHTS_DIR = Path(__file__).parent / "chemeleon_smd" / "weights"
GRAPH_CACHE_DIR = WEIGHTS_DIR / "graph_cache_lipophilicity"


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


def iter_supervised_batches(
    cache: graph_cache.MolGraphCache,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
):
    for batch in cache.iter_batches_from_indices(
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    ):
        yield (
            mx.array(batch.V),
            mx.array(batch.E),
            mx.array(batch.edge_index.astype(np.int32, copy=False)),
            mx.array(batch.rev_edge_index.astype(np.int32, copy=False)),
            mx.array(batch.batch.astype(np.int32, copy=False)),
            batch.num_graphs,
            mx.array(targets[batch.graph_indices]),
        )


def train_and_eval(model_name, model, cache, targets, train_idx, val_idx, test_idx,
                   epochs=20, lr=1e-3, batch_size=16):
    """Train an FFN model and return best validation + test metrics."""
    optimizer = optim.Adam(learning_rate=lr)

    def mse_loss(model, V, E, ei, rev, batch, ng, y):
        pred = model(V, E, ei, rev, batch, ng)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, mse_loss)

    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for V, E, ei, rev, batch_arr, ng, y in iter_supervised_batches(
            cache,
            targets,
            train_idx,
            batch_size=batch_size,
            shuffle=True,
            seed=epoch,
        ):
            loss, grads = loss_and_grad(model, V, E, ei, rev, batch_arr, ng, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        for V, E, ei, rev, batch_arr, ng, y in iter_supervised_batches(
            cache,
            targets,
            val_idx,
            batch_size=batch_size,
            shuffle=False,
        ):
            pred = model(V, E, ei, rev, batch_arr, ng)
            mx.eval(pred)
            val_preds.extend(np.array(pred).tolist())
            val_true.extend(np.array(y).tolist())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_rmse = float(np.sqrt(np.mean((val_preds - val_true) ** 2)))
        val_mae = float(np.mean(np.abs(val_preds - val_true)))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            best_weights = dict(tree_flatten(model.parameters()))

    # Restore best and test
    if best_weights is not None:
        model.load_weights(list(best_weights.items()))
    model.eval()

    test_preds, test_true = [], []
    for V, E, ei, rev, batch_arr, ng, y in iter_supervised_batches(
        cache,
        targets,
        test_idx,
        batch_size=batch_size,
        shuffle=False,
    ):
        pred = model(V, E, ei, rev, batch_arr, ng)
        mx.eval(pred)
        test_preds.extend(np.array(pred).tolist())
        test_true.extend(np.array(y).tolist())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    test_rmse = float(np.sqrt(np.mean((test_preds - test_true) ** 2)))
    test_mae = float(np.mean(np.abs(test_preds - test_true)))
    rmse_gap = test_rmse - best_val_rmse
    mae_gap = test_mae - best_val_mae

    n_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    n_ffn = sum(p.size for n, p in tree_flatten(model.parameters())
                if n.startswith("ffn"))
    print(f"  {model_name:30s}  "
          f"valRMSE={best_val_rmse:.4f}  testRMSE={test_rmse:.4f}  dRMSE={rmse_gap:+.4f}  "
          f"valMAE={best_val_mae:.4f}  testMAE={test_mae:.4f}  dMAE={mae_gap:+.4f}  "
          f"(params: {n_total:,} total, {n_ffn:,} FFN)")
    return {
        "val_rmse": best_val_rmse,
        "val_mae": best_val_mae,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }


def main():
    print("=" * 70)
    print("SCORE-DMPNN Finetuning Demo: Lipophilicity Prediction")
    print("=" * 70)

    # Load full MoleculeNet Lipophilicity dataset (4,200 molecules)
    print("\nLoading MoleculeNet Lipophilicity dataset...")
    df = pd.read_csv(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    )
    smiles = df["smiles"].values
    targets = df["exp"].values.astype(np.float32)
    print(f"  {len(smiles)} molecules, target: lipophilicity (exp)")

    mol_graph_cache = graph_cache.load_or_build_graph_cache(
        smiles.tolist(),
        str(GRAPH_CACHE_DIR),
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

    # Random split 80/10/10 (same as CheMeleon demo)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    print(f"  Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Normalize targets
    train_mean = targets[train_idx].mean()
    train_std = targets[train_idx].std()
    targets_norm = (targets - train_mean) / train_std

    results = []

    # 1) Teacher (CheMeleon original) - frozen backbone
    print("\n--- Teacher: CheMeleon DMPNN (frozen backbone) ---")
    teacher = convert_weights.load_teacher()
    teacher.eval()
    model = MPNN_FFN(teacher, hidden_dim=300, freeze_backbone=True)
    metrics = train_and_eval(
        "CheMeleon (frozen)", model, mol_graph_cache, targets_norm,
        train_idx, val_idx, test_idx, epochs=20, lr=1e-3,
    )
    results.append((
        "CheMeleon teacher (frozen)",
        metrics["val_rmse"] * train_std,
        metrics["test_rmse"] * train_std,
        metrics["val_mae"] * train_std,
        metrics["test_mae"] * train_std,
    ))

    # 2) Teacher - finetuned
    print("\n--- Teacher: CheMeleon DMPNN (finetuned) ---")
    teacher2 = convert_weights.load_teacher()
    model = MPNN_FFN(teacher2, hidden_dim=300, freeze_backbone=False)
    metrics = train_and_eval(
        "CheMeleon (finetuned)", model, mol_graph_cache, targets_norm,
        train_idx, val_idx, test_idx, epochs=20, lr=1e-4,
    )
    results.append((
        "CheMeleon teacher (finetuned)",
        metrics["val_rmse"] * train_std,
        metrics["test_rmse"] * train_std,
        metrics["val_mae"] * train_std,
        metrics["test_mae"] * train_std,
    ))

    # 3) SCORE-DMPNN v3-v6, frozen backbone
    for version in ["v3", "v4", "v5", "v6"]:
        print(f"\n--- SCORE-DMPNN {version} (frozen backbone) ---")
        w_path = WEIGHTS_DIR / f"score_dmpnn_distilled_{version}.npz"
        backbone = score_dmpnn.ScoreDMPNN(
            d_v=72, d_e=14, d_h=2048, n_steps=6, skip_alpha=0.5, dropout=0.0
        )
        backbone.load_weights(list(mx.load(str(w_path)).items()))
        backbone.eval()
        model = MPNN_FFN(backbone, hidden_dim=300, freeze_backbone=True)
        metrics = train_and_eval(
            f"SCORE {version} (frozen)", model, mol_graph_cache, targets_norm,
            train_idx, val_idx, test_idx, epochs=20, lr=1e-3,
        )
        results.append((
            f"SCORE-DMPNN {version} (frozen)",
            metrics["val_rmse"] * train_std,
            metrics["test_rmse"] * train_std,
            metrics["val_mae"] * train_std,
            metrics["test_mae"] * train_std,
        ))

    # 4) SCORE-DMPNN v3-v6 finetuned
    for version in ["v3", "v4", "v5", "v6"]:
        print(f"\n--- SCORE-DMPNN {version} (finetuned) ---")
        w_path = WEIGHTS_DIR / f"score_dmpnn_distilled_{version}.npz"
        backbone = score_dmpnn.ScoreDMPNN(
            d_v=72, d_e=14, d_h=2048, n_steps=6, skip_alpha=0.5, dropout=0.0
        )
        backbone.load_weights(list(mx.load(str(w_path)).items()))
        model = MPNN_FFN(backbone, hidden_dim=300, freeze_backbone=False)
        metrics = train_and_eval(
            f"SCORE {version} (finetuned)", model, mol_graph_cache, targets_norm,
            train_idx, val_idx, test_idx, epochs=20, lr=1e-4,
        )
        results.append((
            f"SCORE-DMPNN {version} (finetuned)",
            metrics["val_rmse"] * train_std,
            metrics["test_rmse"] * train_std,
            metrics["val_mae"] * train_std,
            metrics["test_mae"] * train_std,
        ))

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY (Lipophilicity, {n} molecules, 80/10/10 split)")
    print("=" * 70)
    print(f"  {'Model':32s}  {'Val RMSE':>8s}  {'Test RMSE':>9s}  {'dRMSE':>8s}  "
          f"{'Val MAE':>8s}  {'Test MAE':>8s}  {'dMAE':>8s}")
    print("-" * 95)
    for name, val_rmse, test_rmse, val_mae, test_mae in results:
        print(f"  {name:32s}  {val_rmse:8.4f}  {test_rmse:9.4f}  {test_rmse - val_rmse:+8.4f}  "
              f"{val_mae:8.4f}  {test_mae:8.4f}  {test_mae - val_mae:+8.4f}")


if __name__ == "__main__":
    main()
