"""SCORE-DMPNN finetuning demo: lipophilicity prediction.

Replicates the CheMeleon finetuning_demo.ipynb but using SCORE-DMPNN on MLX.
Compares v3, v4, v5, v6 distilled weights + teacher (original CheMeleon).

Usage:
    cd ChemeleonSMD
    PYTHONPATH=. python finetuning_demo.py
"""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from mlx.utils import tree_flatten
from mlx_graphs.utils import scatter

from chemeleon_smd import convert_weights
from chemeleon_smd import mol_featurizer as mf
from chemeleon_smd import score_dmpnn
from chemeleon_smd.mpnn import CheMeleonBondMPNN

WEIGHTS_DIR = Path(__file__).parent / "chemeleon_smd" / "weights"


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


def featurize_batch(smiles_list):
    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        g = mf.featurize_smiles(smi)
        if g is not None and g.V.shape[0] > 0 and g.E.shape[0] > 0:
            graphs.append(g)
            valid_idx.append(i)
    if not graphs:
        return None
    V, E, ei, rev, batch_arr, ng = mf.collate_mol_graphs(graphs)
    return (
        mx.array(V),
        mx.array(E),
        mx.array(ei.astype(np.int32)),
        mx.array(rev.astype(np.int32)),
        mx.array(batch_arr.astype(np.int32)),
        ng,
        valid_idx,
    )


def train_and_eval(model_name, model, smiles, targets, train_idx, val_idx, test_idx,
                   epochs=20, lr=1e-3, batch_size=16):
    """Train an FFN model and return test RMSE + MAE."""
    optimizer = optim.Adam(learning_rate=lr)

    def mse_loss(model, V, E, ei, rev, batch, ng, y):
        pred = model(V, E, ei, rev, batch, ng)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, mse_loss)

    best_val = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))

        for bstart in range(0, len(perm), batch_size):
            bidx = perm[bstart : bstart + batch_size]
            mol_idx = train_idx[bidx]
            batch_smi = [smiles[i] for i in mol_idx]
            batch_y = targets[mol_idx]

            result = featurize_batch(batch_smi)
            if result is None:
                continue
            V, E, ei, rev, batch_arr, ng, valid = result
            y = mx.array(batch_y[valid])

            loss, grads = loss_and_grad(model, V, E, ei, rev, batch_arr, ng, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        for bstart in range(0, len(val_idx), batch_size):
            bidx = val_idx[bstart : bstart + batch_size]
            batch_smi = [smiles[i] for i in bidx]
            batch_y = targets[bidx]

            result = featurize_batch(batch_smi)
            if result is None:
                continue
            V, E, ei, rev, batch_arr, ng, valid = result
            pred = model(V, E, ei, rev, batch_arr, ng)
            mx.eval(pred)
            val_preds.extend(np.array(pred).tolist())
            val_true.extend(batch_y[valid].tolist())

        val_rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_true)) ** 2))
        if val_rmse < best_val:
            best_val = val_rmse
            best_weights = dict(tree_flatten(model.parameters()))

    # Restore best and test
    if best_weights is not None:
        model.load_weights(list(best_weights.items()))
    model.eval()

    test_preds, test_true = [], []
    for bstart in range(0, len(test_idx), batch_size):
        bidx = test_idx[bstart : bstart + batch_size]
        batch_smi = [smiles[i] for i in bidx]
        batch_y = targets[bidx]

        result = featurize_batch(batch_smi)
        if result is None:
            continue
        V, E, ei, rev, batch_arr, ng, valid = result
        pred = model(V, E, ei, rev, batch_arr, ng)
        mx.eval(pred)
        test_preds.extend(np.array(pred).tolist())
        test_true.extend(batch_y[valid].tolist())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    rmse = float(np.sqrt(np.mean((test_preds - test_true) ** 2)))
    mae = float(np.mean(np.abs(test_preds - test_true)))

    n_total = sum(p.size for _, p in tree_flatten(model.parameters()))
    n_ffn = sum(p.size for n, p in tree_flatten(model.parameters())
                if n.startswith("ffn"))
    print(f"  {model_name:30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"(params: {n_total:,} total, {n_ffn:,} FFN)")
    return rmse, mae


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
    n = len(smiles)
    print(f"  {n} molecules, target: lipophilicity (exp)")

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
    rmse, mae = train_and_eval(
        "CheMeleon (frozen)", model, smiles, targets_norm,
        train_idx, val_idx, test_idx, epochs=20, lr=1e-3,
    )
    results.append(("CheMeleon teacher (frozen)", rmse * train_std, mae * train_std))

    # 2) Teacher - finetuned
    print("\n--- Teacher: CheMeleon DMPNN (finetuned) ---")
    teacher2 = convert_weights.load_teacher()
    model = MPNN_FFN(teacher2, hidden_dim=300, freeze_backbone=False)
    rmse, mae = train_and_eval(
        "CheMeleon (finetuned)", model, smiles, targets_norm,
        train_idx, val_idx, test_idx, epochs=20, lr=1e-4,
    )
    results.append(("CheMeleon teacher (finetuned)", rmse * train_std, mae * train_std))

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
        rmse, mae = train_and_eval(
            f"SCORE {version} (frozen)", model, smiles, targets_norm,
            train_idx, val_idx, test_idx, epochs=20, lr=1e-3,
        )
        results.append((f"SCORE-DMPNN {version} (frozen)", rmse * train_std, mae * train_std))

    # 4) SCORE-DMPNN v3-v6 finetuned
    for version in ["v3", "v4", "v5", "v6"]:
        print(f"\n--- SCORE-DMPNN {version} (finetuned) ---")
        w_path = WEIGHTS_DIR / f"score_dmpnn_distilled_{version}.npz"
        backbone = score_dmpnn.ScoreDMPNN(
            d_v=72, d_e=14, d_h=2048, n_steps=6, skip_alpha=0.5, dropout=0.0
        )
        backbone.load_weights(list(mx.load(str(w_path)).items()))
        model = MPNN_FFN(backbone, hidden_dim=300, freeze_backbone=False)
        rmse, mae = train_and_eval(
            f"SCORE {version} (finetuned)", model, smiles, targets_norm,
            train_idx, val_idx, test_idx, epochs=20, lr=1e-4,
        )
        results.append((f"SCORE-DMPNN {version} (finetuned)", rmse * train_std, mae * train_std))

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY (Lipophilicity, {n} molecules, 80/10/10 split)")
    print("=" * 70)
    print(f"  {'Model':40s}  {'RMSE':>8s}  {'MAE':>8s}")
    print("-" * 62)
    for name, rmse, mae in results:
        print(f"  {name:40s}  {rmse:8.4f}  {mae:8.4f}")


if __name__ == "__main__":
    main()
