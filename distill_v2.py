"""Cached v2 refinement: seed set + hardest OOD molecules from an eval pool."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_graphs.utils import scatter

from chemeleon_smd import graph_cache
from chemeleon_smd import score_dmpnn

WEIGHTS_DIR = Path(__file__).parent / "chemeleon_smd" / "weights"


def graph_batch_to_mx(batch: graph_cache.CachedMolGraphBatch):
    return (
        mx.array(batch.V),
        mx.array(batch.E),
        mx.array(batch.edge_index.astype(np.int32, copy=False)),
        mx.array(batch.rev_edge_index.astype(np.int32, copy=False)),
        mx.array(batch.batch.astype(np.int32, copy=False)),
        batch.num_graphs,
    )


def get_fingerprints(model, V, E, ei, rev, batch, ng):
    h_v = model(V, E, ei, rev)
    return scatter(h_v, batch, out_size=ng, aggr="mean")


def build_combined_dataset(seed_fps_path: Path, pool_fps_path: Path, eval_results_path: Path, n_worst: int):
    seed = np.load(seed_fps_path, allow_pickle=True)
    seed_fps = seed["fingerprints"]
    seed_smiles = list(seed["smiles"].astype(str))
    print(f"Seed set: {len(seed_smiles)} molecules")

    eval_data = np.load(eval_results_path, allow_pickle=True)
    eval_smiles = list(eval_data["smiles"].astype(str))
    eval_cosine = eval_data["cosine_sim"]
    eval_in_domain = eval_data["in_domain"]

    ood_indices = np.where(~eval_in_domain)[0]
    worst_order = np.argsort(eval_cosine[ood_indices])[:n_worst]
    worst_global_idx = ood_indices[worst_order]
    worst_smiles = [eval_smiles[i] for i in worst_global_idx]
    print(f"Selected {len(worst_smiles)} hardest out-of-domain molecules")

    pool = np.load(pool_fps_path, allow_pickle=True)
    pool_fps = pool["fingerprints"]
    pool_smiles = list(pool["smiles"].astype(str))
    pool_lookup = {s: i for i, s in enumerate(pool_smiles)}
    worst_fps = np.array([pool_fps[pool_lookup[s]] for s in worst_smiles], dtype=np.float32)

    combined_smiles = seed_smiles + worst_smiles
    combined_fps = np.concatenate([seed_fps, worst_fps], axis=0).astype(np.float32)
    print(f"Combined v2 training set: {len(combined_smiles)} molecules")
    return combined_smiles, combined_fps


def train_student(
    smiles: list[str],
    teacher_fps: np.ndarray,
    cache_dir: Path,
    weights_out: Path,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    val_fraction: float,
    seed: int,
    shard_size: int,
    rebuild_graph_cache: bool,
):
    cache = graph_cache.load_or_build_graph_cache(
        smiles,
        str(cache_dir),
        shard_size=shard_size,
        rebuild=rebuild_graph_cache,
        log=print,
    )
    if cache.n_graphs != len(smiles):
        raise ValueError(
            f"Training graph cache mismatch: {cache.n_graphs} cached graphs vs {len(smiles)} targets"
        )

    student = score_dmpnn.init_student_from_teacher(
        score_dmpnn.load_teacher_dmpnn(),
        skip_alpha=0.5,
    )
    n_params = sum(p.size for _, p in tree_flatten(student.parameters()))
    print(f"Student SCORE-DMPNN: {n_params:,} params")

    rng = np.random.RandomState(seed)
    n = cache.n_graphs
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    optimizer = optim.Adam(learning_rate=lr)

    def distill_loss(model, V, E, ei, rev, batch_arr, ng, target_fp):
        student_fp = get_fingerprints(model, V, E, ei, rev, batch_arr, ng)
        return mx.mean((student_fp - target_fp) ** 2)

    loss_and_grad = nn.value_and_grad(student, distill_loss)

    best_val = float("inf")
    patience_ctr = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        student.train()

        train_loss_sum = 0.0
        train_batches = 0
        for batch in cache.iter_batches_from_indices(
            train_idx,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            V, E, ei, rev, batch_arr, ng = graph_batch_to_mx(batch)
            target_fp = mx.array(teacher_fps[batch.graph_indices])
            loss, grads = loss_and_grad(student, V, E, ei, rev, batch_arr, ng, target_fp)
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state)
            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        student.eval()
        val_loss_sum = 0.0
        val_batches = 0
        for batch in cache.iter_batches_from_indices(val_idx, batch_size=batch_size, shuffle=False):
            V, E, ei, rev, batch_arr, ng = graph_batch_to_mx(batch)
            target_fp = mx.array(teacher_fps[batch.graph_indices])
            student_fp = get_fingerprints(student, V, E, ei, rev, batch_arr, ng)
            val_loss_sum += mx.mean((student_fp - target_fp) ** 2).item()
            val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  ({elapsed:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_weights = dict(tree_flatten(student.parameters()))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_weights is not None:
        student.load_weights(list(best_weights.items()))

    weights_out.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(weights_out), **dict(tree_flatten(student.parameters())))
    print(f"Saved distilled v2 weights to {weights_out}")


def main():
    parser = argparse.ArgumentParser(description="SCORE-DMPNN Distillation v2")
    parser.add_argument("--seed-fps", type=str, required=True)
    parser.add_argument("--pool-fps", type=str, required=True)
    parser.add_argument("--eval-results", type=str, required=True)
    parser.add_argument("--n-worst", type=int, default=10_000)
    parser.add_argument("--dataset-out", type=str, default=None)
    parser.add_argument("--weights-out", type=str, default=None)
    parser.add_argument("--graph-cache-dir", type=str, default=None)
    parser.add_argument("--graph-cache-shard-size", type=int, default=graph_cache.DEFAULT_SHARD_SIZE)
    parser.add_argument("--rebuild-graph-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    smiles, teacher_fps = build_combined_dataset(
        Path(args.seed_fps),
        Path(args.pool_fps),
        Path(args.eval_results),
        n_worst=args.n_worst,
    )

    dataset_out = Path(args.dataset_out) if args.dataset_out else WEIGHTS_DIR / "distill_v2_dataset.npz"
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dataset_out, fingerprints=teacher_fps, smiles=np.asarray(smiles))
    print(f"Saved v2 training dataset to {dataset_out}")

    weights_out = Path(args.weights_out) if args.weights_out else WEIGHTS_DIR / "score_dmpnn_distilled_v2.npz"
    cache_dir = Path(args.graph_cache_dir) if args.graph_cache_dir else WEIGHTS_DIR / "graph_cache_distill_v2"
    train_student(
        smiles=smiles,
        teacher_fps=teacher_fps,
        cache_dir=cache_dir,
        weights_out=weights_out,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
        shard_size=args.graph_cache_shard_size,
        rebuild_graph_cache=args.rebuild_graph_cache,
    )


if __name__ == "__main__":
    main()
