"""Distill a pretrained CheMeleon DMPNN into SCORE-DMPNN on cached graphs.

The workflow is intentionally two-phase:

1. `--precompute` builds teacher fingerprints once and saves them to disk.
2. `--distill` trains the SCORE student from those cached targets.

That keeps teacher and student out of memory at the same time and removes
repeated RDKit featurization from the epoch loop.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from chemeleon_smd._graph_ops import scatter
from chemeleon_smd import graph_cache
from chemeleon_smd import score_dmpnn

WEIGHTS_DIR = Path(__file__).parent / "chemeleon_smd" / "weights"


def read_smiles_file(smiles_file: str) -> list[str]:
    with open(smiles_file) as f:
        return [line.strip() for line in f if line.strip()]


def graph_batch_to_mx(batch: graph_cache.CachedMolGraphBatch):
    return (
        batch.V,
        batch.E,
        batch.edge_index,
        batch.rev_edge_index,
        batch.batch,
        batch.num_graphs,
    )


def get_fingerprints(model, V, E, ei, rev, batch, ng):
    h_v = model(V, E, ei, rev)
    return scatter(h_v, batch, out_size=ng, aggr="mean")


def precompute(
    cache: graph_cache.MolGraphCache,
    teacher_fps_path: Path,
    batch_size: int,
):
    print("=" * 60)
    print("Phase 1: Precompute teacher fingerprints")
    print("=" * 60)
    print(f"Molecules: {cache.n_graphs} valid cached graphs")

    teacher = score_dmpnn.load_teacher_dmpnn()
    teacher.eval()
    n_params = sum(p.size for _, p in tree_flatten(teacher.parameters()))
    print(f"Teacher loaded: {n_params:,} params")

    fps = None
    cursor = 0
    for batch_idx, batch in enumerate(
        cache.iter_batches(batch_size=batch_size, shuffle=False),
        start=1,
    ):
        V, E, ei, rev, batch_arr, ng = graph_batch_to_mx(batch)
        fp = mx.stop_gradient(get_fingerprints(teacher, V, E, ei, rev, batch_arr, ng))
        mx.eval(fp)
        fp_np = np.array(fp, dtype=np.float32)
        if fps is None:
            fps = np.zeros((cache.n_graphs, fp_np.shape[1]), dtype=np.float32)
        fps[cursor : cursor + fp_np.shape[0]] = fp_np
        cursor += fp_np.shape[0]

        if batch_idx % 50 == 0:
            print(f"  {cursor}/{cache.n_graphs}")

    if fps is None:
        raise ValueError("Graph cache yielded no valid batches during teacher precompute")

    teacher_fps_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        teacher_fps_path,
        fingerprints=fps[:cursor],
        smiles=np.asarray(cache.get_smiles()),
    )
    print(f"Saved teacher fingerprints to {teacher_fps_path}")

    del teacher, fps
    gc.collect()


def distill(
    cache: graph_cache.MolGraphCache,
    teacher_fps_path: Path,
    weights_out: Path,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    val_fraction: float,
    seed: int,
):
    print("=" * 60)
    print("Phase 2: Distill SCORE-DMPNN")
    print("=" * 60)

    data = np.load(teacher_fps_path, allow_pickle=True)
    teacher_fps = data["fingerprints"]
    teacher_fps_mx = mx.array(teacher_fps)
    cached_smiles = data["smiles"].astype(str).tolist()
    if cache.n_graphs != teacher_fps.shape[0]:
        raise ValueError(
            f"Graph cache / teacher fingerprint mismatch: {cache.n_graphs} cached graphs "
            f"vs {teacher_fps.shape[0]} teacher targets"
        )
    if cache.get_smiles() != cached_smiles:
        raise ValueError("Teacher fingerprint SMILES order does not match the graph cache order")

    student = score_dmpnn.init_student_from_teacher(
        score_dmpnn.load_teacher_dmpnn(),
        skip_alpha=0.5,
    )
    student.eval()
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
            target_fp = teacher_fps_mx[batch.graph_indices]
            loss, grads = loss_and_grad(student, V, E, ei, rev, batch_arr, ng, target_fp)
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state)
            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        student.eval()
        val_loss_sum = 0.0
        val_batches = 0
        for batch in cache.iter_batches_from_indices(
            val_idx,
            batch_size=batch_size,
            shuffle=False,
        ):
            V, E, ei, rev, batch_arr, ng = graph_batch_to_mx(batch)
            target_fp = teacher_fps_mx[batch.graph_indices]
            student_fp = get_fingerprints(student, V, E, ei, rev, batch_arr, ng)
            val_loss_sum += mx.mean((student_fp - target_fp) ** 2).item()
            val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.perf_counter() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  "
            f"val={val_loss:.6f}  ({elapsed:.1f}s)"
        )

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
    print(f"Saved distilled student weights to {weights_out}")


def main():
    parser = argparse.ArgumentParser(description="Distill CheMeleon DMPNN -> SCORE-DMPNN")
    parser.add_argument("--smiles-file", type=str, required=True)
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--teacher-fps", type=str, default=None)
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

    if not args.precompute and not args.distill:
        parser.error("Choose at least one of --precompute or --distill")

    stem = Path(args.smiles_file).stem
    teacher_fps_path = Path(args.teacher_fps) if args.teacher_fps else WEIGHTS_DIR / f"teacher_fps_{stem}.npz"
    weights_out = Path(args.weights_out) if args.weights_out else WEIGHTS_DIR / f"score_dmpnn_distilled_{stem}.npz"
    cache_dir = Path(args.graph_cache_dir) if args.graph_cache_dir else WEIGHTS_DIR / f"graph_cache_{stem}"

    smiles = read_smiles_file(args.smiles_file)
    cache = graph_cache.load_or_build_graph_cache(
        smiles,
        str(cache_dir),
        shard_size=args.graph_cache_shard_size,
        rebuild=args.rebuild_graph_cache,
        log=print,
    )

    if args.precompute:
        precompute(cache, teacher_fps_path, batch_size=args.batch_size)

    if args.distill:
        distill(
            cache=cache,
            teacher_fps_path=teacher_fps_path,
            weights_out=weights_out,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
