"""Cached v3 refinement: base dataset + scaffold-diverse hard molecules."""

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
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from chemeleon_smd._graph_ops import scatter
from chemeleon_smd import graph_cache
from chemeleon_smd import score_dmpnn

WEIGHTS_DIR = Path(__file__).parent / "chemeleon_smd" / "weights"


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


def scaffold_select(smiles_list: list[str], n_select: int, seed: int = 42) -> list[int]:
    """Select a scaffold-diverse subset using Murcko scaffolds."""
    blocker = rdBase.BlockLogs()
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = GetScaffoldForMol(mol)
            scaf_smi = scaf.GetSmiles() if scaf.GetNumAtoms() > 0 else "empty"
        except Exception:
            scaf_smi = f"fail_{i}"
        scaffolds.setdefault(scaf_smi, []).append(i)
    del blocker

    rng = np.random.RandomState(seed)
    selected = []
    scaffold_keys = list(scaffolds.keys())
    rng.shuffle(scaffold_keys)

    while len(selected) < n_select and scaffold_keys:
        next_keys = []
        for key in scaffold_keys:
            if len(selected) >= n_select:
                break
            members = scaffolds[key]
            if members:
                idx = rng.choice(len(members))
                selected.append(members.pop(idx))
                if members:
                    next_keys.append(key)
        scaffold_keys = next_keys

    return selected[:n_select]


def build_v3_dataset(
    base_fps_path: Path,
    eval_smiles_file: Path,
    eval_summary_path: Path,
    hard_cache_dir: Path,
    hard_threshold: float,
    medium_threshold: float,
    n_medium_scaffold: int,
    n_hard_scaffold: int,
    shard_size: int,
    rebuild_graph_cache: bool,
):
    base = np.load(base_fps_path, allow_pickle=True)
    base_fps = base["fingerprints"].astype(np.float32)
    base_smiles = list(base["smiles"].astype(str))
    base_set = set(base_smiles)
    print(f"Base v2 training data: {len(base_smiles)} molecules")

    with open(eval_smiles_file) as f:
        eval_smiles = [line.strip() for line in f if line.strip()]

    eval_summary = np.load(eval_summary_path)
    cosine_sim = eval_summary["cosine_sim"]
    if len(eval_smiles) != len(cosine_sim):
        raise ValueError("Eval SMILES file length does not match eval summary cosine array")

    new_mask = np.array([s not in base_set for s in eval_smiles])
    hard_mask = (cosine_sim < hard_threshold) & new_mask
    medium_mask = (cosine_sim < medium_threshold) & (cosine_sim >= hard_threshold) & new_mask

    hard_idx = list(np.where(hard_mask)[0])
    medium_idx = np.where(medium_mask)[0]
    print(f"New molecules cosine < {hard_threshold:.2f}: {len(hard_idx)}")
    print(f"New molecules {hard_threshold:.2f} <= cosine < {medium_threshold:.2f}: {len(medium_idx)}")

    medium_smiles = [eval_smiles[i] for i in medium_idx]
    if n_medium_scaffold > 0 and medium_smiles:
        medium_sel = scaffold_select(medium_smiles, n_select=min(n_medium_scaffold, len(medium_smiles)))
        hard_idx.extend(int(medium_idx[j]) for j in medium_sel)

    hard_smiles_pool = [eval_smiles[i] for i in np.where(hard_mask)[0]]
    if n_hard_scaffold > 0 and hard_smiles_pool:
        hard_sel = scaffold_select(hard_smiles_pool, n_select=min(n_hard_scaffold, len(hard_smiles_pool)))
        hard_pool_idx = np.where(hard_mask)[0]
        hard_idx.extend(int(hard_pool_idx[j]) for j in hard_sel)

    hard_idx = sorted(set(hard_idx))
    hard_smiles = [eval_smiles[i] for i in hard_idx]
    print(f"Selected {len(hard_smiles)} new hard molecules for v3")

    hard_cache = graph_cache.load_or_build_graph_cache(
        hard_smiles,
        str(hard_cache_dir),
        shard_size=shard_size,
        rebuild=rebuild_graph_cache,
        log=print,
    )
    hard_valid_smiles = hard_cache.get_smiles()

    teacher = score_dmpnn.load_teacher_dmpnn()
    teacher.eval()

    hard_fps_list = []
    for batch in hard_cache.iter_batches(batch_size=16, shuffle=False):
        V, E, ei, rev, batch_arr, ng = graph_batch_to_mx(batch)
        fp = mx.stop_gradient(get_fingerprints(teacher, V, E, ei, rev, batch_arr, ng))
        mx.eval(fp)
        hard_fps_list.append(np.array(fp, dtype=np.float32))

    hard_fps = np.concatenate(hard_fps_list, axis=0).astype(np.float32)
    del teacher
    gc.collect()

    all_smiles = base_smiles + hard_valid_smiles
    all_fps = np.concatenate([base_fps, hard_fps], axis=0).astype(np.float32)
    print(f"Total v3 training set: {len(all_smiles)} molecules")
    return all_smiles, all_fps


def train_student(
    smiles: list[str],
    teacher_fps: np.ndarray,
    init_weights: Path,
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
    teacher_fps_mx = mx.array(teacher_fps)
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
    student.load_weights(list(mx.load(str(init_weights)).items()))

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
        for batch in cache.iter_batches_from_indices(val_idx, batch_size=batch_size, shuffle=False):
            V, E, ei, rev, batch_arr, ng = graph_batch_to_mx(batch)
            target_fp = teacher_fps_mx[batch.graph_indices]
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
    print(f"Saved distilled v3 weights to {weights_out}")


def main():
    parser = argparse.ArgumentParser(description="SCORE-DMPNN Distillation v3")
    parser.add_argument("--base-fps", type=str, required=True, help="NPZ with fingerprints + smiles from the current base training set")
    parser.add_argument("--eval-smiles-file", type=str, required=True)
    parser.add_argument("--eval-summary", type=str, required=True, help="NPZ containing cosine_sim aligned to eval-smiles-file")
    parser.add_argument("--init-weights", type=str, default=None, help="Starting checkpoint, typically a v2 distilled model")
    parser.add_argument("--dataset-out", type=str, default=None)
    parser.add_argument("--weights-out", type=str, default=None)
    parser.add_argument("--graph-cache-dir", type=str, default=None)
    parser.add_argument("--hard-graph-cache-dir", type=str, default=None)
    parser.add_argument("--graph-cache-shard-size", type=int, default=graph_cache.DEFAULT_SHARD_SIZE)
    parser.add_argument("--rebuild-graph-cache", action="store_true")
    parser.add_argument("--hard-threshold", type=float, default=0.95)
    parser.add_argument("--medium-threshold", type=float, default=0.98)
    parser.add_argument("--n-medium-scaffold", type=int, default=9_000)
    parser.add_argument("--n-hard-scaffold", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_weights = Path(args.init_weights) if args.init_weights else WEIGHTS_DIR / "score_dmpnn_distilled_v2.npz"
    dataset_out = Path(args.dataset_out) if args.dataset_out else WEIGHTS_DIR / "distill_v3_dataset.npz"
    weights_out = Path(args.weights_out) if args.weights_out else WEIGHTS_DIR / "score_dmpnn_distilled_v3.npz"
    cache_dir = Path(args.graph_cache_dir) if args.graph_cache_dir else WEIGHTS_DIR / "graph_cache_distill_v3"
    hard_cache_dir = Path(args.hard_graph_cache_dir) if args.hard_graph_cache_dir else WEIGHTS_DIR / "graph_cache_distill_v3_hard"

    smiles, teacher_fps = build_v3_dataset(
        base_fps_path=Path(args.base_fps),
        eval_smiles_file=Path(args.eval_smiles_file),
        eval_summary_path=Path(args.eval_summary),
        hard_cache_dir=hard_cache_dir,
        hard_threshold=args.hard_threshold,
        medium_threshold=args.medium_threshold,
        n_medium_scaffold=args.n_medium_scaffold,
        n_hard_scaffold=args.n_hard_scaffold,
        shard_size=args.graph_cache_shard_size,
        rebuild_graph_cache=args.rebuild_graph_cache,
    )

    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dataset_out, fingerprints=teacher_fps, smiles=np.asarray(smiles))
    print(f"Saved v3 training dataset to {dataset_out}")

    train_student(
        smiles=smiles,
        teacher_fps=teacher_fps,
        init_weights=init_weights,
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
