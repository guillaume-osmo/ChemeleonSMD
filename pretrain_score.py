"""Pretrain CheMeleon-SCORE on cached Mordred descriptors.

Reproduces the CheMeleon masked descriptor prediction objective while using a
SCORE-style encoder and a persistent molecular graph cache so RDKit work is
paid once instead of every epoch.
"""

import argparse
from collections import defaultdict, deque
import gzip
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles, RemoveHs

from chemeleon_smd import chemeleon_score as cs
from chemeleon_smd import graph_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"


def download_pubchem(cache_dir: str = "/tmp/chemeleon_data") -> Path:
    """Download PubChem CID-SMILES if not cached."""
    os.makedirs(cache_dir, exist_ok=True)
    gz_path = Path(cache_dir) / "CID-SMILES.gz"
    smiles_path = Path(cache_dir) / "pubchem_all.smiles"

    if smiles_path.exists():
        logger.info("Using cached PubChem SMILES: %s", smiles_path)
        return smiles_path

    if not gz_path.exists():
        logger.info("Downloading PubChem CID-SMILES (~3GB)...")
        urlretrieve(PUBCHEM_URL, gz_path)
        logger.info("Downloaded to %s", gz_path)

    logger.info("Extracting SMILES from CID-SMILES.gz...")
    count = 0
    with gzip.open(gz_path, "rt") as fin, open(smiles_path, "w") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                fout.write(parts[1] + "\n")
                count += 1
    logger.info("Extracted %d SMILES to %s", count, smiles_path)
    return smiles_path


def filter_smiles(smiles: list[str], max_len: int = 150) -> list[str]:
    """Filter SMILES matching CheMeleon's validation rules."""
    blocker = rdBase.BlockLogs()

    filtered = [s for s in smiles if len(s) <= max_len and "." not in s]
    logger.info("After length/mixture filter: %d / %d", len(filtered), len(smiles))

    valid = []
    for smi in filtered:
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            RemoveHs(mol, updateExplicitCount=True)
            valid.append(smi)
        except Exception:
            continue

    del blocker
    logger.info("After RDKit validation: %d / %d", len(valid), len(filtered))
    return valid


def load_and_filter_smiles(
    smiles_file: str | None = None,
    pubchem: bool = False,
    pubchem_sample: int | None = None,
    cache_dir: str = "/tmp/chemeleon_data",
) -> list[str]:
    """Load SMILES from a file or the PubChem export."""
    if smiles_file:
        logger.info("Loading SMILES from %s", smiles_file)
        with open(smiles_file) as f:
            smiles = [line.strip() for line in f if line.strip()]
    elif pubchem:
        path = download_pubchem(cache_dir)
        with open(path) as f:
            smiles = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Provide --smiles-file or --pubchem")

    if pubchem_sample and len(smiles) > pubchem_sample:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(smiles), pubchem_sample, replace=False)
        smiles = [smiles[i] for i in indices]
        logger.info("Sampled %d SMILES from %d total", pubchem_sample, len(smiles))

    return filter_smiles(smiles)


def compute_mordred_descriptors(
    smiles: list[str],
    cache_path: str | None = None,
    n_workers: int = 4,
) -> tuple[np.ndarray, list[int]]:
    """Compute 1,613 Mordred descriptors for a list of SMILES."""
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        logger.info("Loaded cached Mordred features from %s", cache_path)
        return data["features"], data["valid_indices"].tolist()

    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    calc.config(timeout=1)
    n_features = len(calc)
    logger.info("Computing %d Mordred descriptors for %d molecules...", n_features, len(smiles))

    blocker = rdBase.BlockLogs()
    batch_size = 1000
    all_features = []
    valid_indices = []

    for start in range(0, len(smiles), batch_size):
        end = min(start + batch_size, len(smiles))
        batch_smiles = smiles[start:end]

        mols = []
        mol_indices = []
        for i, smi in enumerate(batch_smiles):
            mol = MolFromSmiles(smi)
            if mol is not None:
                mol.SetProp("_Name", "")
                mols.append(mol)
                mol_indices.append(start + i)

        if not mols:
            continue

        df = calc.pandas(mols, quiet=True, nproc=min(n_workers, len(mols)))
        batch_features = df.fill_missing().to_numpy(dtype=np.float32)

        nan_mask = np.isnan(batch_features).any(axis=1)
        for i, (idx, has_nan) in enumerate(zip(mol_indices, nan_mask)):
            if not has_nan:
                all_features.append(batch_features[i])
                valid_indices.append(idx)

        if (start // batch_size) % 10 == 0:
            logger.info(
                "  %d / %d molecules processed (%d valid)...",
                end,
                len(smiles),
                len(valid_indices),
            )

    del blocker
    features = np.array(all_features, dtype=np.float32)
    logger.info("Mordred features: %s (%d valid / %d total)", features.shape, len(valid_indices), len(smiles))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, features=features, valid_indices=np.array(valid_indices))
        logger.info("Cached features to %s", cache_path)

    return features, valid_indices


class GraphBatcher:
    """Generate batches of cached molecular graphs with descriptor targets."""

    def __init__(
        self,
        cache: graph_cache.MolGraphCache,
        descriptors: np.ndarray,
        indices: np.ndarray,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 1701,
    ):
        self.cache = cache
        self.descriptors = descriptors
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self):
        return self.cache.count_batches(self.indices, self.batch_size)

    def __iter__(self):
        batch_seed = None
        if self.shuffle:
            batch_seed = self.seed + self._epoch
            self._epoch += 1

        for batch in self.cache.iter_batches_from_indices(
            self.indices,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=batch_seed,
        ):
            yield (
                mx.array(batch.V),
                mx.array(batch.E),
                mx.array(batch.edge_index.astype(np.int32, copy=False)),
                mx.array(batch.rev_edge_index.astype(np.int32, copy=False)),
                mx.array(batch.batch.astype(np.int32, copy=False)),
                batch.num_graphs,
                mx.array(self.descriptors[batch.graph_indices]),
            )


def align_smiles_to_cache(
    source_smiles: list[str],
    cached_smiles: list[str],
) -> np.ndarray:
    """Map cached graph order back to the descriptor-aligned SMILES order."""
    positions: dict[str, deque[int]] = defaultdict(deque)
    for idx, smi in enumerate(source_smiles):
        positions[smi].append(idx)

    aligned = []
    for smi in cached_smiles:
        if smi not in positions or not positions[smi]:
            raise ValueError(f"Could not align cached SMILES back to descriptors: {smi}")
        aligned.append(positions[smi].popleft())

    return np.array(aligned, dtype=np.int64)


def train_epoch(
    model: cs.CheMeleonSCORE,
    batcher: GraphBatcher,
    optimizer: optim.Optimizer,
) -> float:
    """Train one epoch with masked descriptor loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    def loss_fn(model, V, E, ei, rev, batch, ng, targets):
        scaled_targets = model._scale_targets(targets)
        preds = model(V, E, ei, rev, batch, ng)
        return cs.masked_mse_loss(preds, scaled_targets, model.masking_ratio)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for V, E, ei, rev, batch, ng, targets in batcher:
        loss, grads = loss_and_grad(model, V, E, ei, rev, batch, ng, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model: cs.CheMeleonSCORE, batcher: GraphBatcher) -> float:
    """Validate with full (unmasked) MSE loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for V, E, ei, rev, batch, ng, targets in batcher:
        scaled_targets = model._scale_targets(targets)
        preds = model(V, E, ei, rev, batch, ng)
        loss = cs.full_mse_loss(preds, scaled_targets)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(model: cs.CheMeleonSCORE, path: str, config: dict):
    """Save model weights and config."""
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(path, **weights)

    config_path = path.replace(".npz", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    stats_path = path.replace(".npz", "_stats.npz")
    np.savez(
        stats_path,
        means=np.array(model.feature_means),
        vars=np.array(model.feature_vars),
    )


def load_checkpoint(path: str) -> cs.CheMeleonSCORE:
    """Load a CheMeleonSCORE from checkpoint."""
    config_path = path.replace(".npz", "_config.json")
    with open(config_path) as f:
        config = json.load(f)

    model = cs.CheMeleonSCORE(**config)
    model.load_weights(list(mx.load(path).items()))

    stats_path = path.replace(".npz", "_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        model.set_feature_stats(mx.array(stats["means"]), mx.array(stats["vars"]))

    return model


def main():
    parser = argparse.ArgumentParser(description="Pretrain CheMeleon-SCORE")

    data = parser.add_argument_group("data")
    data.add_argument("--smiles-file", type=str, default=None)
    data.add_argument("--pubchem", action="store_true")
    data.add_argument("--pubchem-sample", type=int, default=None)
    data.add_argument("--mordred-cache", type=str, default=None)
    data.add_argument("--n-workers", type=int, default=4)
    data.add_argument("--graph-cache-dir", type=str, default=None)
    data.add_argument("--graph-cache-shard-size", type=int, default=graph_cache.DEFAULT_SHARD_SIZE)
    data.add_argument("--rebuild-graph-cache", action="store_true")

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--d-h", type=int, default=2048)
    arch.add_argument("--depth", type=int, default=6)
    arch.add_argument("--readout", type=str, default="set2set", choices=["mean", "set2set"])
    arch.add_argument("--set2set-iters", type=int, default=6)
    arch.add_argument("--score-dim", type=int, default=2048)
    arch.add_argument("--score-steps", type=int, default=6)
    arch.add_argument("--skip-alpha", type=float, default=0.5)
    arch.add_argument("--masking-ratio", type=float, default=0.15)
    arch.add_argument("--dropout", type=float, default=0.0)

    train_g = parser.add_argument_group("training")
    train_g.add_argument("--batch-size", type=int, default=128)
    train_g.add_argument("--epochs", type=int, default=500)
    train_g.add_argument("--lr", type=float, default=1e-3)
    train_g.add_argument("--patience", type=int, default=50)
    train_g.add_argument("--val-fraction", type=float, default=0.2)
    train_g.add_argument("--test-fraction", type=float, default=0.1)
    train_g.add_argument("--seed", type=int, default=1701)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"runs/pretrain_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=== CheMeleon-SCORE Pretraining ===")
    logger.info("Output: %s", args.output_dir)

    smiles = load_and_filter_smiles(
        smiles_file=args.smiles_file,
        pubchem=args.pubchem,
        pubchem_sample=args.pubchem_sample,
    )
    logger.info("Filtered SMILES: %d", len(smiles))

    descriptors, valid_indices = compute_mordred_descriptors(
        smiles,
        cache_path=args.mordred_cache,
        n_workers=args.n_workers,
    )
    valid_smiles = [smiles[i] for i in valid_indices]
    logger.info("Valid molecules with descriptors: %d, features: %s", len(valid_smiles), descriptors.shape)

    cache_dir = args.graph_cache_dir or os.path.join(args.output_dir, "graph_cache")
    mol_graph_cache = graph_cache.load_or_build_graph_cache(
        valid_smiles,
        cache_dir,
        shard_size=args.graph_cache_shard_size,
        rebuild=args.rebuild_graph_cache,
        log=logger.info,
    )
    cache_smiles = mol_graph_cache.get_smiles()
    if mol_graph_cache.n_graphs != len(valid_smiles):
        logger.info(
            "Graph cache kept %d / %d descriptor-valid molecules; aligning targets",
            mol_graph_cache.n_graphs,
            len(valid_smiles),
        )
        aligned_idx = align_smiles_to_cache(valid_smiles, cache_smiles)
        descriptors = descriptors[aligned_idx]

    rng = np.random.RandomState(args.seed)
    n = mol_graph_cache.n_graphs
    indices = rng.permutation(n)
    n_test = int(n * args.test_fraction)
    n_val = int(n * args.val_fraction)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_descs = descriptors[train_idx]
    logger.info("Split: train=%d, val=%d, test=%d", n_train, len(val_idx), len(test_idx))

    means = np.nan_to_num(np.nanmean(train_descs, axis=0), nan=0.0)
    vars_ = np.nan_to_num(np.nanvar(train_descs, axis=0), nan=1.0)
    vars_ = np.maximum(vars_, 1e-8)

    model_config = {
        "d_v": 72,
        "d_e": 14,
        "d_h": args.d_h,
        "depth": args.depth,
        "readout": args.readout,
        "set2set_iters": args.set2set_iters,
        "score_dim": args.score_dim,
        "score_steps": args.score_steps,
        "skip_alpha": args.skip_alpha,
        "n_descriptors": descriptors.shape[1],
        "masking_ratio": args.masking_ratio,
        "dropout": args.dropout,
    }

    if args.resume:
        model = load_checkpoint(args.resume)
        logger.info("Resumed from %s", args.resume)
    else:
        model = cs.CheMeleonSCORE(**model_config)

    model.set_feature_stats(mx.array(means.astype(np.float32)), mx.array(vars_.astype(np.float32)))

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    logger.info("Model parameters: %s", f"{n_params:,}")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({**model_config, **vars(args)}, f, indent=2, default=str)

    train_batcher = GraphBatcher(
        mol_graph_cache,
        descriptors,
        train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_batcher = GraphBatcher(
        mol_graph_cache,
        descriptors,
        val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_batcher = GraphBatcher(
        mol_graph_cache,
        descriptors,
        test_idx,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    n_steps_per_epoch = max(len(train_batcher), 1)
    scheduler = optim.cosine_decay(args.lr, args.epochs * n_steps_per_epoch)
    optimizer = optim.Adam(learning_rate=scheduler)

    best_val = float("inf")
    best_epoch = 0
    patience_ctr = 0
    best_path = os.path.join(args.output_dir, "best.npz")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_batcher, optimizer)
        val_loss = validate(model, val_batcher)

        logger.info(
            "Epoch %03d/%03d  train=%.6f  val=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_ctr = 0
            save_checkpoint(model, best_path, model_config)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Best validation loss %.6f at epoch %d", best_val, best_epoch)
    model = load_checkpoint(best_path)
    test_loss = validate(model, test_batcher)
    logger.info("Test loss: %.6f", test_loss)


if __name__ == "__main__":
    main()
