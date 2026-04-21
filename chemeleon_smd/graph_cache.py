"""Persistent sharded molecular graph cache for MLX training loops.

The goal is to pay the RDKit/featurization cost once, then reuse cached graph
arrays across epochs and reruns. The cache stores valid molecular graphs in
shards of concatenated NumPy arrays and reconstructs batched graphs on demand.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from chemeleon_smd import mol_featurizer as mf

CACHE_VERSION = 1
DEFAULT_SHARD_SIZE = 2048


def _noop(_: str) -> None:
    return None


def smiles_sha256(smiles_list: Sequence[str]) -> str:
    """Hash a SMILES sequence for cache validation."""
    digest = hashlib.sha256()
    for smi in smiles_list:
        digest.update(smi.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_valid_graph(graph: mf.MolGraphData | None) -> bool:
    return graph is not None and graph.V.shape[0] > 0 and graph.E.shape[0] > 0


@dataclass(frozen=True)
class CachedMolGraphBatch:
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    batch: np.ndarray
    num_graphs: int
    graph_indices: np.ndarray
    smiles: list[str]


class MolGraphCache:
    """Read batches from an on-disk sharded molecular graph cache."""

    def __init__(self, cache_dir: str):
        self.cache_dir = str(Path(cache_dir))
        meta_path = Path(self.cache_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Graph cache metadata missing: {meta_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)

        if self.meta.get("version") != CACHE_VERSION:
            raise ValueError(
                f"Unsupported graph cache version {self.meta.get('version')} "
                f"(expected {CACHE_VERSION})"
            )

        self._shard_files = [
            Path(self.cache_dir) / shard["file"]
            for shard in self.meta["shards"]
        ]
        self._shard_counts = np.array(
            [int(shard["n_graphs"]) for shard in self.meta["shards"]],
            dtype=np.int64,
        )
        if len(self._shard_counts) > 0:
            self._shard_starts = np.concatenate(
                [np.array([0], dtype=np.int64), np.cumsum(self._shard_counts[:-1])]
            )
        else:
            self._shard_starts = np.zeros(0, dtype=np.int64)
        self._shard_ends = self._shard_starts + self._shard_counts
        self._smiles_cache: list[str] | None = None

    @property
    def n_graphs(self) -> int:
        return int(self.meta["n_valid_graphs"])

    @property
    def n_shards(self) -> int:
        return len(self._shard_files)

    def get_smiles(self) -> list[str]:
        """Return cached valid SMILES in graph order."""
        if self._smiles_cache is None:
            smiles: list[str] = []
            for shard_file in self._shard_files:
                with np.load(shard_file, allow_pickle=False) as shard:
                    smiles.extend(shard["smiles"].astype(str).tolist())
            self._smiles_cache = smiles
        return list(self._smiles_cache)

    def count_batches(self, indices: Sequence[int], batch_size: int) -> int:
        """Return the actual number of batches yielded for a subset."""
        indices_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices_arr.size == 0:
            return 0
        shard_ids = np.searchsorted(self._shard_ends, indices_arr, side="right")
        counts = np.bincount(shard_ids, minlength=self.n_shards)
        return int(sum(math.ceil(int(c) / batch_size) for c in counts if c > 0))

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[CachedMolGraphBatch]:
        """Iterate over all cached graphs."""
        yield from self.iter_batches_from_indices(
            np.arange(self.n_graphs, dtype=np.int64),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def iter_batches_from_indices(
        self,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[CachedMolGraphBatch]:
        """Iterate over a subset of cached graphs grouped by shard."""
        indices_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices_arr.size == 0:
            return

        if np.any(indices_arr < 0) or np.any(indices_arr >= self.n_graphs):
            raise IndexError("Graph cache batch indices out of range")

        rng = np.random.RandomState(seed) if shuffle else None
        grouped = self._group_indices_by_shard(indices_arr, rng=rng)

        for shard_id, local_indices in grouped:
            with np.load(self._shard_files[shard_id], allow_pickle=False) as shard:
                for start in range(0, len(local_indices), batch_size):
                    chunk = local_indices[start : start + batch_size]
                    yield self._collate_batch(
                        shard=shard,
                        shard_id=shard_id,
                        local_indices=chunk,
                    )

    def _group_indices_by_shard(
        self,
        indices: np.ndarray,
        rng: np.random.RandomState | None,
    ) -> list[tuple[int, np.ndarray]]:
        shard_ids = np.searchsorted(self._shard_ends, indices, side="right")

        grouped: dict[int, list[int]] = {}
        for global_idx, shard_id in zip(indices.tolist(), shard_ids.tolist()):
            local_idx = global_idx - int(self._shard_starts[shard_id])
            grouped.setdefault(shard_id, []).append(local_idx)

        shard_order = list(grouped.keys())
        if rng is not None:
            rng.shuffle(shard_order)

        result: list[tuple[int, np.ndarray]] = []
        for shard_id in shard_order:
            local = np.array(grouped[shard_id], dtype=np.int64)
            if rng is not None:
                rng.shuffle(local)
            result.append((shard_id, local))
        return result

    def _collate_batch(
        self,
        shard: np.lib.npyio.NpzFile,
        shard_id: int,
        local_indices: np.ndarray,
    ) -> CachedMolGraphBatch:
        V_all = shard["V"]
        E_all = shard["E"]
        edge_src_all = shard["edge_src"]
        edge_dst_all = shard["edge_dst"]
        rev_all = shard["rev_edge_index"]
        atom_offsets = shard["atom_offsets"]
        edge_offsets = shard["edge_offsets"]
        shard_smiles = shard["smiles"].astype(str)

        V_parts = []
        E_parts = []
        src_parts = []
        dst_parts = []
        rev_parts = []
        batch_parts = []
        smiles = []

        atom_offset = 0
        edge_offset = 0

        for batch_pos, local_idx in enumerate(local_indices.tolist()):
            a0 = int(atom_offsets[local_idx])
            a1 = int(atom_offsets[local_idx + 1])
            e0 = int(edge_offsets[local_idx])
            e1 = int(edge_offsets[local_idx + 1])

            V_parts.append(V_all[a0:a1])
            E_parts.append(E_all[e0:e1])
            src_parts.append(edge_src_all[e0:e1] + atom_offset)
            dst_parts.append(edge_dst_all[e0:e1] + atom_offset)
            rev_parts.append(rev_all[e0:e1] + edge_offset)
            batch_parts.append(np.full(a1 - a0, batch_pos, dtype=np.int32))
            smiles.append(str(shard_smiles[local_idx]))

            atom_offset += a1 - a0
            edge_offset += e1 - e0

        V = np.concatenate(V_parts, axis=0)
        E = np.concatenate(E_parts, axis=0)
        edge_index = np.stack(
            [np.concatenate(src_parts), np.concatenate(dst_parts)],
            axis=0,
        )
        rev_edge_index = np.concatenate(rev_parts)
        batch = np.concatenate(batch_parts)
        global_indices = self._shard_starts[shard_id] + local_indices

        return CachedMolGraphBatch(
            V=V,
            E=E,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            batch=batch,
            num_graphs=len(local_indices),
            graph_indices=global_indices.astype(np.int64, copy=False),
            smiles=smiles,
        )


def load_or_build_graph_cache(
    smiles_list: Sequence[str],
    cache_dir: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
    rebuild: bool = False,
    log: Callable[[str], None] | None = None,
) -> MolGraphCache:
    """Load an existing graph cache or build it from SMILES."""
    log = log or _noop
    cache_path = Path(cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_hash = smiles_sha256(smiles_list)
    if not rebuild and _cache_matches(
        cache_path, smiles_hash=smiles_hash, n_input_smiles=len(smiles_list)
    ):
        log(f"Using cached molecular graphs from {cache_path}")
        return MolGraphCache(str(cache_path))

    _build_graph_cache(
        smiles_list=smiles_list,
        cache_path=cache_path,
        shard_size=shard_size,
        smiles_hash=smiles_hash,
        log=log,
    )
    return MolGraphCache(str(cache_path))


def _cache_matches(
    cache_path: Path,
    smiles_hash: str,
    n_input_smiles: int,
) -> bool:
    meta_path = cache_path / "meta.json"
    if not meta_path.exists():
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    return (
        meta.get("version") == CACHE_VERSION
        and meta.get("smiles_sha256") == smiles_hash
        and int(meta.get("n_input_smiles", -1)) == int(n_input_smiles)
    )


def _build_graph_cache(
    smiles_list: Sequence[str],
    cache_path: Path,
    shard_size: int,
    smiles_hash: str,
    log: Callable[[str], None],
) -> None:
    if shard_size <= 0:
        raise ValueError("graph cache shard size must be positive")

    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"{cache_path.name}.tmp.", dir=str(cache_path.parent))
    )
    shard_entries: list[dict[str, int | str]] = []
    shard_graphs: list[mf.MolGraphData] = []
    shard_smiles: list[str] = []

    n_valid = 0
    total_atoms = 0
    total_edges = 0
    progress_every = max(shard_size, 10_000)

    try:
        log(
            f"Building molecular graph cache at {cache_path} "
            f"from {len(smiles_list)} SMILES..."
        )

        for idx, smi in enumerate(smiles_list, start=1):
            graph = mf.featurize_smiles(smi)
            if _is_valid_graph(graph):
                shard_graphs.append(graph)
                shard_smiles.append(smi)
                n_valid += 1
                total_atoms += int(graph.V.shape[0])
                total_edges += int(graph.E.shape[0])

            if len(shard_graphs) >= shard_size:
                shard_entries.append(
                    _write_shard(
                        cache_dir=tmp_dir,
                        shard_idx=len(shard_entries),
                        graphs=shard_graphs,
                        smiles=shard_smiles,
                    )
                )
                shard_graphs = []
                shard_smiles = []

            if idx % progress_every == 0:
                log(f"  {idx}/{len(smiles_list)} SMILES processed ({n_valid} valid)")

        if shard_graphs:
            shard_entries.append(
                _write_shard(
                    cache_dir=tmp_dir,
                    shard_idx=len(shard_entries),
                    graphs=shard_graphs,
                    smiles=shard_smiles,
                )
            )

        if n_valid == 0:
            raise ValueError("No valid molecular graphs were produced from the input SMILES")

        meta = {
            "version": CACHE_VERSION,
            "smiles_sha256": smiles_hash,
            "n_input_smiles": len(smiles_list),
            "n_valid_graphs": n_valid,
            "n_shards": len(shard_entries),
            "shard_size": shard_size,
            "total_atoms": total_atoms,
            "total_edges": total_edges,
            "shards": shard_entries,
        }
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.move(str(tmp_dir), str(cache_path))
        log(
            f"Built molecular graph cache: {cache_path} "
            f"({n_valid} valid graphs across {len(shard_entries)} shards)"
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _write_shard(
    cache_dir: Path,
    shard_idx: int,
    graphs: list[mf.MolGraphData],
    smiles: list[str],
) -> dict[str, int | str]:
    atom_offsets = [0]
    edge_offsets = [0]
    V_parts = []
    E_parts = []
    edge_src_parts = []
    edge_dst_parts = []
    rev_parts = []

    for graph in graphs:
        V_parts.append(graph.V.astype(np.float32, copy=False))
        E_parts.append(graph.E.astype(np.float32, copy=False))
        edge_src_parts.append(graph.edge_index[0].astype(np.int32, copy=False))
        edge_dst_parts.append(graph.edge_index[1].astype(np.int32, copy=False))
        rev_parts.append(graph.rev_edge_index.astype(np.int32, copy=False))
        atom_offsets.append(atom_offsets[-1] + int(graph.V.shape[0]))
        edge_offsets.append(edge_offsets[-1] + int(graph.E.shape[0]))

    shard_file = cache_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(
        shard_file,
        V=np.concatenate(V_parts, axis=0),
        E=np.concatenate(E_parts, axis=0),
        edge_src=np.concatenate(edge_src_parts, axis=0),
        edge_dst=np.concatenate(edge_dst_parts, axis=0),
        rev_edge_index=np.concatenate(rev_parts, axis=0),
        atom_offsets=np.array(atom_offsets, dtype=np.int64),
        edge_offsets=np.array(edge_offsets, dtype=np.int64),
        smiles=np.array(smiles, dtype=f"<U{max(len(s) for s in smiles)}"),
    )
    return {"file": shard_file.name, "n_graphs": len(graphs)}
