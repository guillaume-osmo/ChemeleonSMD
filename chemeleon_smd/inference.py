"""High-level inference API for ChemeleonSMD fingerprints."""

from pathlib import Path
from typing import List, Literal, Optional, Union

import mlx.core as mx
import numpy as np
from mlx_graphs.utils import scatter

from chemeleon_smd import mol_featurizer as mf
from chemeleon_smd import score_dmpnn

WEIGHTS_DIR = Path(__file__).parent / "weights"

_CACHED_MODEL: Optional[score_dmpnn.ScoreDMPNN] = None


def load_model(
    weights_path: Optional[Union[str, Path]] = None,
) -> score_dmpnn.ScoreDMPNN:
    """Load the distilled SCORE-DMPNN model.

    If no weights_path is given, looks for the bundled v3 weights, then v2.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and weights_path is None:
        return _CACHED_MODEL

    if weights_path is None:
        for name in [
            "score_dmpnn_distilled_v6.npz",
            "score_dmpnn_distilled_v5.npz",
            "score_dmpnn_distilled_v4.npz",
            "score_dmpnn_distilled_v3.npz",
        ]:
            candidate = WEIGHTS_DIR / name
            if candidate.exists():
                weights_path = candidate
                break

    if weights_path is None or not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No distilled weights found. Expected in {WEIGHTS_DIR}. "
            "See README.md for instructions on downloading or training weights."
        )

    model = score_dmpnn.ScoreDMPNN(
        d_v=72, d_e=14, d_h=2048, n_steps=6, skip_alpha=0.5, dropout=0.0
    )
    weights = list(mx.load(str(weights_path)).items())
    model.load_weights(weights)
    model.eval()

    if weights_path is None:
        _CACHED_MODEL = model

    return model


def fingerprint(
    smiles: Union[str, List[str]],
    model: Optional[score_dmpnn.ScoreDMPNN] = None,
    readout: Literal["mean", "molattfp"] = "mean",
    batch_size: int = 64,
) -> mx.array:
    """Compute 2048-dim molecular fingerprints for one or more SMILES.

    Args:
        smiles: A single SMILES string or a list of SMILES strings.
        model: Pre-loaded model. If None, loads the default bundled weights.
        readout: Pooling strategy for atom -> molecule fingerprints.
            "mean" (default): mean pooling over atoms. This is what the model
                was distilled against, so it matches the teacher exactly.
            "molattfp": MolAttFPReadout (AttentiveFP-style GRU + attention).
                Experimental alternative that may capture richer structural
                information, but its attention weights are randomly initialized
                (not trained). Fine-tune before using in production.
        batch_size: Number of molecules per forward pass.

    Returns:
        mx.array of shape (N, 2048) where N is the number of valid molecules.
        Invalid SMILES are silently skipped (rows will be fewer than input).
    """
    if model is None:
        model = load_model()

    if isinstance(smiles, str):
        smiles = [smiles]

    att_readout = None
    if readout == "molattfp":
        att_readout = score_dmpnn.MolAttFPReadout(hidden_dim=model.d_h, num_steps=2)
        att_readout.eval()

    all_fps = []

    for start in range(0, len(smiles), batch_size):
        batch_smiles = smiles[start : start + batch_size]
        graphs = []
        for smi in batch_smiles:
            g = mf.featurize_smiles(smi)
            if g is not None and g.V.shape[0] > 0 and g.E.shape[0] > 0:
                graphs.append(g)

        if not graphs:
            continue

        V, E, ei, rev, batch_arr, ng = mf.collate_mol_graphs(graphs)
        V_mx = mx.array(V)
        E_mx = mx.array(E)
        ei_mx = mx.array(ei.astype(np.int32))
        rev_mx = mx.array(rev.astype(np.int32))
        batch_mx = mx.array(batch_arr.astype(np.int32))

        H_v = model(V_mx, E_mx, ei_mx, rev_mx)

        if att_readout is not None:
            fp = att_readout(H_v, batch_mx, ng)
        else:
            fp = scatter(H_v, batch_mx, out_size=ng, aggr="mean")

        fp = mx.stop_gradient(fp)
        mx.eval(fp)
        all_fps.append(fp)

    if not all_fps:
        return mx.zeros((0, model.d_h))

    return mx.concatenate(all_fps, axis=0)
