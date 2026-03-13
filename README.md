# ChemeleonSMD

**SCORE MLX Distilled** CheMeleon molecular fingerprints — fast 2048-dim molecular representations on Apple Silicon.

ChemeleonSMD distills the [CheMeleon](https://zenodo.org/records/15460715) pretrained Directed Message Passing Neural Network (DMPNN) into a [SCORE](https://arxiv.org/abs/2603.10544)-style architecture running natively on [MLX](https://github.com/ml-explore/mlx). The SCORE Euler skip connection replaces ChemProp's hard-overwrite message passing with provably contractive dynamics, distilled using only **~82K molecules** from CheMeleon's 944K PubChem training set (mean cosine similarity **0.991** on the full 944K).

## Key Features

- **2048-dim molecular fingerprints** from a pretrained graph neural network
- **Apple Silicon native** — runs on MLX, no CUDA required
- **Contractive SCORE dynamics** — Euler skip connection (α=0.5, fixed) guarantees stable convergence, unlike ChemProp's hard-overwrite iterations
- **ChemProp-compatible** featurization (72-dim atoms, 14-dim bonds)
- **Distilled from CheMeleon** — iteratively refined with scaffold-diverse hard case mining on PubChem

## Installation

```bash
git clone https://github.com/guillaume-osmo/ChemeleonSMD.git
cd ChemeleonSMD

# Fetch the model weights (stored with Git LFS)
git lfs install
git lfs pull

pip install -e .
```

> **Note:** The `.npz` weight files are stored with [Git LFS](https://git-lfs.com/).
> If you cloned without LFS installed, the weight files will be small pointer files.
> Run `git lfs install && git lfs pull` to download the actual weights.

### Dependencies

- `mlx` — Apple's ML framework
- `mlx-graphs` — Graph neural network operations for MLX
- `rdkit` — Molecular featurization
- `numpy`

### Optional (for weight conversion from PyTorch)

```bash
pip install torch
```

## Quick Start

```python
from chemeleon_smd import fingerprint

# Single molecule
fp = fingerprint("CCO")
print(fp.shape)  # (1, 2048)

# Batch of molecules
fps = fingerprint(["CCO", "c1ccccc1", "CC(=O)O"])
print(fps.shape)  # (3, 2048)
```

## Architecture

Both the teacher and student share `W_h` across message passing iterations — this is how ChemProp was designed ([CheMeleon paper](https://zenodo.org/records/15460715): "8.7 million parameters" for the message passing network at depth=6). The difference is in the **forward dynamics**, not the parameter count.

### Teacher: CheMeleon DMPNN (hard overwrite)

```
H_t = ReLU(H_0 + W_h · M_t)    for t = 1..5
```

Each step **completely replaces** the previous hidden state. Information from earlier steps only survives indirectly through neighbor messages. This can oscillate or lose signal at depth.

### Student: SCORE-DMPNN (Euler skip connection)

```
H_new = ReLU(H_0 + W_h · M_t)
H_t   = 0.5 · H_{t-1} + 0.5 · H_new    for t = 1..5
```

Each step **blends** 50% of the previous state with 50% of the new computation. The [SCORE paper (arXiv:2603.10544)](https://arxiv.org/abs/2603.10544) proves this Euler integration makes the recurrence **contractive** — the system is guaranteed to converge rather than oscillate. `α=0.5` is fixed, not trained.

The mean-pooled atom representations produce 2048-dim molecular fingerprints.

### Parameters

Both models have **identical parameter counts** (8,714,240 = 8.7M) since ChemProp already shares `W_h`:

| Weight | Shape | Parameters |
|---|---|---|
| `W_i` (input projection) | 2048 × 86 | 176,128 |
| `W_h` (message passing, shared) | 2048 × 2048 | 4,194,304 |
| `W_o` (output projection + bias) | 2048 × 2120 + 2048 | 4,343,808 |
| **Total** | | **8,714,240** |

The SCORE contribution is not parameter reduction — it is **replacing hard-overwrite dynamics with provably contractive Euler integration**, yielding more stable message passing with the same weights.

### Distillation

The SCORE-DMPNN student was distilled to reproduce the teacher's 2048-dim fingerprints (MSE loss) through iterative hard case mining. Each round evaluates the current student on a large molecule pool, identifies molecules with low cosine similarity (< 0.98), selects scaffold-diverse subsets, and retrains:

| Version | Training strategy | Total train | Epochs | Source |
|---|---|---|---|---|
| **v3** | 10K seed + 10K OOD (100K pool) + 10K hard (diverse pool) | 31K | 50 | diverse molecule pools |
| **v4** | v3 + 10K scaffold-diverse hard cases | 40K | 50 | PubChem 944K |
| **v5** | v4 + 20K scaffold-diverse hard cases | 50K | 20 | PubChem 944K |
| **v6** | v5 + all remaining OOD cases | 82K | 10 | PubChem 944K |

Only **~82K molecules** (8.7% of PubChem's 944K) were needed to distill the full CheMeleon model into SCORE dynamics.

### Evaluation on PubChem (944K molecules)

Cosine similarity between teacher (CheMeleon) and SCORE-DMPNN student fingerprints, evaluated on CheMeleon's full PubChem training set:

| Version | Mean cos | < 0.98 | < 0.95 | < 0.90 | P01 |
|---|---|---|---|---|---|
| v3 | 0.9865 | 152,615 (16.2%) | 20,483 (2.17%) | 3,694 (0.39%) | 0.930 |
| v4 | 0.9892 | 75,517 (8.0%) | 8,285 (0.88%) | 1,291 (0.14%) | 0.953 |
| v5 | 0.9905 | 37,037 (3.9%) | 3,018 (0.32%) | 442 (0.05%) | 0.967 |
| **v6** | **0.9910** | **17,324 (1.8%)** | **555 (0.06%)** | **53 (0.01%)** | **0.977** |

From v3 to v6: OOD (< 0.98) dropped **88%**, molecules below 0.95 dropped **97%**, and below 0.90 dropped **99%**. The default shipped model is **v6**.

### Finetuning: Lipophilicity Benchmark

Finetuning on MoleculeNet [Lipophilicity](https://moleculenet.org/) (4,200 molecules, 80/10/10 random split) with a 2-layer FFN head (300 hidden, LeakyReLU):

| Model | RMSE | MAE |
|---|---|---|
| CheMeleon teacher (frozen) | 0.619 | 0.478 |
| CheMeleon teacher (finetuned) | 0.538 | 0.394 |
| SCORE-DMPNN v6 (frozen) | 0.623 | 0.455 |
| **SCORE-DMPNN v4 (finetuned)** | **0.518** | **0.380** |
| SCORE-DMPNN v6 (finetuned) | 0.519 | 0.385 |

SCORE-DMPNN matches or slightly outperforms the original CheMeleon teacher on downstream finetuning, while providing contractive stability guarantees.

## Weight Conversion

To convert from the original PyTorch CheMeleon checkpoint:

```bash
python -m chemeleon_smd.convert_weights
```

This downloads the checkpoint from Zenodo and saves MLX weights to `chemeleon_smd/weights/`.

## Project Structure

```
chemeleon_smd/
├── __init__.py           # Public API
├── inference.py          # High-level fingerprint() function
├── mol_featurizer.py     # ChemProp-compatible atom/bond featurization
├── mpnn.py               # Teacher model (CheMeleonBondMPNN)
├── score_dmpnn.py        # Student model (ScoreDMPNN + MolAttFPReadout)
├── convert_weights.py    # PyTorch → MLX weight conversion
└── weights/
    ├── chemeleon_mpnn.npz           # Teacher weights (MLX)
    ├── chemeleon_mpnn_config.json   # Model hyperparameters
    ├── score_dmpnn_distilled_v3.npz # Distilled student v3
    ├── score_dmpnn_distilled_v4.npz # Distilled student v4
    ├── score_dmpnn_distilled_v5.npz # Distilled student v5
    └── score_dmpnn_distilled_v6.npz # Distilled student v6 (default)
```

## License

MIT

## References

If you use ChemeleonSMD, please cite:

```bibtex
@article{godin2026score,
  title={SCORE: Replacing Layer Stacking with Contractive Recurrent Depth},
  author={Godin, Guillaume},
  journal={arXiv preprint arXiv:2603.10544},
  year={2026},
  url={https://arxiv.org/abs/2603.10544},
}

@misc{chemeleon2024,
  title={CheMeleon: A Foundation Model for Molecular Property Prediction},
  url={https://zenodo.org/records/15460715},
}
```

- **SCORE** — [arXiv:2603.10544](https://arxiv.org/abs/2603.10544): The contractive recurrent depth method (Euler skip connections) used in the SCORE-DMPNN student architecture.
- **CheMeleon** — [Zenodo](https://zenodo.org/records/15460715): The pretrained DMPNN teacher model.
