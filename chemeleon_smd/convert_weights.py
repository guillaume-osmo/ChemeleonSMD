"""Download and convert pretrained CheMeleon PyTorch weights to MLX.

Downloads the CheMeleon checkpoint from Zenodo and converts it to MLX format.

Usage:
    python -m chemeleon_smd.convert_weights
    python -m chemeleon_smd.convert_weights --output my_weights.npz
"""

import argparse
import json
import logging
from pathlib import Path
from urllib.request import urlretrieve

import mlx.core as mx
import numpy as np

from chemeleon_smd import mpnn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
WEIGHTS_DIR = Path(__file__).parent / "weights"


def download_checkpoint() -> Path:
    """Download the CheMeleon pretrained weights from Zenodo."""
    cache_dir = Path.home() / ".chemprop"
    cache_dir.mkdir(exist_ok=True)
    ckpt_path = cache_dir / "chemeleon_mp.pt"
    if not ckpt_path.exists():
        logger.info("Downloading CheMeleon weights from %s...", ZENODO_URL)
        urlretrieve(ZENODO_URL, ckpt_path)
        logger.info("Saved to %s", ckpt_path)
    else:
        logger.info("Using cached weights at %s", ckpt_path)
    return ckpt_path


def convert_checkpoint(output_dir: Path | None = None) -> Path:
    """Download, convert, and save MLX weights. Returns path to .npz file."""
    import torch

    out_dir = output_dir or WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "chemeleon_mpnn.npz"
    config_path = out_dir / "chemeleon_mpnn_config.json"

    if npz_path.exists() and config_path.exists():
        logger.info("MLX weights already exist at %s", npz_path)
        return npz_path

    ckpt_path = download_checkpoint()
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    sd = {k: v.numpy() for k, v in ckpt["state_dict"].items()}

    weight_map = {
        "W_i.weight": "mpnn.W_i.weight",
        "W_h.weight": "mpnn.W_h.weight",
        "W_o.weight": "mpnn.W_o.weight",
        "W_o.bias": "mpnn.W_o.bias",
    }

    mlx_weights = {}
    for torch_key, np_val in sd.items():
        mlx_key = weight_map.get(torch_key)
        if mlx_key is None:
            logger.warning("Skipping unknown weight: %s", torch_key)
            continue
        mlx_weights[mlx_key] = mx.array(np_val.astype(np.float32))
        logger.info("  %s -> %s  shape=%s", torch_key, mlx_key, np_val.shape)

    mx.savez(str(npz_path), **mlx_weights)
    logger.info("Saved MLX weights to %s", npz_path)

    config = {
        "d_v": hp["d_v"],
        "d_e": hp["d_e"],
        "d_h": hp["d_h"],
        "depth": hp["depth"],
        "dropout": hp["dropout"],
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to %s", config_path)

    return npz_path


def load_teacher(
    weights_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> mpnn.CheMeleonBondMPNN:
    """Load pretrained CheMeleon DMPNN from MLX weights.

    If weights don't exist locally, downloads and converts from Zenodo.
    """
    w_path = Path(weights_path) if weights_path else WEIGHTS_DIR / "chemeleon_mpnn.npz"
    c_path = (
        Path(config_path) if config_path else w_path.with_suffix("").with_name(
            w_path.stem.replace(".npz", "") + "_config.json"
        )
    )

    if not w_path.exists():
        logger.info("Weights not found, downloading and converting...")
        convert_checkpoint(w_path.parent)

    c_path = w_path.parent / "chemeleon_mpnn_config.json"
    with open(c_path) as f:
        config = json.load(f)

    model = mpnn.CheMeleonBondMPNN(**config)
    raw_weights = mx.load(str(w_path))
    clean = {k.replace("mpnn.", ""): v for k, v in raw_weights.items()}
    model.load_weights(list(clean.items()))
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert CheMeleon weights to MLX")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else None
    path = convert_checkpoint(out_dir)
    logger.info("Done. Weights at %s", path)


if __name__ == "__main__":
    main()
