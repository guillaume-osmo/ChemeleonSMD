"""ChemeleonSMD: SCORE MLX Distilled CheMeleon molecular fingerprints.

A lightweight, Apple-Silicon-native molecular fingerprinting model based on
CheMeleon's Directed Message Passing Neural Network (DMPNN), distilled into
a SCORE-style architecture running on MLX.

Usage:
    from chemeleon_smd import fingerprint

    fps = fingerprint(["CCO", "c1ccccc1", "CC(=O)O"])
    print(fps.shape)  # (3, 2048)
"""

__version__ = "0.1.0"

from chemeleon_smd.inference import fingerprint

__all__ = ["fingerprint", "__version__"]
