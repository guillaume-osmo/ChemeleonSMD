"""MLX layers for descriptor pretraining and SCORE-style refinement."""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


class Periodic(nn.Module):
    """Periodic activation: x -> 2*pi*weight*x -> [cos, sin]."""

    def __init__(self, n_features: int, k: int = 48, sigma: float = 0.01):
        super().__init__()
        bound = sigma * 3
        self.weight = mx.random.truncated_normal(
            lower=mx.array(-bound / sigma),
            upper=mx.array(bound / sigma),
            shape=(n_features, k),
        ) * sigma

    def __call__(self, x: mx.array) -> mx.array:
        x = 2 * math.pi * self.weight * x[..., None]
        return mx.concatenate([mx.cos(x), mx.sin(x)], axis=-1)


class NLinear(nn.Module):
    """N separate linear layers for N feature embeddings."""

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        d_in_rsqrt = in_features ** -0.5
        self.weight = mx.random.uniform(
            low=-d_in_rsqrt, high=d_in_rsqrt, shape=(n, in_features, out_features)
        )
        self.bias = (
            mx.random.uniform(low=-d_in_rsqrt, high=d_in_rsqrt, shape=(n, out_features))
            if bias
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, axes=(1, 0, 2))
        x = x @ self.weight
        x = mx.transpose(x, axes=(1, 0, 2))
        if self.bias is not None:
            x = x + self.bias
        return x


class PeriodicEmbeddings(nn.Module):
    """Embeddings for continuous features via periodic activations."""

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool = False,
    ):
        super().__init__()
        self.periodic = Periodic(n_features, n_frequencies, frequency_init_scale)
        if lite:
            if not activation:
                raise ValueError("lite=True requires activation=True")
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.linear = NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU() if activation else None

    def __call__(self, x: mx.array) -> mx.array:
        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class WinsorizeStdevN(nn.Module):
    """Clamp values to [-n, +n] standard deviations."""

    def __init__(self, n: float):
        super().__init__()
        self.n = n

    def __call__(self, x: mx.array) -> mx.array:
        return mx.clip(x, -self.n, self.n)


class DenseBlock(nn.Module):
    """Single MLP block: [LayerNorm] -> [Dropout] -> Linear -> [Activation]."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Optional[str] = "leaky_relu",
        norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        if norm:
            layers.append(nn.LayerNorm(in_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, out_dim))
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "relu":
            layers.append(nn.ReLU())
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class ScoreEncoder(nn.Module):
    """SCORE-style encoder with shared block and skip averaging."""

    def __init__(
        self,
        input_dim: int,
        score_dim: int = 256,
        score_steps: int = 5,
        skip_alpha: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, score_dim) if input_dim != score_dim else None
        self.shared_block = DenseBlock(
            score_dim,
            score_dim,
            activation="leaky_relu",
            norm=True,
            dropout=dropout,
        )
        self.score_steps = score_steps
        self.skip_alpha = skip_alpha
        self.out_dim = score_dim

    def __call__(self, x: mx.array) -> mx.array:
        if self.proj_in is not None:
            x = self.proj_in(x)
        alpha = self.skip_alpha
        for _ in range(self.score_steps):
            x = alpha * x + (1.0 - alpha) * self.shared_block(x)
        return x


class Decoder(nn.Module):
    """MLP decoder: encoding -> reconstruction of input features."""

    def __init__(
        self,
        encoding_size: int,
        hidden_sizes: List[int],
        output_dim: int,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        if hidden_sizes:
            prev = encoding_size
            for h in hidden_sizes:
                blocks.append(DenseBlock(prev, h, activation="leaky_relu", norm=False))
                prev = h
            blocks.append(nn.Linear(prev, output_dim))
        else:
            blocks.append(nn.Linear(encoding_size, output_dim))
        self.blocks = blocks

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return x
