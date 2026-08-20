"""Frozen observation encoders for the cogames GA evaluators.

The four architecture evaluators (Hebbian / static MLP / GRU / Perceiver) all
flatten raw token observations into a high-dim vector that then becomes the
input dim of a weight matrix in the evolved gene. Searching that weight matrix
under CMA-ES at 60k-300k dims is essentially random — see the
project-arch-comparison notes for the diagnosis.

A frozen random projection cheaply collapses the raw vector to a few-dozen
dimensions before it ever touches the evolved weights. With the same seed,
every architecture (and every worker) builds the same projection matrix, so
results stay comparable across the four evaluators.
"""

from __future__ import annotations

import numpy as np


VALID_KINDS = ('none', 'random_proj')


class FrozenRandomProjection:
    """Deterministic Gaussian random projection input_dim -> output_dim.

    Entries are N(0, 1/sqrt(input_dim)) (Johnson-Lindenstrauss scaling), so
    unit-norm inputs roughly preserve their norm after projection. The matrix
    is built from a seeded numpy Generator — same (input_dim, output_dim, seed)
    yields the same matrix in every process.
    """

    __slots__ = ('input_dim', 'output_dim', 'seed', 'W')

    def __init__(self, input_dim: int, output_dim: int, seed: int = 1337):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        scale = np.float32(1.0 / np.sqrt(max(self.input_dim, 1)))
        self.W = (rng.standard_normal((self.output_dim, self.input_dim))
                  .astype(np.float32) * scale).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size != self.input_dim:
            raise ValueError(
                f'random projection expected {self.input_dim} dims, got {x.size}'
            )
        return self.W @ x


def maybe_make_encoder(kind: str, input_dim: int,
                       output_dim: int, seed: int):
    """Return (encoder_callable_or_None, effective_output_dim).

    kind == 'none' / '' / None -> identity (no encoder, input_dim passes through).
    kind == 'random_proj'      -> FrozenRandomProjection(input_dim -> output_dim).
    """
    if kind in (None, '', 'none', 'off', 'false', 'False'):
        return None, int(input_dim)
    if kind == 'random_proj':
        return FrozenRandomProjection(input_dim, output_dim, seed), int(output_dim)
    raise ValueError(
        f'Unknown obs_encoder kind: {kind!r} (valid: {VALID_KINDS})'
    )


def effective_dim(kind: str, raw_dim: int, output_dim: int) -> int:
    """Convenience: post-encoder dim of a single raw vector, without building W."""
    if kind in (None, '', 'none', 'off', 'false', 'False'):
        return int(raw_dim)
    if kind == 'random_proj':
        return int(output_dim)
    raise ValueError(
        f'Unknown obs_encoder kind: {kind!r} (valid: {VALID_KINDS})'
    )
