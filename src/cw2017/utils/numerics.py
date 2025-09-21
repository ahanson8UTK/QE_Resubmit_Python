"""Numerical guard rails shared across Kalman and sampling code."""

from __future__ import annotations

import jax.numpy as jnp

from ..typing import Array


def symmetrize(matrix: Array) -> Array:
    """Return the symmetrised version of ``matrix``."""

    return 0.5 * (matrix + matrix.T)


def clip_variances(diag: Array, minimum: float = 1e-10) -> Array:
    """Ensure diagonal elements remain above ``minimum``."""

    return jnp.maximum(diag, minimum)


__all__ = ["symmetrize", "clip_variances"]
