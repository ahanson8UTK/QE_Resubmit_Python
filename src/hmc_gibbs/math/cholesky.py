"""Cholesky factorisation helpers.

TODOs:
- implement :func:`stable_cholesky` with QR fallback
- add reverse-mode differentiation friendly solve routines
- expose log-determinant utilities for Jacobian corrections
"""

from __future__ import annotations

import jax.numpy as jnp

from ..typing import Array


def stable_cholesky(matrix: Array) -> Array:
    """Placeholder for a robust Cholesky factorisation."""

    raise NotImplementedError("stable_cholesky awaits implementation")


def cholesky_logdet(matrix: Array) -> jnp.float64:
    """Compute the log-determinant via the Cholesky factor."""

    raise NotImplementedError("cholesky_logdet is not yet implemented")


__all__ = ["stable_cholesky", "cholesky_logdet"]
