"""Generic constraint handling utilities.

TODOs:
- implement affine constraint solvers for measurement equations
- integrate equality constraints into HMC reparameterisations
- expose projection operators compatible with JAX autodiff
"""

from __future__ import annotations

import jax.numpy as jnp

from ..typing import Array


def project_simplex(x: Array) -> Array:
    """Project ``x`` onto the probability simplex."""

    if x.ndim != 1:
        raise ValueError("Simplex projection expects a 1D array")
    sorted_x = jnp.sort(x)[::-1]
    cssv = jnp.cumsum(sorted_x)
    rho = jnp.argmax(sorted_x - (cssv - 1) / (jnp.arange(x.shape[0]) + 1) > 0)
    theta = (cssv[rho] - 1) / (rho + 1)
    return jnp.maximum(x - theta, 0.0)


def enforce_bounds(x: Array, lower: float, upper: float) -> Array:
    """Clip values between ``lower`` and ``upper``."""

    return jnp.clip(x, lower, upper)


__all__ = ["project_simplex", "enforce_bounds"]
