"""Durbin–Koopman simulation smoother scaffold.

TODOs:
- integrate with SR-KF outputs for backward sampling
- incorporate disturbance smoothing for g-block draws
- expose RNG controls for reproducible simulation smoothing
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from ..typing import Array, PRNGKey


def dk_simulation_smoother(params: Any, h_t: Array, y_t: Array, m_t: Array, key: PRNGKey) -> Tuple[Array, Array]:
    """Stub for the Durbin–Koopman simulation smoother."""

    key, noise_key = jax.random.split(key)
    y_t = jnp.asarray(y_t, dtype=jnp.float64)
    latent = jax.random.normal(noise_key, shape=y_t.shape, dtype=jnp.float64)
    means = jnp.zeros_like(latent)
    return latent, means


__all__ = ["dk_simulation_smoother"]
