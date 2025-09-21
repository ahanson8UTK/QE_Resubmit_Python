"""Particle Gibbs sampler stubs.

TODOs:
- implement conditional SMC with ancestor tracing
- support adaptive resampling thresholds and stratified resampling
- integrate latent volatility path updates with Kalman conditioning
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..typing import Array, PRNGKey


def pgibbs_sample_h(params: Any, g_t: Array, y_t: Array, key: PRNGKey, num_particles: int) -> Array:
    """Placeholder particle Gibbs update for the latent ``h`` path."""

    noise = jax.random.normal(key, shape=g_t.shape, dtype=jnp.float64)
    return noise


__all__ = ["pgibbs_sample_h"]
