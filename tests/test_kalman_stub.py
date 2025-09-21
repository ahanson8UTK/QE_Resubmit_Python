"""Ensure the SR-KF stub executes without numerical issues."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cw2017.utils import jax_setup  # noqa: F401
from cw2017.kalman import sr_kf


def test_sr_kf_random_system() -> None:
    key = jax.random.PRNGKey(42)
    y_t = jax.random.normal(key, (4, 3), dtype=jnp.float64)
    h_t = jnp.zeros_like(y_t)
    m_t = jnp.zeros_like(y_t)
    loglik, aux = sr_kf.sr_kf_loglik({}, h_t, y_t, m_t)
    assert jnp.isfinite(loglik)
    assert jnp.all(jnp.isfinite(aux["innovations"]))
