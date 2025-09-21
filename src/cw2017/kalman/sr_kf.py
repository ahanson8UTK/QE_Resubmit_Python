"""Square-root Kalman filter scaffold.

TODOs:
- assemble measurement and transition matrices from ``params``
- implement Joseph-form covariance updates in log-domain
- propagate work-unit accounting hooks for each SR-KF evaluation
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax import lax

from ..typing import Array


def sr_kf_loglik(params: Any, h_t: Array, y_t: Array, m_t: Array) -> Tuple[jnp.float64, Dict[str, Array]]:
    """Run a square-root Kalman filter and return the log-likelihood.

    This stub focuses on array plumbing so unit tests can verify shapes and NaN guards.
    The actual implementation will propagate square-root factors and innovations.
    """

    y_t = jnp.asarray(y_t, dtype=jnp.float64)
    T, obs_dim = y_t.shape
    h_t = jnp.asarray(h_t, dtype=jnp.float64)
    m_t = jnp.asarray(m_t, dtype=jnp.float64)

    def step(carry: Array, inputs: Tuple[Array, Array, Array]) -> Tuple[Array, Array]:
        _, y_obs, _ = inputs
        innovation = y_obs - carry
        new_state = jnp.zeros_like(carry)
        return new_state, innovation

    initial_mean = jnp.zeros(obs_dim, dtype=jnp.float64)
    _, innovations = lax.scan(step, initial_mean, (h_t, y_t, m_t))
    loglik = jnp.array(0.0, dtype=jnp.float64)
    aux: Dict[str, Array] = {"innovations": innovations}
    if not jnp.all(jnp.isfinite(innovations)):
        raise FloatingPointError("Non-finite innovations encountered in SR-KF stub")
    return loglik, aux


__all__ = ["sr_kf_loglik"]
