"""SR-Kalman filter smoke tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cw2017.kalman import sr_kf
from cw2017.utils import jax_setup  # noqa: F401


def _identity_builders(state_dim: int, obs_dim: int):
    def build_HR(t, params, h_t, m_t):
        H = jnp.eye(state_dim, dtype=jnp.float64)[:obs_dim, :]
        R = 0.1 * jnp.eye(obs_dim, dtype=jnp.float64)
        return H, R

    def build_FQ(t, params, h_t, m_t):
        F = jnp.eye(state_dim, dtype=jnp.float64)
        Q = 0.05 * jnp.eye(state_dim, dtype=jnp.float64)
        return F, Q

    return build_HR, build_FQ


def test_sr_kf_loglik_shapes() -> None:
    key = jax.random.PRNGKey(0)
    T, state_dim, obs_dim = 8, 4, 3
    y_key, h_key, m_key = jax.random.split(key, 3)
    y_t = jax.random.normal(y_key, (T, obs_dim), dtype=jnp.float64)
    h_t = jax.random.normal(h_key, (T, state_dim), dtype=jnp.float64)
    m_t = jax.random.normal(m_key, (T, 2), dtype=jnp.float64)

    build_HR, build_FQ = _identity_builders(state_dim, obs_dim)
    params = {
        "a0": jnp.zeros(state_dim, dtype=jnp.float64),
        "S0": jnp.eye(state_dim, dtype=jnp.float64) * 0.4,
        "fns": {"build_HR": build_HR, "build_FQ": build_FQ},
    }

    loglik, aux = sr_kf.sr_kf_loglik(params, h_t, y_t, m_t)
    assert jnp.isfinite(loglik)
    assert aux["filtered_means"].shape == (T, state_dim)
    assert aux["filtered_sqrt_covs"].shape == (T, state_dim, state_dim)
    assert not bool(aux["nan_detected"])

    compiled = jax.jit(lambda h, y, m: sr_kf.sr_kf_loglik(params, h, y, m))
    loglik_jit, aux_jit = compiled(h_t, y_t, m_t)
    assert jnp.isfinite(loglik_jit)
    assert aux_jit["filtered_means"].shape == (T, state_dim)
