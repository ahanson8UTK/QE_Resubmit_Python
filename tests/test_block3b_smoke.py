"""Smoke-level tests for the Block 3b HMC update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hmc_gibbs.models.conditionals import logpost_block2_q_eigs, transform_q_measure_eigs
from hmc_gibbs.samplers import gibbs
from hmc_gibbs.utils import jax_setup  # noqa: F401


@pytest.fixture(scope="module")
def block3b_setup():
    rng = jax.random.PRNGKey(1)
    T = 30
    d_state = 4
    d_y = 3
    d_m = 2

    key_y, key_m, key_h, key_theta = jax.random.split(rng, 4)
    y_t = 0.1 * jax.random.normal(key_y, (T, d_y), dtype=jnp.float64)
    m_t = 0.1 * jax.random.normal(key_m, (T, d_m), dtype=jnp.float64)
    h_t = 0.1 * jax.random.normal(key_h, (T, d_state), dtype=jnp.float64)
    theta = 0.05 * jax.random.normal(key_theta, (d_state,), dtype=jnp.float64)
    lambda_q = jnp.linspace(0.8, 0.4, d_state, dtype=jnp.float64)

    fixed = {"h_t": h_t}
    params = {"a0": jnp.zeros(d_state), "S0": jnp.eye(d_state), "theta": theta, "lambda_q": lambda_q}
    data = {
        "y_t": y_t,
        "m_t": m_t,
        "fixed": fixed,
        "params": params,
        "rho_max": 0.995,
        "prior_scale_q": 0.25,
    }
    return rng, fixed, data


def test_q_measure_transform_bounds(block3b_setup) -> None:
    _rng, _fixed, data = block3b_setup
    rho_max = float(data["rho_max"])
    raw = jnp.array([2.0, -1.0, 0.5, 1.5], dtype=jnp.float64)
    eigs, logdet = transform_q_measure_eigs(raw, rho_max)
    assert jnp.all(jnp.abs(eigs) < rho_max)
    assert jnp.isfinite(logdet)


def test_block3b_logdensity_finite(block3b_setup) -> None:
    _rng, fixed, data = block3b_setup
    theta_raw = jnp.zeros(4, dtype=jnp.float64)
    value = logpost_block2_q_eigs(theta_raw, fixed, data)
    assert jnp.isfinite(value)


def test_block3b_hmc_step(block3b_setup) -> None:
    rng, fixed, data = block3b_setup
    num_chains = 2
    theta3a = jnp.zeros((num_chains, 4), dtype=jnp.float64)
    theta3b = 0.05 * jax.random.normal(rng, (num_chains, 4), dtype=jnp.float64)
    state = gibbs.SmokeState(theta_block3a=theta3a, theta_block3b=theta3b)

    cfg = {
        "warmup": {
            "num_warmup_steps": 25,
            "target_accept": 0.7,
            "adam_lr": 0.03,
            "jitter_amount": 0.8,
            "decay_rate": 0.4,
            "initial_step_size": 0.2,
        },
        "block3b": {
            "warmup": {
                "num_warmup_steps": 25,
                "target_accept": 0.7,
            },
            "mass_regularization": 3.0,
            "step_size_jitter": 0.2,
            "boundary_threshold": 0.98,
            "trajectory_shrink": 0.5,
            "max_reject_streak": 3,
            "reject_step_size_shrink": 0.7,
        },
    }

    rng, state, metrics = gibbs.run_one_sweep_smoke(rng, state, cfg, data)
    assert any(m["block"] == "3b" for m in metrics)
    assert state.tuned_params_3b is not None

    rng, state, metrics = gibbs.run_one_sweep_smoke(rng, state, cfg, data)
    metrics_3b = next(m for m in metrics if m["block"] == "3b")
    assert 0.0 <= metrics_3b["acceptance"] <= 1.0
    assert state.hmc_state_3b is not None
