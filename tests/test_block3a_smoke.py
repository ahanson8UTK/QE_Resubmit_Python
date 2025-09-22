"""Smoke tests for the Block 3a HMC update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from cw2017.models.conditionals import make_logdensity_block3a
from cw2017.samplers.gibbs import Block3aSamplerState, run_block3a_hmc_step
from hmc_gibbs.utils import jax_setup  # noqa: F401


@pytest.fixture(scope="module")
def block3a_setup():
    rng = jax.random.PRNGKey(0)
    T = 40
    d_m, d_h, d_y = 5, 7, 7
    rng, m_key, h_key, y_key, u_key = jax.random.split(rng, 5)
    m_t = jax.random.normal(m_key, (T, d_m), dtype=jnp.float64) * 0.1
    h_t = jax.random.normal(h_key, (T, d_h), dtype=jnp.float64) * 0.1
    y_t = jax.random.normal(y_key, (T, d_y), dtype=jnp.float64) * 0.1

    Sigma_g = jnp.eye(3, dtype=jnp.float64)
    fixed = {
        "m_t": m_t,
        "h_t": h_t,
        "mu_g": jnp.zeros(3, dtype=jnp.float64),
        "Q_g^Q": jnp.eye(3, dtype=jnp.float64),
        "Lambda_g^Q": jnp.diag(jnp.array([0.9, 0.8, 0.7], dtype=jnp.float64)),
        "Sigma_g": Sigma_g,
        "Gamma0": jnp.zeros(d_m + 3, dtype=jnp.float64),
        "Gamma1": jnp.zeros((d_m + 3, d_h), dtype=jnp.float64),
        "mu_h_bar": jnp.zeros(d_h, dtype=jnp.float64),
        "mu_g^{Q,u}": jnp.zeros(2, dtype=jnp.float64),
    }
    data = {"y_t": y_t}
    cfg = {"rho_max": 0.995, "priors_3a": {}}
    warmup_cfg = {
        "num_warmup_steps": 200,
        "target_accept": 0.651,
        "adam_lr": 0.02,
        "jitter_amount": 1.0,
        "decay_rate": 0.5,
    }
    info = {"Nm": 5, "Ng": 3, "Nh": 7, "Nstates": 15, "vTau_star": 60}
    maturities = [3, 6, 12, 36, 60, 84, 120]

    dy = len(maturities)
    qs_len = 0
    from cw2017.math.fill_q import count_qs_entries

    qs_len = count_qs_entries(info)
    dim = dy + 3 + (info["Nm"] + info["Ng"]) + info["Nh"] + qs_len + 3 + 1 + 1 + 2
    init_vec = jax.random.normal(u_key, (dim,), dtype=jnp.float64) * 0.01

    return rng, fixed, data, cfg, warmup_cfg, info, maturities, init_vec


def test_logdensity_finite(block3a_setup) -> None:
    rng, fixed, data, cfg, _warmup, info, maturities, init_vec = block3a_setup
    logdensity = make_logdensity_block3a(fixed, data, cfg, info, maturities)
    value = logdensity(init_vec)
    assert jnp.isfinite(value)


def test_hmc_warmup_and_step(block3a_setup) -> None:
    rng, fixed, data, cfg, warmup_cfg, info, maturities, init_vec = block3a_setup
    num_chains = 2
    init_positions = jnp.tile(init_vec, (num_chains, 1))
    state = Block3aSamplerState(positions={"3a": init_positions})

    rng, state, metrics = run_block3a_hmc_step(
        rng, state, fixed, data, cfg, warmup_cfg, info, maturities
    )
    assert metrics["step_type"] == "warmup"
    tuned = state.tuned_params["3a"]
    assert float(tuned["step_size"]) > 0.0
    assert "3a" in state.kernels

    rng, state, metrics = run_block3a_hmc_step(
        rng, state, fixed, data, cfg, warmup_cfg, info, maturities
    )
    assert metrics["step_type"] == "sample"
    assert 0.0 <= metrics["acceptance"] <= 1.0

    logdensity = make_logdensity_block3a(fixed, data, cfg, info, maturities)
    logps = jax.vmap(logdensity)(state.positions["3a"])
    assert jnp.all(jnp.isfinite(logps))
