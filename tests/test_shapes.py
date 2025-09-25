"""Shape and dtype smoke tests for selected utilities."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from kalman.ffbs import Step

from hmc_gibbs.equity import constraints
from hmc_gibbs.samplers.hmc_block import adapt_block_chees, hmc_block_step
from hmc_gibbs.utils import jax_setup  # noqa: F401


def test_smoother_output_shapes() -> None:
    key = jax.random.PRNGKey(0)
    from hmc_gibbs.kalman import smoother  # imported lazily to avoid circular deps

    T = 5
    g_dim = 2
    obs_dim = 6

    steps = []
    Z = jnp.zeros((obs_dim, g_dim), dtype=jnp.float64)
    d = jnp.zeros((obs_dim,), dtype=jnp.float64)
    H = jnp.eye(obs_dim, dtype=jnp.float64) * 0.05
    T_mat = jnp.eye(g_dim, dtype=jnp.float64) * 0.95
    c = jnp.zeros((g_dim,), dtype=jnp.float64)
    R = jnp.eye(g_dim, dtype=jnp.float64)
    Q = jnp.eye(g_dim, dtype=jnp.float64) * 0.1
    for _ in range(T):
        steps.append(Step(Z=Z, d=d, H=H, T=T_mat, c=c, R=R, Q=Q))

    state_spec = {
        "steps": steps,
        "x0_mean": jnp.zeros((g_dim,), dtype=jnp.float64),
        "x0_cov": jnp.eye(g_dim, dtype=jnp.float64),
        "slices": {"g": slice(0, g_dim), "mu_m": slice(g_dim, g_dim), "mu_g_u": slice(g_dim, g_dim), "mu_gQ_u": slice(g_dim, g_dim)},
    }

    params = {"state_space": state_spec}

    g_sample, means = smoother.dk_simulation_smoother(
        params, jnp.zeros((T, 2)), jnp.zeros((T, 2)), jnp.zeros((T, 2)), key
    )
    assert g_sample.shape == (T, g_dim)
    assert means.shape == (0,)


def test_equity_constraint_enforces_inequality() -> None:
    means = jnp.array([0.1, -0.2, 0.3, 0.4, 0.5], dtype=jnp.float64)
    params = {
        "slices": {
            "mu_m": slice(0, 2),
            "mu_g_u": slice(2, 3),
            "mu_gQ_u": slice(3, 5),
        },
        "theta_m": jnp.array([0.5, -0.25], dtype=jnp.float64),
        "theta_g": jnp.array([0.3], dtype=jnp.float64),
        "theta_g_q": jnp.array([0.4, -0.6], dtype=jnp.float64),
        "V": 0.1,
        "pivot_index": 1,
        "slack_free": jnp.array(0.2, dtype=jnp.float64),
    }

    solution = constraints.solve_linear_equity_constraint(params, means)
    updated = solution.means

    mu_m = updated[params["slices"]["mu_m"]]
    mu_gu = updated[params["slices"]["mu_g_u"]]
    mu_gq = updated[params["slices"]["mu_gQ_u"]]

    lhs = (
        jnp.dot(params["theta_m"], mu_m)
        + jnp.dot(params["theta_g"], mu_gu)
        + jnp.dot(params["theta_g_q"], mu_gq)
        + params["V"]
    )
    np.testing.assert_allclose(lhs, -solution.slack, rtol=1e-5, atol=1e-5)


def test_hmc_block_adaptation_smoke() -> None:
    key = jax.random.PRNGKey(123)

    def logdensity(theta: jnp.ndarray) -> jnp.float64:
        return -0.5 * jnp.sum(theta**2)

    init_positions = jnp.zeros((3, 4), dtype=jnp.float64)
    warmup_cfg = {
        "num_warmup_steps": 6,
        "initial_step_size": 0.2,
        "target_accept": 0.65,
        "jitter_amount": 0.8,
    }
    last_states, tuned_params, diagnostics = adapt_block_chees(key, init_positions, logdensity, warmup_cfg)
    assert last_states.position.shape == init_positions.shape
    assert float(tuned_params["step_size"]) > 0.0
    assert float(diagnostics["final_trajectory_length"]) > 0.0

    sample_key = jax.random.PRNGKey(456)
    new_state, info = hmc_block_step(sample_key, last_states, tuned_params)
    assert new_state.position.shape == init_positions.shape
    assert info["acceptance"].shape == (init_positions.shape[0],)
