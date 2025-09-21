"""Shape and dtype smoke tests for selected utilities."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from cw2017.equity import constraints
from cw2017.samplers.hmc_block import adapt_block_chees, hmc_block_step
from cw2017.utils import jax_setup  # noqa: F401


def test_smoother_output_shapes() -> None:
    key = jax.random.PRNGKey(0)
    from cw2017.kalman import smoother  # imported lazily to avoid circular deps

    g_sample, means = smoother.dk_simulation_smoother(
        {}, jnp.zeros((5, 2)), jnp.zeros((5, 2)), jnp.zeros((5, 2)), key
    )
    assert g_sample.shape == (5, 2)
    assert means.shape == (5, 2)


def test_equity_constraint_identity() -> None:
    means = jnp.ones((3,))
    constrained = constraints.solve_linear_equity_constraint({}, means)
    np.testing.assert_allclose(constrained, means)


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
