"""Shape and dtype smoke tests for key stubs."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from cw2017.utils import jax_setup  # noqa: F401
from cw2017.kalman import sr_kf, smoother
from cw2017.equity import constraints


def test_sr_kf_shapes() -> None:
    params = {}
    h_t = jnp.zeros((5, 2))
    y_t = jnp.zeros((5, 2))
    m_t = jnp.zeros((5, 2))
    loglik, aux = sr_kf.sr_kf_loglik(params, h_t, y_t, m_t)
    assert loglik.shape == ()
    assert aux["innovations"].shape == (5, 2)


def test_smoother_output_shapes() -> None:
    key = jax.random.PRNGKey(0)
    g_sample, means = smoother.dk_simulation_smoother({}, jnp.zeros((5, 2)), jnp.zeros((5, 2)), jnp.zeros((5, 2)), key)
    assert g_sample.shape == (5, 2)
    assert means.shape == (5, 2)


def test_equity_constraint_identity() -> None:
    means = jnp.ones((3,))
    constrained = constraints.solve_linear_equity_constraint({}, means)
    np.testing.assert_allclose(constrained, means)
