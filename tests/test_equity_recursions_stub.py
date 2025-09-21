"""Test the equity price recursion placeholder."""

from __future__ import annotations

import jax.numpy as jnp

from hmc_gibbs.utils import jax_setup  # noqa: F401
from hmc_gibbs.equity import price_recursions


def test_price_recursions_shape() -> None:
    states = jnp.zeros((5,))
    value = price_recursions.price_equity_series({}, states, horizons=3, cache=None)
    assert value.shape == (5,)
