"""Equity price recursion placeholders.

TODOs:
- implement matrices A_n, B_n, C_n, D_n per Creal & Wu (2017)
- connect recursion outputs to observed equity price residuals
- cache recursion intermediates to avoid redundant computation during Gibbs sweeps
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
from jax import lax

from ..typing import Array


def price_equity_series(params: Any, states: Array, horizons: int, cache: Dict[str, Array] | None = None) -> Array:
    """Compute placeholder fundamental equity value series via linear recursions."""

    states = jnp.asarray(states, dtype=jnp.float64)

    def step(carry: Array, _) -> tuple[Array, Array]:
        next_carry = carry + 0.1
        return next_carry, next_carry

    initial = jnp.zeros(states.shape[0], dtype=jnp.float64)
    _, outputs = lax.scan(step, initial, jnp.arange(horizons))
    return outputs[-1]


__all__ = ["price_equity_series"]
