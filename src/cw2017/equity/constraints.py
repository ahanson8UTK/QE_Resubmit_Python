"""Equity pricing linear constraints.

TODOs:
- populate θ matrices from Creal & Wu (2017) Appendix
- implement solver respecting partitioned means between g and m blocks
- add automatic differentiation tests verifying constraint Jacobians
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..typing import Array


def solve_linear_equity_constraint(params: Any, means: Array) -> Array:
    """Apply the linear equity pricing constraint to the conditional means.

    Parameters
    ----------
    params:
        Placeholder parameter bundle that will eventually expose θ matrices and variance
        terms required for the constraint.
    means:
        Array of conditional means prior to enforcing the restriction. The final
        implementation will adjust one element to satisfy θ_m μ_m + θ_g μ_g = V.

    Returns
    -------
    Array
        Means satisfying the placeholder constraint. Currently the function returns
        ``means`` unchanged and serves as a hook for future development.
    """

    return jnp.asarray(means, dtype=jnp.float64)


__all__ = ["solve_linear_equity_constraint"]
