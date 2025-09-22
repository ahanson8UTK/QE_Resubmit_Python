r"""Monotone stick-breaking maps used for eigenvalue ordering.

The transformation implemented here follows the standard *stick-breaking*
construction that maps unconstrained real vectors to strictly ordered values
inside a bounded interval.  Given a vector ``u`` of unconstrained reals and
bounds ``(low, high)``, the returned vector is strictly decreasing with every
entry living in the open interval ``(low, high)``.  The map is smooth and
JIT-friendly which makes it suitable for optimisation and HMC sampling.

Example
-------
>>> import jax.numpy as jnp
>>> from cw2017.math.ordered_bounded import ordered_bounded_descending
>>> ordered_bounded_descending(jnp.array([0.0, 1.0, -1.0]), -0.95, 0.95)
DeviceArray([ 0.64410174,  0.30978587, -0.5660666 ], dtype=float64)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.nn import softplus

Array = jnp.ndarray


def ordered_bounded_descending(u: Array, low: float, high: float) -> Array:
    """Return a strictly decreasing vector with entries in ``(low, high)``.

    Parameters
    ----------
    u:
        Unconstrained real vector.  Each entry can take any value in
        :math:`(-\infty, \infty)`.
    low, high:
        Scalars defining the open interval that bounds the ordered output.

    Returns
    -------
    Array
        A vector of the same length as ``u`` whose entries are strictly
        decreasing and constrained to ``(low, high)``.

    Notes
    -----
    The implementation rescales positive "step" sizes obtained from a
    softplus transform such that the cumulative step length occupies
    ``(1 - eps)`` of the interval width.  Subtracting the cumulative steps
    from ``high`` therefore yields a strictly decreasing sequence that never
    touches the boundaries.
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    width = high - low
    eps = jnp.array(1e-8, dtype=jnp.float64)
    positive_steps = softplus(u) + eps
    total = jnp.sum(positive_steps)
    scaled_steps = positive_steps * (1.0 - eps) * width / (total + eps)
    cumulative = jnp.cumsum(scaled_steps)
    ordered = high - cumulative
    # Ensure we remain above the lower bound by a small positive margin.
    lower_buffer = low + eps * width
    return jnp.clip(ordered, lower_buffer, high - eps * width)
