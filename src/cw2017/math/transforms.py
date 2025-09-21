"""Deterministic parameter transforms and constraints.

TODOs:
- refine :func:`ordered_eigenvalues_transform` to match Creal & Wu structure
- complete :func:`linear_equity_constraint_placeholder` with model-specific matrices
- add bijectors for unit-diagonal lower-triangular matrices and covariance factors
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
import jax.nn as jnn

from ..typing import Array


def safe_softplus(x: Array, beta: float = 1.0) -> Array:
    """Numerically stable softplus transform."""

    return jnn.softplus(beta * x) / beta


def logit(x: Array, low: float = 0.0, high: float = 1.0) -> Array:
    """Map unconstrained values to ``(low, high)``."""

    span = high - low
    return low + span / (1.0 + jnp.exp(-x))


def ordered_transform(raw: Array) -> Array:
    """Ensure monotonically increasing elements via cumulative softplus increments."""

    if raw.ndim == 0:
        return raw
    base = raw[..., :1]
    increments = jnn.softplus(raw[..., 1:])
    ordered_tail = base + jnp.cumsum(increments, axis=-1)
    return jnp.concatenate([base, ordered_tail], axis=-1)


def skew_params_to_rotation(params: Array) -> Array:
    """Map 3-vector axis-angle parameters to a 3x3 rotation matrix."""

    if params.shape[-1] != 3:
        raise ValueError("Rotation parameterisation expects 3 elements")
    theta = jnp.linalg.norm(params, axis=-1, keepdims=True)
    eye = jnp.eye(3, dtype=params.dtype)

    def _rod(params: Array, theta: Array) -> Array:
        axis = params / theta
        kx, ky, kz = axis
        K = jnp.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=params.dtype)
        sin_term = jnp.sin(theta)
        cos_term = jnp.cos(theta)
        return eye + sin_term * K + (1.0 - cos_term) * (K @ K)

    return lax.cond(
        jnp.all(theta < 1e-8),
        lambda _: eye + _skew(params),
        lambda _: _rod(params, theta.squeeze()),
        operand=None,
    )


def _skew(vec: Array) -> Array:
    x, y, z = vec
    return jnp.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=vec.dtype)


def ordered_eigenvalues_transform(raw: Array, radius: float | None = None) -> Array:
    """Transform unconstrained values to strictly ordered eigenvalues."""

    ordered = ordered_transform(raw)
    if radius is not None:
        ordered = jnp.tanh(ordered / radius) * radius
    return ordered


def linear_equity_constraint_placeholder(*args: Array, **kwargs: Array) -> Array:
    """Placeholder for solving the linear equity pricing restriction.

    The final implementation will accept structural matrices ``theta_m``, ``theta_g``,
    ``theta_g_q``, and ``V`` alongside unconstrained mean parameters and return the
    constrained means. This scaffold simply raises to highlight the missing component.
    """

    raise NotImplementedError("Linear equity constraint solver is pending implementation")


__all__ = [
    "safe_softplus",
    "logit",
    "ordered_transform",
    "skew_params_to_rotation",
    "ordered_eigenvalues_transform",
    "linear_equity_constraint_placeholder",
]
