"""Transform utilities for HMC blocks."""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp

_EPS = 1e-12


def _ordered_unit_interval(z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return cumulative stick-breaking map and log-Jacobian."""

    z = jnp.asarray(z, dtype=jnp.float64)

    def _body(carry, z_i):
        cumulative, remaining, log_jac = carry
        v = jax.nn.sigmoid(z_i)
        contrib = v * remaining
        new_cumulative = cumulative + contrib
        new_remaining = remaining * (1.0 - v)
        log_jac_inc = jnp.log(v * (1.0 - v) + _EPS) + jnp.log(remaining + _EPS)
        return (new_cumulative, new_remaining, log_jac + log_jac_inc), new_cumulative

    init = (jnp.array(0.0, dtype=z.dtype), jnp.array(1.0, dtype=z.dtype), jnp.array(0.0, dtype=z.dtype))
    (final_state, _remaining, log_jac), ordered = jax.lax.scan(_body, init, z)
    del final_state, _remaining
    return ordered, log_jac


def _inverse_ordered_unit_interval(ordered: jnp.ndarray) -> jnp.ndarray:
    """Inverse of :func:`_ordered_unit_interval` returning unconstrained vector."""

    ordered = jnp.asarray(ordered, dtype=jnp.float64)

    def _body(carry, o_i):
        previous, remaining = carry
        increment = (o_i - previous) / (remaining + _EPS)
        increment = jnp.clip(increment, 1e-9, 1.0 - 1e-9)
        z_i = jax.scipy.special.logit(increment)
        new_remaining = remaining * (1.0 - increment)
        return (o_i, new_remaining), z_i

    init = (jnp.array(0.0, dtype=ordered.dtype), jnp.array(1.0, dtype=ordered.dtype))
    (_, _), z = jax.lax.scan(_body, init, ordered)
    return z


def eigenvalues_q_transform(zeta: jnp.ndarray, delta: float, eps: float = 1e-3) -> Tuple[jnp.ndarray, float]:
    """Map unconstrained values to ordered, spectrally bounded eigenvalues."""

    if zeta.ndim != 1:
        raise ValueError("zeta must be a vector")
    rho_max = (1.0 - float(eps)) / float(delta)
    ordered_unit, log_jac = _ordered_unit_interval(zeta)
    ordered_unit = jnp.clip(ordered_unit, 1e-9, 1.0 - 1e-9)
    eta = jax.scipy.special.logit(ordered_unit)
    tanh_eta = jnp.tanh(eta)
    lam = rho_max * tanh_eta

    log_jac += zeta.size * jnp.log(rho_max)
    log_jac += jnp.sum(jnp.log1p(-tanh_eta**2 + _EPS))
    log_jac -= jnp.sum(jnp.log(ordered_unit) + jnp.log1p(-ordered_unit))
    return lam, float(log_jac)


def inv_eigenvalues_q_transform(lambda_vec: jnp.ndarray, delta: float, eps: float = 1e-3) -> jnp.ndarray:
    """Inverse of :func:`eigenvalues_q_transform`."""

    rho_max = (1.0 - float(eps)) / float(delta)
    x = jnp.clip(lambda_vec / rho_max, -0.999999, 0.999999)
    eta = jnp.arctanh(x)
    ordered = jax.nn.sigmoid(eta)
    return _inverse_ordered_unit_interval(ordered)


def _infer_cholesky_dim(num_params: int, unit_diag: bool) -> int:
    disc = 1 + 8 * num_params
    if unit_diag:
        n = (1.0 + math.sqrt(disc)) / 2.0
    else:
        n = (-1.0 + math.sqrt(disc)) / 2.0
    if n <= 0:
        raise ValueError("Invalid parameter count for triangular matrix")
    return int(round(n))


def cholesky_unconstr_to_lower(u: jnp.ndarray, unit_diag: bool = False) -> Tuple[jnp.ndarray, float]:
    """Map unconstrained vector to lower-triangular matrix."""

    u = jnp.asarray(u, dtype=jnp.float64)
    n = _infer_cholesky_dim(u.size, unit_diag)
    L = jnp.zeros((n, n), dtype=jnp.float64)
    idx = 0
    log_jac = jnp.array(0.0, dtype=jnp.float64)
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                if unit_diag:
                    L = L.at[i, i].set(1.0)
                else:
                    x = u[idx]
                    val = jax.nn.softplus(x)
                    log_jac = log_jac + jnp.log(jax.nn.sigmoid(x) + _EPS)
                    L = L.at[i, i].set(val)
                    idx += 1
            else:
                L = L.at[i, j].set(u[idx])
                idx += 1
    return L, float(log_jac)


def lower_to_unconstr(L: jnp.ndarray, unit_diag: bool = False) -> jnp.ndarray:
    """Inverse map of :func:`cholesky_unconstr_to_lower`."""

    L = jnp.asarray(L, dtype=jnp.float64)
    n = L.shape[0]
    elems = []
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                if unit_diag:
                    continue
                d = jnp.clip(L[i, i], _EPS, None)
                elems.append(jnp.log(jnp.expm1(d)))
            else:
                elems.append(L[i, j])
    if not elems:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.stack(elems)


__all__ = [
    "eigenvalues_q_transform",
    "inv_eigenvalues_q_transform",
    "cholesky_unconstr_to_lower",
    "lower_to_unconstr",
]
