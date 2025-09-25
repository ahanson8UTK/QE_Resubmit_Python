"""Equity pricing constraint utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ..typing import Array


class EquityConstraintSolution(NamedTuple):
    """Result of enforcing the equity-pricing inequality."""

    means: jnp.ndarray
    log_jacobian: jnp.float64
    slack: jnp.float64


def _get(mapping: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(name, default)
    return getattr(mapping, name, default)


def _softplus_tau(z: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    scaled = z / temperature
    return temperature * jnp.log1p(jnp.exp(scaled))


def _softplus_tau_prime(z: jnp.ndarray, temperature: jnp.ndarray) -> jnp.ndarray:
    scaled = z / temperature
    return jax.nn.sigmoid(scaled)


def _ensure_slice(slices: Mapping[str, slice], key: str) -> slice:
    sl = slices.get(key)
    if sl is None:
        raise KeyError(f"Missing slice for '{key}' in equity constraint parameters")
    return sl


def solve_linear_equity_constraint(params: Any, means: Array) -> EquityConstraintSolution:
    """Apply the linear equity constraint using the slack reparameterisation.

    The implementation mirrors the description in Section 3.3 of
    ``docs/algorithm_overview`` while using the coefficient structure reported in
    Section 1.3 of ``docs/model_writeup``.  We introduce a free scalar ``ζ`` and
    map it through a temperature-controlled softplus to obtain a strictly
    positive slack ``s`` that keeps the inequality

    .. math::

        \theta_m^\top \bar{\mu}_m + \theta_g^\top \bar{\mu}_g^u
        + (\theta_g^{\mathbb{Q}})^\top \bar{\mu}_g^{\mathbb{Q},u} + V = -s

    satisfied for every Gibbs iteration.  One element of
    :math:`\bar{\mu}_g^{\mathbb{Q},u}` is treated as the pivot and solved for
    analytically, yielding the Jacobian contribution that must be added to the log
    posterior.
    """

    means_arr = jnp.asarray(means, dtype=jnp.float64)

    slices = _get(params, "slices", {})
    if not isinstance(slices, Mapping):
        raise TypeError("`params` must expose a mapping `slices` with state partitions")

    mu_m_slice = _ensure_slice(slices, "mu_m")
    mu_gu_slice = _ensure_slice(slices, "mu_g_u")
    mu_gq_slice = _ensure_slice(slices, "mu_gQ_u")

    mu_m = means_arr[mu_m_slice]
    mu_gu = means_arr[mu_gu_slice]
    mu_gq = means_arr[mu_gq_slice]

    theta_m = jnp.asarray(_get(params, "theta_m", jnp.zeros_like(mu_m)), dtype=jnp.float64)
    theta_g = jnp.asarray(_get(params, "theta_g", jnp.zeros_like(mu_gu)), dtype=jnp.float64)
    theta_gq = jnp.asarray(_get(params, "theta_g_q", jnp.zeros_like(mu_gq)), dtype=jnp.float64)
    V = jnp.asarray(_get(params, "V", 0.0), dtype=jnp.float64)

    pivot_rel = int(_get(params, "pivot_index", mu_gq.shape[0] - 1))
    if pivot_rel < 0 or pivot_rel >= mu_gq.shape[0]:
        raise IndexError("Pivot index for μ_g^{Q,u} is out of range")

    temperature = jnp.clip(
        jnp.asarray(_get(params, "slack_temperature", 1.0), dtype=jnp.float64),
        0.5,
        2.0,
    )
    zeta = jnp.asarray(_get(params, "slack_free", 0.0), dtype=jnp.float64)

    slack = _softplus_tau(zeta, temperature)
    theta_m_dot = jnp.dot(theta_m, mu_m) if mu_m.size else jnp.array(0.0, dtype=jnp.float64)
    theta_g_dot = jnp.dot(theta_g, mu_gu) if mu_gu.size else jnp.array(0.0, dtype=jnp.float64)

    pivot_coef = theta_gq[pivot_rel]
    if not jnp.isfinite(pivot_coef) or jnp.abs(pivot_coef) < 1e-12:
        raise ValueError("Pivot coefficient for μ_g^{Q,u} must be finite and non-zero")

    idx = jnp.arange(mu_gq.shape[0])
    mask = idx != pivot_rel
    tail_coef = theta_gq[mask]
    tail_mu = mu_gq[mask]
    tail_contrib = jnp.dot(tail_coef, tail_mu) if tail_coef.size else jnp.array(0.0, dtype=jnp.float64)

    numerator = -slack - theta_m_dot - theta_g_dot - tail_contrib - V
    pivot_value = numerator / pivot_coef
    mu_gq = mu_gq.at[pivot_rel].set(pivot_value)

    jacobian_term = _softplus_tau_prime(zeta, temperature)
    log_jac = jnp.log(jacobian_term) - jnp.log(jnp.abs(pivot_coef))

    updated = means_arr.at[mu_m_slice].set(mu_m)
    updated = updated.at[mu_gu_slice].set(mu_gu)
    updated = updated.at[mu_gq_slice].set(mu_gq)

    return EquityConstraintSolution(means=updated, log_jacobian=log_jac, slack=slack)


__all__ = ["solve_linear_equity_constraint", "EquityConstraintSolution"]
