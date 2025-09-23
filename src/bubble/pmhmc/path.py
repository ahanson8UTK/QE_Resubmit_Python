"""Deterministic path generation for the bubble block."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import linalg as jsp_linalg

from .types import BubbleData, BubbleParams, PMHMCConfig

ArrayDict = Dict[str, jnp.ndarray]


def generate_paths(
    theta: BubbleParams,
    u: Mapping[str, jnp.ndarray],
    T: int,
    h_b0: float = 0.0,
) -> ArrayDict:
    """Generate latent volatility paths and growth scales for the bubble process.

    Parameters
    ----------
    theta:
        Constrained model parameters.  Only ``mu_b``, ``phi_b`` and ``sigma_h``
        are used here.
    u:
        Dictionary containing the standard normal draws required by the
        recursions.  The entry ``"z_b"`` (shape ``(T,)``) provides the
        innovations for the volatility recursion while ``"eps_b"`` contains the
        standard normal shocks for the bubble growth equation.  The growth
        innovations are passed separately to :func:`propagate_B`.
    T:
        Number of time steps to propagate.
    h_b0:
        Initial log-volatility value :math:`h_{b,0}`.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary with entries ``"h_b"`` (shape ``(T,)``) containing the
        volatility path from :math:`t=1` through :math:`T` and ``"scale"``
        holding the standard deviations :math:`s_t`.
    """

    if T <= 0:
        raise ValueError("Time dimension T must be positive.")

    z_b = jnp.asarray(u["z_b"], dtype=jnp.float64)
    if z_b.shape[0] != T:
        raise ValueError("Input u['z_b'] must have leading dimension T.")

    eps_b = jnp.asarray(u["eps_b"], dtype=jnp.float64)
    if eps_b.shape[0] != T:
        raise ValueError("Input u['eps_b'] must have leading dimension T.")
    _ = eps_b  # ensure dependency for shape checking without returning

    def step(h_prev: jnp.ndarray, innovation: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_next = theta.mu_b + theta.phi_b * h_prev + innovation
        return h_next, h_next

    _, h_traj = lax.scan(step, jnp.asarray(h_b0, dtype=jnp.float64), z_b)
    scale = jnp.exp(0.5 * theta.sigma_h * h_traj)

    return {"h_b": h_traj, "scale": scale}


def _solve_sigma_dg(
    sigma_g: jnp.ndarray,
    Dg_t: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``(Sigma_g D_{g,t}) x = rhs`` for each time index ``t``."""

    sigma_g = jnp.asarray(sigma_g, dtype=jnp.float64)
    Dg_t = jnp.asarray(Dg_t, dtype=jnp.float64)
    rhs = jnp.asarray(rhs, dtype=jnp.float64)

    mats = sigma_g[None, :, :] * Dg_t[:, None, :]
    return jax.vmap(lambda mat, vec: jsp_linalg.solve(mat, vec, assume_a="pos"))(mats, rhs)


def compute_lambda_g(
    sigma_g: jnp.ndarray,
    Dg_t: jnp.ndarray,
    mu_g: jnp.ndarray,
    mu_g_q: jnp.ndarray,
    Phi_gm: jnp.ndarray,
    Phi_g: jnp.ndarray,
    Phi_g_q: jnp.ndarray,
    Phi_gh: jnp.ndarray,
    m_t: jnp.ndarray,
    g_t: jnp.ndarray,
    h_t: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the market price of ``g``-block risk :math:`\Lambda_{g,t}`.

    Parameters
    ----------
    sigma_g:
        Cholesky factor :math:`\Sigma_g` of shape ``(d_g, d_g)``.
    Dg_t:
        Diagonal scale terms :math:`D_{g,t}` with shape ``(T, d_g)``.
    mu_g, mu_g_q:
        Physical and risk-neutral drifts for the ``g`` block (shape ``(d_g,)``).
    Phi_gm, Phi_g, Phi_g_q, Phi_gh:
        Transition matrices appearing in the state dynamics with shapes
        ``(d_g, d_m)``, ``(d_g, d_g)``, ``(d_g, d_g)`` and ``(d_g, d_h)``
        respectively.
    m_t, g_t, h_t:
        State trajectories for the corresponding blocks.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(T, d_g)`` containing :math:`\Lambda_{g,t}` for each
        time period.
    """

    diff_mu = jnp.asarray(mu_g, dtype=jnp.float64) - jnp.asarray(mu_g_q, dtype=jnp.float64)
    gm_term = jnp.einsum("ij,tj->ti", jnp.asarray(Phi_gm, dtype=jnp.float64), jnp.asarray(m_t, dtype=jnp.float64))
    g_term = jnp.einsum(
        "ij,tj->ti",
        jnp.asarray(Phi_g, dtype=jnp.float64) - jnp.asarray(Phi_g_q, dtype=jnp.float64),
        jnp.asarray(g_t, dtype=jnp.float64),
    )
    h_term = jnp.einsum("ij,tj->ti", jnp.asarray(Phi_gh, dtype=jnp.float64), jnp.asarray(h_t, dtype=jnp.float64))
    rhs = diff_mu[None, :] + gm_term + g_term + h_term
    return _solve_sigma_dg(jnp.asarray(sigma_g, dtype=jnp.float64), jnp.asarray(Dg_t, dtype=jnp.float64), rhs)


def propagate_B(
    theta: BubbleParams,
    data: BubbleData,
    paths: Mapping[str, jnp.ndarray],
    B0: float,
    *,
    eps_b: jnp.ndarray,
    Lambda_g: Optional[jnp.ndarray] = None,
    config: Optional[PMHMCConfig] = None,
    truncation_floor: float = 1e-12,
) -> ArrayDict:
    """Propagate the bubble level ``B_t`` from latent paths and innovations."""

    scale = jnp.asarray(paths["scale"], dtype=jnp.float64)
    T = scale.shape[0]

    r_t = jnp.asarray(data.r_t, dtype=jnp.float64)
    if r_t.shape[0] != T:
        raise ValueError("Mismatch between provided paths and observed data horizon.")

    eps_b = jnp.asarray(eps_b, dtype=jnp.float64)
    if eps_b.shape[0] != T:
        raise ValueError("eps_b must have length T.")

    Dm = jnp.asarray(data.Dm_t, dtype=jnp.float64)
    Dg = jnp.asarray(data.Dg_t, dtype=jnp.float64)
    sigma_m = jnp.asarray(data.Sigma_m, dtype=jnp.float64)
    sigma_g = jnp.asarray(data.Sigma_g, dtype=jnp.float64)
    sigma_gm = jnp.asarray(data.Sigma_gm, dtype=jnp.float64)

    rho_bm = jnp.asarray(theta.rho_bm, dtype=jnp.float64)
    rho_bg = jnp.asarray(theta.rho_bg, dtype=jnp.float64)

    if Lambda_g is None:
        raise ValueError(
            "Lambda_g must be provided. Use `compute_lambda_g` if the caller "
            "needs to construct it internally."
        )
    Lambda_g = jnp.asarray(Lambda_g, dtype=jnp.float64)
    if Lambda_g.shape[0] != T:
        raise ValueError("Lambda_g must have shape (T, d_g).")

    rho_bm_scaled = Dm * rho_bm
    rho_bg_scaled = Dg * rho_bg

    leverage_sq = 1.0 - jnp.sum(rho_bm_scaled**2, axis=1) - jnp.sum(rho_bg_scaled**2, axis=1)
    leverage_sq = jnp.clip(leverage_sq, a_min=0.0)
    leverage = jnp.sqrt(leverage_sq)

    sigma_m_scaled = jnp.einsum("ij,tj->ti", sigma_m, rho_bm_scaled)
    m_contrib = jnp.einsum("ti,i->t", sigma_m_scaled, rho_bm)

    sigma_g_scaled = jnp.einsum("ij,tj->ti", sigma_g, Dg * Lambda_g)
    g_contrib = jnp.einsum("ti,i->t", sigma_g_scaled, rho_bg)

    sigma_gm_scaled = jnp.einsum("ij,tj->ti", sigma_gm, rho_bm_scaled)
    gm_contrib = jnp.einsum("ti,i->t", sigma_gm_scaled, rho_bg)

    drift = jnp.log1p(r_t) - (m_contrib + g_contrib + gm_contrib) - 0.5 * (scale**2) * leverage_sq
    shock = scale * leverage * eps_b
    log_growth = drift + shock
    growth = jnp.exp(log_growth)

    if config is not None and getattr(config, "use_truncation", False):
        growth_for_logs = jnp.clip(growth, a_min=truncation_floor)
    else:
        growth_for_logs = growth

    log_B = jnp.log(jnp.asarray(B0, dtype=jnp.float64)) + jnp.cumsum(jnp.log(growth_for_logs))
    B_path = jnp.exp(log_B)

    return {"growth": growth, "B": B_path}


__all__ = ["generate_paths", "compute_lambda_g", "propagate_B"]
