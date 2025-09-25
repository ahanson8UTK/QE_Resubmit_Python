"""State-space design utilities for the CW2017 bond-pricing model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp

from .utils import cholesky_psd, compute_diag_scales

Array = jnp.ndarray


@dataclass(frozen=True)
class Block3aKalmanDesign:
    """Container holding the linear-dynamical-system inputs for block 3a."""

    observations: Array
    measurement_matrix: Array
    measurement_noise_chol: Array
    transition_matrix: Array
    state_offsets: Array
    process_noise_chol: Array
    initial_mean: Array
    initial_sqrt_cov: Array


def _asarray(x: Array) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def build_block3a_design(
    params3a: any,
    fixed: Dict[str, Array],
    measurement_terms: Tuple[Array, Array, Array, Array, Array],
    y_t: Array,
    m_t: Array,
    h_t: Array,
    maturities: Iterable[int],
) -> Block3aKalmanDesign:
    """Return the arrays required by the square-root Kalman filter.

    The construction follows the six-state system in Section 1.3 of
    :mod:`docs/model_writeup`, matching Algorithm Overview §3.1.d.  The
    observation stack is ``[m_t, y_t, h_t]`` while the dynamic state vector is
    ``[m_t, g_t, h_t, \bar{\mu}_m, \bar{\mu}_g^u, \bar{\mu}_g^{\mathbb{Q},u}]``.
    """

    del maturities  # The measurement terms already encode the maturity set.

    A0, A1, B, M0Q, M1Q = measurement_terms
    y_t = _asarray(y_t)
    m_t = _asarray(m_t)
    h_t = _asarray(h_t)

    T = y_t.shape[0]
    d_m = m_t.shape[1]
    d_y = y_t.shape[1]
    d_h = h_t.shape[1]
    d_g = B.shape[1]

    M1 = _asarray(fixed["M1"])
    d_mu_gu = M1.shape[1]
    d_mu_gq = M1Q.shape[1]

    mu_m_bar = _asarray(fixed["mu_m_bar"])
    mu_gu_bar = _asarray(fixed["mu_g^u_bar"])
    mu_gq_bar = _asarray(fixed["mu_g^{Q,u}"])
    mu_h_bar = _asarray(fixed["mu_h_bar"])

    Phi_m = _asarray(fixed["Phi_m"])
    Phi_mg = _asarray(fixed["Phi_mg"])
    Phi_mh = _asarray(fixed["Phi_mh"])
    Phi_h = _asarray(fixed["Phi_h"])

    Phi_blocks = params3a.PhiP_blocks
    Phi_gm = _asarray(Phi_blocks.Phi_gm)
    Phi_gh = _asarray(Phi_blocks.Phi_gh)
    Phi_g = _asarray(Phi_blocks.Phi_gg)

    bar_blocks = fixed["bar"]
    bar_mm = _asarray(bar_blocks["mm"])
    bar_mg = _asarray(bar_blocks["mg"])
    bar_mh = _asarray(bar_blocks["mh"])
    bar_gm = _asarray(bar_blocks["gm"])
    bar_gg = _asarray(bar_blocks["gg"])
    bar_gh = _asarray(bar_blocks["gh"])
    bar_hh = _asarray(bar_blocks["hh"])

    Sigma_m = _asarray(fixed["Sigma_m"])
    Sigma_gm = _asarray(fixed["Sigma_gm"])
    Sigma_hm = _asarray(fixed["Sigma_hm"])
    Sigma_hg = _asarray(fixed["Sigma_hg"])
    Sigma_h = _asarray(fixed["Sigma_h"])
    Sigma_g = _asarray(params3a.Sigma_g)

    Gamma0 = _asarray(params3a.Gamma0)
    Gamma1 = _asarray(params3a.Gamma1)

    # Observation residuals after subtracting the deterministic offsets.
    obs_offset_y = (_asarray(A0) + _asarray(A1) @ _asarray(M0Q)).reshape(-1)
    obs_offset = jnp.concatenate(
        [
            jnp.zeros((d_m,), dtype=jnp.float64),
            obs_offset_y,
            jnp.zeros((d_h,), dtype=jnp.float64),
        ]
    )
    obs_offset = jnp.broadcast_to(obs_offset, (T, d_m + d_y + d_h))
    observations = jnp.concatenate(
        [m_t, y_t, h_t], axis=1
    ) - obs_offset

    state_dim = d_m + d_g + d_h + d_m + d_mu_gu + d_mu_gq

    # Measurement matrix following the block structure in Section 1.3.
    measurement_matrix = jnp.zeros((d_m + d_y + d_h, state_dim), dtype=jnp.float64)
    row_m = slice(0, d_m)
    row_y = slice(d_m, d_m + d_y)
    row_h = slice(d_m + d_y, d_m + d_y + d_h)
    idx_m = slice(0, d_m)
    idx_g = slice(d_m, d_m + d_g)
    idx_h = slice(d_m + d_g, d_m + d_g + d_h)
    idx_mu_m = slice(d_m + d_g + d_h, d_m + d_g + d_h + d_m)
    idx_mu_gu = slice(idx_mu_m.stop, idx_mu_m.stop + d_mu_gu)
    idx_mu_gq = slice(idx_mu_gu.stop, idx_mu_gu.stop + d_mu_gq)

    measurement_matrix = measurement_matrix.at[row_m, idx_m].set(jnp.eye(d_m))
    measurement_matrix = measurement_matrix.at[row_y, idx_g].set(_asarray(B))
    measurement_matrix = measurement_matrix.at[row_y, idx_mu_gq].set(
        _asarray(A1) @ _asarray(M1Q)
    )
    measurement_matrix = measurement_matrix.at[row_h, idx_h].set(jnp.eye(d_h))

    # Observation noise: zero for m/h, Omega for the yields.
    R = jnp.zeros((d_m + d_y + d_h, d_m + d_y + d_h), dtype=jnp.float64)
    det_noise = 1e-6
    det_sqrt = jnp.sqrt(det_noise)
    if d_m > 0:
        R = R.at[row_m, row_m].set(det_sqrt * jnp.eye(d_m, dtype=jnp.float64))
    Omega_sqrt = jnp.sqrt(_asarray(params3a.Omega_diag))
    R = R.at[row_y, row_y].set(
        jnp.diag(Omega_sqrt)
    )
    if d_h > 0:
        R = R.at[row_h, row_h].set(det_sqrt * jnp.eye(d_h, dtype=jnp.float64))

    # Transition matrix in block form.
    transition_matrix = jnp.zeros((state_dim, state_dim), dtype=jnp.float64)
    transition_matrix = transition_matrix.at[idx_m, idx_m].set(Phi_m)
    transition_matrix = transition_matrix.at[idx_m, idx_g].set(Phi_mg)
    transition_matrix = transition_matrix.at[idx_m, idx_h].set(Phi_mh)
    transition_matrix = transition_matrix.at[idx_m, idx_mu_m].set(bar_mm)
    transition_matrix = transition_matrix.at[idx_m, idx_mu_gu].set(bar_mg)

    transition_matrix = transition_matrix.at[idx_g, idx_m].set(Phi_gm)
    transition_matrix = transition_matrix.at[idx_g, idx_g].set(Phi_g)
    transition_matrix = transition_matrix.at[idx_g, idx_h].set(Phi_gh)
    transition_matrix = transition_matrix.at[idx_g, idx_mu_m].set(bar_gm)
    transition_matrix = transition_matrix.at[idx_g, idx_mu_gu].set(bar_gg)

    transition_matrix = transition_matrix.at[idx_h, idx_h].set(Phi_h)

    transition_matrix = transition_matrix.at[idx_mu_m, idx_mu_m].set(jnp.eye(d_m))
    transition_matrix = transition_matrix.at[idx_mu_gu, idx_mu_gu].set(jnp.eye(d_mu_gu))
    transition_matrix = transition_matrix.at[idx_mu_gq, idx_mu_gq].set(jnp.eye(d_mu_gq))

    offset_vec = jnp.zeros((state_dim,), dtype=jnp.float64)
    offset_vec = offset_vec.at[idx_m].set(bar_mh @ mu_h_bar)
    offset_vec = offset_vec.at[idx_g].set(bar_gh @ mu_h_bar)
    offset_vec = offset_vec.at[idx_h].set(bar_hh @ mu_h_bar)
    state_offsets = jnp.broadcast_to(offset_vec, (T, state_dim))

    # Diagonal scaling for ``D_{m,t}`` and ``D_{g,t}``.
    diag_scales = compute_diag_scales(Gamma0, Gamma1, h_t)
    diag_m = diag_scales[:, :d_m]
    diag_g = diag_scales[:, d_m: d_m + d_g]

    if d_m > 0:
        h_lag = jnp.concatenate([h_t[:1], h_t[:-1]], axis=0)
        gamma_last = Gamma1[d_m - 1]
        last_exp = Gamma0[d_m - 1] + h_lag @ gamma_last
        last_exp = jnp.clip(last_exp, -20.0, 20.0)
        diag_last = jnp.exp(0.5 * last_exp)
        diag_m = diag_m.at[:, d_m - 1].set(diag_last)

    def process_chol(diag_m_t: Array, diag_g_t: Array) -> Array:
        noise_dim = d_m + d_g + d_h
        G = jnp.zeros((state_dim, noise_dim), dtype=jnp.float64)

        G = G.at[idx_m, :d_m].set(Sigma_m * diag_m_t[None, :])
        G = G.at[idx_g, :d_m].set(Sigma_gm * diag_m_t[None, :])
        G = G.at[idx_g, d_m : d_m + d_g].set(Sigma_g * diag_g_t[None, :])
        G = G.at[idx_h, :d_m].set(Sigma_hm)
        G = G.at[idx_h, d_m : d_m + d_g].set(Sigma_hg)
        G = G.at[idx_h, d_m + d_g :].set(Sigma_h)
        cov = G @ G.T
        return cholesky_psd(cov)

    process_noise_chol = jax.vmap(process_chol)(diag_m, diag_g)

    g0 = _asarray(fixed.get("g0_mean", jnp.zeros((d_g,), dtype=jnp.float64)))
    initial_mean = jnp.concatenate(
        [m_t[0], g0, h_t[0], mu_m_bar, mu_gu_bar, mu_gq_bar]
    )
    initial_sqrt_cov = process_noise_chol[0]

    return Block3aKalmanDesign(
        observations=observations,
        measurement_matrix=measurement_matrix,
        measurement_noise_chol=R,
        transition_matrix=transition_matrix,
        state_offsets=state_offsets,
        process_noise_chol=process_noise_chol,
        initial_mean=initial_mean,
        initial_sqrt_cov=initial_sqrt_cov,
    )


__all__ = ["Block3aKalmanDesign", "build_block3a_design"]

