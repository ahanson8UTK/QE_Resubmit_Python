"""State-space builders for Block 3c sampling.

The layouts follow Form I in Appendix B.1.1 of Creal & Wu (2017,
*International Economic Review*) together with the yield rotations from
Appendix A.2.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax.numpy as jnp

from kalman import Step
from utils.selectors import block_diag

ArrayDict = Dict[str, jnp.ndarray]
ConfigDict = Dict[str, Dict[str, int]]


def _q_g_process_cov(
    sigma_gm: jnp.ndarray, sigma_g: jnp.ndarray, d_g_t: jnp.ndarray
) -> jnp.ndarray:
    """Return ``Q_{g,t}`` for the :math:`g_t` innovation."""

    sigma_gm64 = jnp.asarray(sigma_gm, dtype=jnp.float64)
    sigma_g64 = jnp.asarray(sigma_g, dtype=jnp.float64)
    diag_scale = jnp.asarray(d_g_t, dtype=jnp.float64)

    sigma_g_scaled = sigma_g64 * diag_scale
    return sigma_gm64 @ sigma_gm64.T + sigma_g_scaled @ sigma_g_scaled.T


def build_steps_augmented(
    fixed: ArrayDict,
    data: ArrayDict,
    cfg: ConfigDict,
    priors: ArrayDict,
    state: Dict,
) -> Tuple[Callable[[int], Step], jnp.ndarray, jnp.ndarray, Dict[str, slice]]:
    """Return the Form I LDS with static means kept in the state.

    The construction follows Appendix B.1.1 (Form I) of Creal & Wu (2017,
    *International Economic Review*) while mapping the measurement block using
    the rotations from Appendix A.2.
    """

    dims = cfg["dims"]
    d_m = int(dims["d_m"])
    d_g = int(dims["d_g"])
    d_h = int(dims["d_h"])
    d_y = int(dims["d_y"])

    phi_g = jnp.asarray(fixed["Phi_g"], dtype=jnp.float64)
    phi_gm = jnp.asarray(fixed["Phi_gm"], dtype=jnp.float64)
    phi_gh = jnp.asarray(fixed["Phi_gh"], dtype=jnp.float64)

    bar_blocks = fixed["bar"]
    bar_gm = jnp.asarray(bar_blocks["gm"], dtype=jnp.float64)
    bar_gg = jnp.asarray(bar_blocks["gg"], dtype=jnp.float64)
    bar_gh = jnp.asarray(bar_blocks["gh"], dtype=jnp.float64)

    m1 = jnp.asarray(fixed["M1"], dtype=jnp.float64)
    m1_q = jnp.asarray(fixed["M1Q"], dtype=jnp.float64)
    m0_q = jnp.asarray(fixed["M0Q"], dtype=jnp.float64)

    a0 = jnp.asarray(fixed["A0"], dtype=jnp.float64)
    a1 = jnp.asarray(fixed["A1"], dtype=jnp.float64)
    b_mat = jnp.asarray(fixed["B"], dtype=jnp.float64)
    omega_diag = jnp.asarray(fixed["Omega_diag"], dtype=jnp.float64)

    mu_h_bar = jnp.asarray(fixed["mu_h_bar"], dtype=jnp.float64)

    sigma_gm = jnp.asarray(fixed["Sigma_gm"], dtype=jnp.float64)
    sigma_g = jnp.asarray(fixed["Sigma_g"], dtype=jnp.float64)

    m_obs = jnp.asarray(data["m"], dtype=jnp.float64)
    h_obs = jnp.asarray(data["h"], dtype=jnp.float64)
    d_g_series = jnp.asarray(data["Dg"], dtype=jnp.float64)

    idx_g = slice(0, d_g)
    idx_mu_m = slice(d_g, d_g + d_m)
    idx_mu_gu = slice(d_g + d_m, d_g + d_m + 2)
    idx_mu_gq = slice(d_g + d_m + 2, d_g + d_m + 4)
    idx_static = slice(d_g, d_g + d_m + 4)

    g0_mean = jnp.asarray(
        state.get("g0_mean", jnp.zeros((d_g,), dtype=jnp.float64)),
        dtype=jnp.float64,
    )
    g0_cov = jnp.asarray(
        state.get("g0_cov", jnp.eye(d_g, dtype=jnp.float64) * 1e-2),
        dtype=jnp.float64,
    )

    mu_m_mean = jnp.asarray(priors["mu_m_mean"], dtype=jnp.float64)
    mu_m_cov = jnp.asarray(priors["mu_m_cov"], dtype=jnp.float64)
    mu_gu_mean = jnp.asarray(priors["mu_gu_mean"], dtype=jnp.float64)
    mu_gu_cov = jnp.asarray(priors["mu_gu_cov"], dtype=jnp.float64)
    mu_gq_mean = jnp.asarray(priors["mu_gQu_mean"], dtype=jnp.float64)
    mu_gq_cov = jnp.asarray(priors["mu_gQu_cov"], dtype=jnp.float64)

    x0_mean = jnp.concatenate([g0_mean, mu_m_mean, mu_gu_mean, mu_gq_mean])
    x0_cov = block_diag(g0_cov, mu_m_cov, mu_gu_cov, mu_gq_cov)

    h_mat = jnp.diag(omega_diag)
    z_template = jnp.zeros((d_y, d_g + d_m + 4), dtype=jnp.float64)
    z_template = z_template.at[:, idx_g].set(b_mat)
    z_template = z_template.at[:, idx_mu_gq].set(a1 @ m1_q)

    t_template = jnp.eye(d_g + d_m + 4, dtype=jnp.float64)
    t_template = t_template.at[idx_g, idx_g].set(phi_g)
    t_template = t_template.at[idx_g, idx_mu_m].set(bar_gm)
    t_template = t_template.at[idx_g, idx_mu_gu].set(bar_gg @ m1)

    r_template = jnp.zeros((d_g + d_m + 4, d_g), dtype=jnp.float64)
    r_template = r_template.at[idx_g, :].set(jnp.eye(d_g, dtype=jnp.float64))

    d_vec = a0 + a1 @ m0_q

    def steps(t: int) -> Step:
        q_g = _q_g_process_cov(sigma_gm, sigma_g, d_g_series[t])

        c_vec = jnp.zeros((d_g + d_m + 4,), dtype=jnp.float64)
        c_top = phi_gm @ m_obs[t] + phi_gh @ h_obs[t] + bar_gh @ mu_h_bar
        c_vec = c_vec.at[idx_g].set(c_top)

        return Step(
            Z=z_template,
            d=d_vec,
            H=h_mat,
            T=t_template,
            c=c_vec,
            R=r_template,
            Q=q_g,
        )

    slices = {
        "g": idx_g,
        "mu_m": idx_mu_m,
        "mu_gu": idx_mu_gu,
        "mu_gQ_u": idx_mu_gq,
        "static_all": idx_static,
    }

    return steps, x0_mean, x0_cov, slices


def build_steps_reduced(
    fixed: ArrayDict,
    data: ArrayDict,
    cfg: ConfigDict,
    mus: Dict[str, jnp.ndarray],
) -> Tuple[Callable[[int], Step], jnp.ndarray, jnp.ndarray]:
    """Return the Form I LDS with static means absorbed into the offsets."""

    dims = cfg["dims"]
    d_m = int(dims["d_m"])
    d_g = int(dims["d_g"])
    d_h = int(dims["d_h"])
    d_y = int(dims["d_y"])

    phi_g = jnp.asarray(fixed["Phi_g"], dtype=jnp.float64)
    phi_gm = jnp.asarray(fixed["Phi_gm"], dtype=jnp.float64)
    phi_gh = jnp.asarray(fixed["Phi_gh"], dtype=jnp.float64)

    bar_blocks = fixed["bar"]
    bar_gm = jnp.asarray(bar_blocks["gm"], dtype=jnp.float64)
    bar_gg = jnp.asarray(bar_blocks["gg"], dtype=jnp.float64)
    bar_gh = jnp.asarray(bar_blocks["gh"], dtype=jnp.float64)

    m1 = jnp.asarray(fixed["M1"], dtype=jnp.float64)
    m1_q = jnp.asarray(fixed["M1Q"], dtype=jnp.float64)
    m0_q = jnp.asarray(fixed["M0Q"], dtype=jnp.float64)

    a0 = jnp.asarray(fixed["A0"], dtype=jnp.float64)
    a1 = jnp.asarray(fixed["A1"], dtype=jnp.float64)
    b_mat = jnp.asarray(fixed["B"], dtype=jnp.float64)
    omega_diag = jnp.asarray(fixed["Omega_diag"], dtype=jnp.float64)

    mu_h_bar = jnp.asarray(fixed["mu_h_bar"], dtype=jnp.float64)

    sigma_gm = jnp.asarray(fixed["Sigma_gm"], dtype=jnp.float64)
    sigma_g = jnp.asarray(fixed["Sigma_g"], dtype=jnp.float64)

    m_obs = jnp.asarray(data["m"], dtype=jnp.float64)
    h_obs = jnp.asarray(data["h"], dtype=jnp.float64)
    d_g_series = jnp.asarray(data["Dg"], dtype=jnp.float64)

    mu_m = jnp.asarray(mus["mu_m"], dtype=jnp.float64)
    mu_gu = jnp.asarray(mus["mu_gu"], dtype=jnp.float64)
    mu_gq = jnp.asarray(mus["mu_gQ_u"], dtype=jnp.float64)

    mu_g = m1 @ mu_gu

    z_mat = jnp.asarray(b_mat, dtype=jnp.float64)
    h_mat = jnp.diag(omega_diag)
    a1_m1q = a1 @ m1_q
    d_vec = a0 + a1 @ m0_q + a1_m1q @ mu_gq

    r_mat = jnp.eye(d_g, dtype=jnp.float64)

    def steps(t: int) -> Step:
        c_vec = (
            phi_gm @ m_obs[t]
            + phi_gh @ h_obs[t]
            + bar_gm @ mu_m
            + bar_gg @ mu_g
            + bar_gh @ mu_h_bar
        )
        q_g = _q_g_process_cov(sigma_gm, sigma_g, d_g_series[t])

        return Step(
            Z=z_mat,
            d=d_vec,
            H=h_mat,
            T=phi_g,
            c=c_vec,
            R=r_mat,
            Q=q_g,
        )

    x0_mean = jnp.zeros((d_g,), dtype=jnp.float64)
    x0_cov = jnp.eye(d_g, dtype=jnp.float64) * 1e-2
    return steps, x0_mean, x0_cov


__all__ = ["build_steps_augmented", "build_steps_reduced"]
