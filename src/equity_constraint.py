"""Equity constraint coefficients derived from Creal & Wu (2017)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from utils.selectors import e_last, e1
from tsm.q_covariance import compute_sigma_g_Q


def compute_M0Q_from_B_and_Sigma_gQ(
    B_rows: jnp.ndarray,
    *,
    Sigma_g: jnp.ndarray,
    Gamma0: jnp.ndarray,
    Gamma1: jnp.ndarray,
    mu_h_bar: jnp.ndarray,
    n_star: int = 60,
) -> jnp.ndarray:
    r"""Return :math:`M_0^Q` using Appendix A.2 of Creal & Wu (2017).

    Parameters
    ----------
    B_rows:
        Matrix whose ``i``-th row stores :math:`b_{i,g}` for horizon ``i``. The
        matrix must include at least ``n_star - 1`` rows.
    Sigma_g:
        Lower-triangular matrix :math:`\Sigma_g` (Appendix B.4) whose Q-measure
        rotation is constructed internally.
    Gamma0, Gamma1, mu_h_bar:
        Parameters determining the long-run diagonal scaling entering
        :math:`\Sigma_g^Q`.
    n_star:
        Horizon count :math:`n^*` used for the annuity averaging.

    Returns
    -------
    jax.Array
        Vector ``(3,)`` holding ``M_0^Q`` as defined in Appendix A.2.
    """

    if n_star < 2:
        raise ValueError("n_star must be at least 2")

    B64 = jnp.asarray(B_rows, dtype=jnp.float64)
    if B64.shape[0] < n_star - 1:
        raise ValueError("B_rows must have at least n_star - 1 rows")
    if B64.ndim != 2:
        raise ValueError("B_rows must be a matrix")
    Sigma_g_np = np.asarray(Sigma_g, dtype=np.float64)
    Gamma0_np = np.asarray(Gamma0, dtype=np.float64)
    Gamma1_np = np.asarray(Gamma1, dtype=np.float64)
    mu_h_np = np.asarray(mu_h_bar, dtype=np.float64)

    if Sigma_g_np.ndim != 2 or Sigma_g_np.shape[0] != Sigma_g_np.shape[1]:
        raise ValueError("Sigma_g must be square")
    if B64.shape[1] != Sigma_g_np.shape[0]:
        raise ValueError("Column dimension mismatch between B_rows and Sigma_g")

    d_g = Sigma_g_np.shape[0]
    total = Gamma0_np.shape[0]
    d_m = total - d_g
    if d_m <= 0:
        raise ValueError("Gamma0 must have length greater than d_g")
    if Gamma1_np.shape != (total, mu_h_np.shape[0]):
        raise ValueError("Gamma1 must have shape (d_m + d_g, d_h)")

    Sigma_g_Q_np = compute_sigma_g_Q(
        Sigma_g_np,
        Gamma0_np,
        Gamma1_np,
        mu_h_np,
        d_m,
        d_g,
    )
    Sigma64 = jnp.asarray(Sigma_g_Q_np, dtype=jnp.float64)

    GG = Sigma64 @ Sigma64.T
    inner = jnp.float64(0.0)
    for i in range(1, n_star):
        b = B64[i - 1, :]
        inner = inner + (i**2) * (b @ GG @ b)
    top = -0.5 * inner / n_star
    return jnp.array([top, 0.0, 0.0], dtype=jnp.float64)


def equity_coeffs(fixed: dict[str, jnp.ndarray], cfg: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute ``(a, b)`` for ``a^T \mu + b < 0`` (Appendix B.1.1, A.2).

    The coefficient vectors follow the Form I restriction from Creal & Wu (2017,
    Appendix B.1.1) while mapping Q-measure quantities using the rotations in
    Appendix A.2.  The state mean is ordered as
    ``mu = concat([mu_m_bar, mu_g_u_bar, mu_gQ_u_bar])`` with ``d_g = 3``.
    """

    d_m = int(cfg["dims"]["d_m"])

    e_d = e_last(d_m)
    e1g = e1(3)

    Phi_m = jnp.asarray(fixed["Phi_m"], dtype=jnp.float64)
    Phi_mg = jnp.asarray(fixed["Phi_mg"], dtype=jnp.float64)
    Phi_mh = jnp.asarray(fixed["Phi_mh"], dtype=jnp.float64)
    Phi_gQ = jnp.asarray(fixed["Phi_gQ"], dtype=jnp.float64)
    Sigma_h = jnp.asarray(fixed["Sigma_h"], dtype=jnp.float64)
    Sigma_hm = jnp.asarray(fixed["Sigma_hm"], dtype=jnp.float64)
    Sigma_hg = jnp.asarray(fixed["Sigma_hg"], dtype=jnp.float64)
    mu_hbar = jnp.asarray(fixed["mu_h_bar"], dtype=jnp.float64)
    Sigma_g = np.asarray(fixed["Sigma_g"], dtype=np.float64)
    Gamma0 = np.asarray(fixed["Gamma0"], dtype=np.float64)
    Gamma1 = np.asarray(fixed["Gamma1"], dtype=np.float64)

    bar = fixed["bar"]
    bar_mm = jnp.asarray(bar["mm"], dtype=jnp.float64)
    bar_mg = jnp.asarray(bar["mg"], dtype=jnp.float64)
    bar_mh = jnp.asarray(bar["mh"], dtype=jnp.float64)
    bar_hh = jnp.asarray(bar["hh"], dtype=jnp.float64)

    M0Q = jnp.asarray(fixed["M0Q"], dtype=jnp.float64)
    M1Q = jnp.asarray(fixed["M1Q"], dtype=jnp.float64)

    Sg_free = jnp.asarray(fixed["Sg_free"], dtype=jnp.float64)

    Im = jnp.eye(d_m, dtype=jnp.float64)
    solve_Im_Phi_m = jax.scipy.linalg.solve(Im - Phi_m, Im, assume_a="gen")

    theta_m = e_d @ solve_Im_Phi_m @ bar_mm
    theta_g = e_d @ solve_Im_Phi_m @ bar_mg

    row_q = (e_d @ solve_Im_Phi_m @ Phi_mg) - e1g
    theta_gQ = row_q @ M1Q

    term1 = (e_d @ solve_Im_Phi_m @ bar_mh + e_d @ Phi_mh @ bar_hh) @ mu_hbar
    term2 = row_q @ M0Q

    d_g = Sigma_g.shape[0]
    Sigma_g_Q_np = compute_sigma_g_Q(
        Sigma_g,
        Gamma0,
        Gamma1,
        np.asarray(mu_hbar),
        d_m,
        d_g,
    )
    Sigma_gQ = jnp.asarray(Sigma_g_Q_np, dtype=jnp.float64)
    if Phi_gQ.shape[0] != d_g:
        raise ValueError("Phi_gQ dimension must match Sigma_g")
    Ig = jnp.eye(d_g, dtype=jnp.float64)
    S = jax.scipy.linalg.solve(Ig - Phi_gQ, Sigma_gQ, assume_a="gen")
    vec = row_q @ S
    add = (e_d @ Phi_mh) @ Sigma_hg

    term3 = 0.5 * (vec @ (vec + add))
    Sigma_block = Sigma_hm @ Sigma_hm.T + Sigma_h @ Sigma_h.T
    term4 = 0.5 * (e_d @ Phi_mh) @ Sigma_block @ (Phi_mh.T @ e_d)
    term5 = 0.5 * (add @ add)

    V = term1 + term2 + term3 + term4 + term5

    theta_g_free = Sg_free @ theta_g

    a = jnp.concatenate([theta_m, theta_g_free, theta_gQ])
    b = jnp.asarray(V, dtype=jnp.float64)
    return a, b
