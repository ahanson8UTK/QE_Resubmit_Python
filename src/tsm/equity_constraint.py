"""Equity inequality margin for the covariance block sampler."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def equity_margin(
    theta_m: np.ndarray,
    theta_g: np.ndarray,
    theta_g_Q: np.ndarray,
    mu_m_bar: np.ndarray,
    mu_g_u_bar: np.ndarray,
    mu_g_Q_u_bar: np.ndarray,
    Phi_m: np.ndarray,
    Phi_mg: np.ndarray,
    Phi_mh: np.ndarray,
    Phi_h: np.ndarray,
    M0_Q: np.ndarray,
    M1_Q: np.ndarray,
    Sigma_g_Q: np.ndarray,
    Sigma_hm: np.ndarray,
    Sigma_hg: np.ndarray,
    Sigma_h: np.ndarray,
    mu_h_bar: np.ndarray,
    Phi_g_Q: np.ndarray,
) -> float:
    """
    Returns LHS = theta_m mu_m_bar + theta_g mu_g_u_bar + theta_g_Q mu_g_Q_u_bar + V(...)
    where
    V = [e_d'(I-Phi_m)^{-1}Phi_mh_bar + e_d' Phi_mh Phi_h_bar] mu_h_bar
        + [e_d'(I-Phi_m)^{-1}Phi_mg - e_1'] M0_Q
        + 1/2 [e_d'(I-Phi_m)^{-1}Phi_mg - e_1'] (I-Phi_g^Q)^{-1} Sigma_g^Q(
              [e_d'(I-Phi_m)^{-1}Phi_mg - e_1'] (I-Phi_g^Q)^{-1} Sigma_g^Q
              + e_d' Phi_mh Sigma_hg )
        + 1/2 e_d' Phi_mh [Sigma_hm Sigma_hm' + Sigma_h Sigma_h'] Phi_mh' e_d
        + 1/2 (e_d' Phi_mh Sigma_hg)(e_d' Phi_mh Sigma_hg)'.
    """

    theta_m = np.asarray(theta_m, dtype=np.float64)
    theta_g = np.asarray(theta_g, dtype=np.float64)
    theta_g_Q = np.asarray(theta_g_Q, dtype=np.float64)
    mu_m_bar = np.asarray(mu_m_bar, dtype=np.float64)
    mu_g_u_bar = np.asarray(mu_g_u_bar, dtype=np.float64)
    mu_g_Q_u_bar = np.asarray(mu_g_Q_u_bar, dtype=np.float64)
    Phi_m = np.asarray(Phi_m, dtype=np.float64)
    Phi_mg = np.asarray(Phi_mg, dtype=np.float64)
    Phi_mh = np.asarray(Phi_mh, dtype=np.float64)
    Phi_h = np.asarray(Phi_h, dtype=np.float64)
    M0_Q = np.asarray(M0_Q, dtype=np.float64)
    Sigma_g_Q = np.asarray(Sigma_g_Q, dtype=np.float64)
    Sigma_hm = np.asarray(Sigma_hm, dtype=np.float64)
    Sigma_hg = np.asarray(Sigma_hg, dtype=np.float64)
    Sigma_h = np.asarray(Sigma_h, dtype=np.float64)
    mu_h_bar = np.asarray(mu_h_bar, dtype=np.float64)
    Phi_g_Q = np.asarray(Phi_g_Q, dtype=np.float64)

    d_m = Phi_m.shape[0]
    d_g = Sigma_g_Q.shape[0]
    e_d = np.zeros(d_m, dtype=np.float64)
    e_d[-1] = 1.0
    e1 = np.zeros(d_g, dtype=np.float64)
    e1[0] = 1.0

    I_m = np.eye(d_m, dtype=np.float64)
    solve_m = linalg.solve(I_m - Phi_m, I_m, assume_a="gen", check_finite=False)

    d_h = Phi_h.shape[0]
    I_h = np.eye(d_h, dtype=np.float64)
    Phi_h_bar = linalg.solve(I_h - Phi_h, I_h, assume_a="gen", check_finite=False)
    Phi_mh_bar = Phi_mh @ Phi_h_bar

    term1_vec = e_d @ solve_m @ Phi_mh_bar + e_d @ Phi_mh @ Phi_h_bar
    term1 = float(term1_vec @ mu_h_bar)

    row_q = e_d @ solve_m @ Phi_mg - e1
    term2 = float(row_q @ M0_Q)

    I_g = np.eye(d_g, dtype=np.float64)
    solve_g = linalg.solve(I_g - Phi_g_Q, Sigma_g_Q, assume_a="gen", check_finite=False)
    vec = row_q @ solve_g

    add = (e_d @ Phi_mh) @ Sigma_hg
    term3 = 0.5 * float(vec @ (vec + add))

    Sigma_block = Sigma_hm @ Sigma_hm.T + Sigma_h @ Sigma_h.T
    middle = (e_d @ Phi_mh) @ Sigma_block @ (Phi_mh.T @ e_d)
    term4 = 0.5 * float(middle)

    term5 = 0.5 * float(add @ add)

    V = term1 + term2 + term3 + term4 + term5

    const = float(theta_m @ mu_m_bar + theta_g @ mu_g_u_bar + theta_g_Q @ mu_g_Q_u_bar)

    return const + V
