from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

Array = np.ndarray


@dataclass
class EquityPricingInputs:
    """Model primitives for the ATSM equity recursion.

    Dimensions
    ----------
    d_m : int
        Number of macro states :math:`m_t`.
    d_g : int
        Number of yield states :math:`g_t`.
    d_h : int
        Number of volatility states :math:`h_t`.
    """

    d_m: int
    d_g: int
    d_h: int

    # P-measure transition blocks
    Phi_m: Array          # (d_m, d_m)
    Phi_mg: Array         # (d_m, d_g)
    Phi_mh: Array         # (d_m, d_h)
    Phi_h: Array          # (d_h, d_h)

    # g-dynamics under P and Q
    Phi_g: Array          # (d_g, d_g)
    Phi_g_Q: Array        # (d_g, d_g)
    Phi_gm: Array         # (d_g, d_m)
    Phi_gh: Array         # (d_g, d_h)

    # Covariances (time-invariant)
    Sigma_m: Array        # (d_m, d_m)
    Sigma_g: Array        # (d_g, d_g)
    Sigma_g_Q: Array      # (d_g, d_g)
    Sigma_hm: Array       # (d_h, d_m)
    Sigma_hg: Array       # (d_h, d_g)
    Sigma_h: Array        # (d_h, d_h)

    # Means
    mu_m: Array           # (d_m,)
    mu_g: Array           # (d_g,)
    mu_g_Q: Array         # (d_g,)
    mu_h: Array           # (d_h,)

    # Volatility mapping for diag(D_{m,t}), diag(D_{g,t})
    Gamma0: Array         # (d_m + d_g,)
    Gamma1: Array         # (d_m + d_g, d_h)

    # Dividend-variance parameters
    gamma_dd0: float
    gamma_dd2: Array      # (d_h,)

    # Indices
    e_div_ix: Optional[int] = None
    e1_g_ix: int = 0

    def __post_init__(self) -> None:
        if self.e_div_ix is None:
            self.e_div_ix = self.d_m - 1

        dm, dg, dh = self.d_m, self.d_g, self.d_h

        assert self.Phi_m.shape == (dm, dm)
        assert self.Phi_mg.shape == (dm, dg)
        assert self.Phi_mh.shape == (dm, dh)
        assert self.Phi_h.shape == (dh, dh)

        assert self.Phi_g.shape == (dg, dg)
        assert self.Phi_g_Q.shape == (dg, dg)
        assert self.Phi_gm.shape == (dg, dm)
        assert self.Phi_gh.shape == (dg, dh)

        assert self.Sigma_m.shape == (dm, dm)
        assert self.Sigma_g.shape == (dg, dg)
        assert self.Sigma_g_Q.shape == (dg, dg)
        assert self.Sigma_hm.shape == (dh, dm)
        assert self.Sigma_hg.shape == (dh, dg)
        assert self.Sigma_h.shape == (dh, dh)

        assert self.mu_m.shape == (dm,)
        assert self.mu_g.shape == (dg,)
        assert self.mu_g_Q.shape == (dg,)
        assert self.mu_h.shape == (dh,)

        assert self.Gamma0.shape == (dm + dg,)
        assert self.Gamma1.shape == (dm + dg, dh)

        assert self.gamma_dd2.shape == (dh,)

        assert 0 <= self.e_div_ix < dm
        assert 0 <= self.e1_g_ix < dg


def _row(length: int, ix: int) -> Array:
    e = np.zeros(length)
    e[ix] = 1.0
    return e


def price_equity_ratio(
    inp: EquityPricingInputs,
    m_t: Array,
    g_t: Array,
    h_t: Array,
    n_max: int = 600,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    min_terms: int = 5,
) -> Array:
    """Compute the dividend-price ratio series via the ATSM recursion.

    Parameters
    ----------
    inp
        Model inputs.
    m_t, g_t, h_t
        State realizations with shapes ``(T, d_m)``, ``(T, d_g)``, and ``(T, d_h)``.
    n_max
        Maximum number of recursion terms.
    rtol, atol
        Relative and absolute tolerances for truncating the sum.
    min_terms
        Minimum number of terms before applying the truncation test.
    """

    T, dm, dg, dh = m_t.shape[0], inp.d_m, inp.d_g, inp.d_h
    if m_t.shape != (T, dm) or g_t.shape != (T, dg) or h_t.shape != (T, dh):
        raise ValueError("State arrays have inconsistent shapes with model dimensions.")

    e_d = _row(dm, inp.e_div_ix)
    e1 = _row(dg, inp.e1_g_ix)

    Phi_m, Phi_mg, Phi_mh, Phi_h = inp.Phi_m, inp.Phi_mg, inp.Phi_mh, inp.Phi_h
    Phi_g, Phi_g_Q, Phi_gm, Phi_gh = inp.Phi_g, inp.Phi_g_Q, inp.Phi_gm, inp.Phi_gh

    Sigma_m, Sigma_g, Sigma_g_Q = inp.Sigma_m, inp.Sigma_g, inp.Sigma_g_Q
    Sigma_hm, Sigma_hg, Sigma_h = inp.Sigma_hm, inp.Sigma_hg, inp.Sigma_h

    mu_m, mu_g, mu_g_Q, mu_h = inp.mu_m, inp.mu_g, inp.mu_g_Q, inp.mu_h

    # Pre-compute selectors and constants
    ed_Phi_mh = e_d @ Phi_mh
    const_term4 = 0.5 * float(
        ed_Phi_mh @ (Sigma_hm @ Sigma_hm.T + Sigma_h @ Sigma_h.T) @ ed_Phi_mh.T
    )
    lambda_coef = Sigma_hg.T @ ed_Phi_mh
    rhs_vec = Sigma_hm.T @ (Phi_mh.T @ e_d)
    ed_Phi_mh_Phi_h = ed_Phi_mh @ Phi_h

    # Build diag(D_{m,t}), diag(D_{g,t}) from the Gamma mapping
    z_all = inp.Gamma0[None, :] + h_t @ inp.Gamma1.T
    diag_all = np.exp(0.5 * z_all)
    Dm_vec = diag_all[:, :dm]
    Dg_vec = diag_all[:, dm:]
    Dm_sq = Dm_vec ** 2

    # Dividend variance shortcut σ_{Δd,t}^2(h_t) = 0.5 * (γ_dd0 + γ_dd2' h_t)
    sigma_dd = 0.5 * (inp.gamma_dd0 + h_t @ inp.gamma_dd2)

    # Λ_{g,t} using Σ_g D_{g,t} Λ = (...)
    b_t = (
        (mu_g - mu_g_Q)[None, :]
        + m_t @ Phi_gm.T
        + g_t @ (Phi_g - Phi_g_Q).T
        + h_t @ Phi_gh.T
    )
    u = np.linalg.solve(Sigma_g, b_t.T).T
    lambda_g_t = u / Dg_vec

    # Recursion initial conditions
    log_sum = np.zeros(T)  # log(1)
    current_sum = np.ones(T)

    A_prev = 0.0
    B_prev = np.zeros(dm)
    C_prev = np.zeros(dg)

    for n in range(1, n_max + 1):
        ed_plus_B = e_d + B_prev
        B_n = ed_plus_B @ Phi_m
        C_n = -e1 + ed_plus_B @ Phi_mg + C_prev @ Phi_g_Q

        tmp_row = C_prev @ Sigma_g_Q + ed_Phi_mh @ Sigma_hg
        A_inc = (
            ed_plus_B @ mu_m
            + C_prev @ mu_g_Q
            + ed_Phi_mh @ mu_h
            + 0.5 * float(tmp_row @ tmp_row.T)
            + const_term4
        )
        A_n = A_prev + A_inc

        # D_n(m_t, g_t, h_t)
        ed_plus_B_Phi_mh = ed_plus_B @ Phi_mh
        part1 = h_t @ ed_plus_B_Phi_mh
        part2 = h_t @ ed_Phi_mh_Phi_h
        part3 = -(lambda_g_t @ lambda_coef)
        coef5 = (ed_plus_B @ Sigma_m) * rhs_vec
        part5 = Dm_vec @ coef5
        coef6 = (ed_plus_B @ Sigma_m) ** 2
        part6 = 0.5 * (Dm_sq @ coef6)

        D_n_vec = part1 + part2 + sigma_dd + part3 + part5 + part6

        x_n = A_n + m_t @ B_n.T + g_t @ C_n.T + D_n_vec
        log_sum = np.logaddexp(log_sum, x_n)
        new_terms = np.exp(x_n)
        current_sum += new_terms

        if n >= min_terms:
            threshold = np.maximum(atol, rtol * current_sum.max())
            if new_terms.max() < threshold:
                break

        A_prev, B_prev, C_prev = A_n, B_n, C_n

    return np.exp(log_sum)


def price_equity_series(
    inp: EquityPricingInputs,
    m_t: Array,
    g_t: Array,
    h_t: Array,
    dividend_raw: Array,
    spx_price_obs: Array,
    **kwargs,
) -> Dict[str, Array]:
    """Convenience wrapper returning equity series and decomposition."""

    ratio = price_equity_ratio(inp, m_t, g_t, h_t, **kwargs)
    fundamental = ratio * dividend_raw
    net_fundamental = spx_price_obs - fundamental
    return {
        "ratio": ratio,
        "fundamental": fundamental,
        "net_fundamental": net_fundamental,
        "n_obs": np.arange(m_t.shape[0]),
    }
