"""Recursive covariance block draws for the Gibbs sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.random import Generator
from scipy import linalg

from .equity_constraint import equity_margin
from .math_utils import (
    ensure_spd_inverse,
    iw_sample,
    matrix_normal_sample,
    safe_cholesky,
    triangular_solve_unit_lower,
)


@dataclass
class CovarianceDraw:
    Sigma_m: np.ndarray
    eps_m: np.ndarray
    Sigma_gm: np.ndarray
    eps_g: np.ndarray
    Sigma_hm: np.ndarray
    Sigma_hg: np.ndarray
    Sigma_h: np.ndarray
    constraint_rejects: int
    constraint_used_fallback: bool


@dataclass
class TriangularRegressionPrior:
    row_covariances: list[np.ndarray]


@dataclass
class MatrixNormalInverseWishartPrior:
    M0: np.ndarray
    V0: np.ndarray
    S0: np.ndarray
    nu0: float


@dataclass
class CovarianceBlockPriors:
    rng: Generator
    sigma_m: TriangularRegressionPrior
    sigma_gm: TriangularRegressionPrior
    sigma_h: MatrixNormalInverseWishartPrior


@dataclass
class EquityConstraintParams:
    theta_m: np.ndarray
    theta_g: np.ndarray
    theta_g_Q: np.ndarray
    mu_m_bar: np.ndarray
    mu_g_u_bar: np.ndarray
    mu_g_Q_u_bar: np.ndarray
    mu_h_bar: np.ndarray
    Phi_m: np.ndarray
    Phi_mg: np.ndarray
    Phi_mh: np.ndarray
    Phi_h: np.ndarray
    M0_Q: np.ndarray
    M1_Q: np.ndarray
    Sigma_g_Q: np.ndarray
    Phi_g_Q: np.ndarray


def _check_triangular_prior(prior: TriangularRegressionPrior, rows: int) -> None:
    if len(prior.row_covariances) < rows:
        raise ValueError("insufficient prior rows provided")


def draw_sigma_m(
    r_m: np.ndarray,
    Dm: np.ndarray,
    priors: CovarianceBlockPriors,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Model residuals (Appendix B.1):
    Let

    rtm ≡mt+1−μm−Φmmt−Φmggt−Φmhht(B.3)
    rtg ≡gt+1−μg−Φgmmt−Φggt−Φghht(B.4)
    rth ≡ht+1−μh−Φhht(B.5)

    with shocks εm,εg,εh∼iidN(0,I) and

    rtm =Σm diag(Dm,t) εm,t,
    rtg =Σgmεm,t+Σg diag(Dg,t) εg,t,
    rth =Σhmεm,t+Σhgεg,t+Σhεh,t.

    Parameterize the precision transform Tm≡Σm−1, which is also unit lower-triangular. Then

    εm,t=Tmutm,soui,tm=−βi⊤u1:i−1,tm+εm,i,t,  εm,i,t∼N(0,1),

    where βi≡(Tm)i,1:i−1 and the diagonal of Tm is 1.
    """

    rng = priors.rng
    T, d_m = r_m.shape
    _check_triangular_prior(priors.sigma_m, d_m)

    if Dm.shape != r_m.shape:
        raise ValueError("Dm must match r_m in shape")

    u = np.array(r_m / Dm, dtype=np.float64)
    T_m = np.zeros((d_m, d_m), dtype=np.float64)
    np.fill_diagonal(T_m, 1.0)

    for i in range(1, d_m - 1):
        cols = i
        if cols == 0:
            continue
        X = -u[:, :cols]
        y = u[:, i]
        V0 = np.asarray(priors.sigma_m.row_covariances[i], dtype=np.float64)
        if V0.shape != (cols, cols):
            raise ValueError("prior covariance has incorrect shape for row")
        V0_inv = ensure_spd_inverse(V0)
        XtX = X.T @ X
        precision = V0_inv + XtX
        chol_precision = safe_cholesky(precision)
        Vn = linalg.cho_solve((chol_precision, True), np.eye(cols))
        mean = Vn @ (X.T @ y)
        beta = rng.multivariate_normal(mean=mean, cov=Vn)
        T_m[i, :cols] = beta

    Sigma_m = triangular_solve_unit_lower(T_m, np.eye(d_m))
    eps_m = u @ T_m.T
    return Sigma_m, eps_m


def draw_sigma_gm_and_eps_g(
    r_g: np.ndarray,
    Dg: np.ndarray,
    Sigma_g: np.ndarray,
    eps_m: np.ndarray,
    priors: CovarianceBlockPriors,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Goal: draw Σgm honoring the recursion based on (B.4) and then compute εg.

    From (B.4),

    rtg=Σgmεm,t+Σg diag(Dg,t) εg,t.

    Left-multiply by Σg−1 and divide rowwise by Dg,t:

    zi,t≡ei⊤Σg−1rtgDg,i,t=(1Dg,i,tεm,t⊤)bi  +  εg,i,t,bi⊤≡ei⊤Σg−1Σgm.

    For each i=1,…,dg, run a Gaussian regression of zi,⋅ on row-scaled εm to draw bi.
    Finally compute εg,i,t=zi,t−Xi,tbi for all i,t.
    """

    rng = priors.rng
    T, d_g = r_g.shape
    d_m = eps_m.shape[1]
    _check_triangular_prior(priors.sigma_gm, d_g)

    if Dg.shape != r_g.shape:
        raise ValueError("Dg must match r_g in shape")
    if Sigma_g.shape[0] != d_g or Sigma_g.shape[1] != d_g:
        raise ValueError("Sigma_g must be square with dimension d_g")

    z = np.empty_like(r_g, dtype=np.float64)
    for t in range(T):
        vec = linalg.solve_triangular(
            Sigma_g,
            r_g[t],
            lower=True,
            unit_diagonal=True,
            check_finite=False,
        )
        z[t] = vec / Dg[t]

    B = np.zeros((d_g, d_m), dtype=np.float64)
    eps_g = np.zeros((T, d_g), dtype=np.float64)

    for i in range(d_g):
        X = eps_m / Dg[:, i][:, None]
        y = z[:, i]
        V0 = np.asarray(priors.sigma_gm.row_covariances[i], dtype=np.float64)
        if V0.shape != (d_m, d_m):
            raise ValueError("prior covariance has incorrect shape for Sigma_gm row")
        V0_inv = ensure_spd_inverse(V0)
        XtX = X.T @ X
        precision = V0_inv + XtX
        chol_precision = safe_cholesky(precision)
        Vn = linalg.cho_solve((chol_precision, True), np.eye(d_m))
        mean = Vn @ (X.T @ y)
        b = rng.multivariate_normal(mean=mean, cov=Vn)
        B[i] = b
        eps_g[:, i] = y - X @ b

    Sigma_gm = Sigma_g @ B
    return Sigma_gm, eps_g


def draw_sigma_h_block_with_constraint(
    r_h: np.ndarray,
    eps_m: np.ndarray,
    eps_g: np.ndarray,
    equity_params: EquityConstraintParams,
    priors: CovarianceBlockPriors,
    max_rejects: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Goal: jointly draw (Σhm,Σhg,Σh) from the conjugate multivariate regression implied by (B.5),
    truncated to the set where your equity inequality holds.

    Let Xt≡[εm,t⊤  εg,t⊤]∈Rdm+dg, Yt≡rth∈Rdh. Stack into X∈RT×(dm+dg), Y∈RT×dh. The regression is

    Y=XB⊤+E,Et⋅∼N(0, ΣhΣh⊤),

    with B=[Σhm Σhg]∈Rdh×(dm+dg). Use matrix-normal–inverse-Wishart prior

    B∣Σh∼MN(M0, ΣhΣh⊤, V0),ΣhΣh⊤∼IW(ν0,S0).

    Posterior (closed form):

    Vn =(V0−1+X⊤X)−1,Mn=(Y⊤X) Vn,
    Sn =S0+Y⊤Y+M0⊤V0−1M0−Mn⊤Vn−1Mn,
    νn =ν0+T.

    Draw ΣhΣh⊤∼IW(νn,Sn), set Σh=chol(ΣhΣh⊤), then draw B∼MN(Mn, ΣhΣh⊤, Vn) and split B→(Σhm,Σhg).

    Hard equity constraint (truncate): reject draws where the inequality margin is non-negative.
    """

    rng = priors.rng
    T, d_h = r_h.shape
    d_m = eps_m.shape[1]
    d_g = eps_g.shape[1]

    X = np.hstack([eps_m, eps_g])
    Y = np.array(r_h, dtype=np.float64)

    M0 = np.asarray(priors.sigma_h.M0, dtype=np.float64)
    V0 = np.asarray(priors.sigma_h.V0, dtype=np.float64)
    S0 = np.asarray(priors.sigma_h.S0, dtype=np.float64)
    nu0 = float(priors.sigma_h.nu0)

    if M0.shape != (d_h, d_m + d_g):
        raise ValueError("M0 has incorrect shape")
    if V0.shape != (d_m + d_g, d_m + d_g):
        raise ValueError("V0 has incorrect shape")
    if S0.shape != (d_h, d_h):
        raise ValueError("S0 has incorrect shape")

    V0_inv = ensure_spd_inverse(V0)
    XtX = X.T @ X
    precision = V0_inv + XtX
    chol_precision = safe_cholesky(precision)
    Vn = linalg.cho_solve((chol_precision, True), np.eye(d_m + d_g))
    Mn = (Y.T @ X) @ Vn
    prior_term = M0 @ (V0_inv @ M0.T)
    post_term = Mn @ (precision @ Mn.T)
    Sn = S0 + Y.T @ Y + prior_term - post_term
    Sn = (Sn + Sn.T) / 2.0
    nu_n = nu0 + T

    rejects = 0
    used_fallback = False

    while True:
        Sigma_h_cov = iw_sample(rng, nu_n, Sn)
        Sigma_h = safe_cholesky(Sigma_h_cov)
        B = matrix_normal_sample(rng, Mn, Sigma_h_cov, Vn)
        Sigma_hm = B[:, :d_m]
        Sigma_hg = B[:, d_m:]
        margin = equity_margin(
            equity_params.theta_m,
            equity_params.theta_g,
            equity_params.theta_g_Q,
            equity_params.mu_m_bar,
            equity_params.mu_g_u_bar,
            equity_params.mu_g_Q_u_bar,
            equity_params.Phi_m,
            equity_params.Phi_mg,
            equity_params.Phi_mh,
            equity_params.Phi_h,
            equity_params.M0_Q,
            equity_params.M1_Q,
            equity_params.Sigma_g_Q,
            Sigma_hm,
            Sigma_hg,
            Sigma_h,
            equity_params.mu_h_bar,
            equity_params.Phi_g_Q,
        )
        if margin < 0.0:
            break
        rejects += 1
        if rejects > max_rejects:
            raise RuntimeError("Equity constraint too tight; rejection limit reached")

    diagnostics = {
        "constraint_rejects": rejects,
        "constraint_used_fallback": used_fallback,
    }
    return Sigma_hm, Sigma_hg, Sigma_h, diagnostics
