from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from common.vol_mapping import diag_vols_series


# --------------------------
# Utilities
# --------------------------
def safe_cholesky(A: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Compute a numerically stable Cholesky factor."""
    A = 0.5 * (A + A.T)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:  # pragma: no cover - rare but essential
        d = A.shape[0]
        return np.linalg.cholesky(A + jitter * np.eye(d, dtype=A.dtype))


def logmvnorm(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log-density of a multivariate normal."""
    L = safe_cholesky(cov, 1e-12)
    r = x - mean
    z = np.linalg.solve(L, r)  # Solve L z = r
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    k = x.shape[-1]
    return -0.5 * (logdet + np.dot(z, z) + k * np.log(2.0 * np.pi))


def logsumexp(v: np.ndarray) -> float:
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


def systematic_resampling(weights: np.ndarray, rng: np.random.Generator, N: int) -> np.ndarray:
    """Systematic resampling; weights are normalized 1D array of length ``N``."""
    positions = (rng.random() + np.arange(N)) / N
    cdf = np.cumsum(weights)
    return np.searchsorted(cdf, positions)


# --------------------------
# Model params container
# --------------------------
@dataclass
class VolParams:
    d_m: int
    d_g: int
    d_h: int

    mu_m: np.ndarray
    mu_g: np.ndarray
    Phi_m: np.ndarray
    Phi_mg: np.ndarray
    Phi_mh: np.ndarray
    Phi_gm: np.ndarray
    Phi_g: np.ndarray
    Phi_gh: np.ndarray

    Sigma_m: np.ndarray
    Sigma_gm: np.ndarray
    Sigma_g: np.ndarray

    Gamma0: np.ndarray
    Gamma1: np.ndarray

    mu_h: np.ndarray
    Phi_h: np.ndarray
    Sigma_h: np.ndarray
    Sigma_hm: np.ndarray
    Sigma_hg: np.ndarray

    h0_mean: np.ndarray
    h0_cov: np.ndarray

    e_div_ix: Optional[int] = None
    dividend_uses_lagged_h: bool = True

    Sh: Optional[np.ndarray] = None
    Sh_chol: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.e_div_ix is None:
            self.e_div_ix = self.d_m - 1
        if not isinstance(self.dividend_uses_lagged_h, bool):  # pragma: no cover
            raise TypeError("dividend_uses_lagged_h must be a boolean")
        self.e_div_ix = int(self.e_div_ix)
        if not (0 <= self.e_div_ix < self.d_m):  # pragma: no cover - sanity guard
            raise ValueError("e_div_ix must index the macro block")

    def precompute(self) -> None:
        """Pre-compute covariance for the volatility innovations."""
        self.Sh = (
            self.Sigma_h @ self.Sigma_h.T
            + self.Sigma_hm @ self.Sigma_hm.T
            + self.Sigma_hg @ self.Sigma_hg.T
        )
        self.Sh = 0.5 * (self.Sh + self.Sh.T)
        self.Sh_chol = safe_cholesky(self.Sh, 1e-12)


# --------------------------
# Log observation density at time t for (m_{t+1}, g_{t+1})
# --------------------------
def log_obs_density_t(
    t: int,
    h_t: np.ndarray,
    m: np.ndarray,
    g: np.ndarray,
    params: VolParams,
    *,
    h_prev: Optional[np.ndarray] = None,
) -> float:
    """Log density ``log p(m_{t+1}, g_{t+1} | m_t, g_t, h_t; θ)``."""
    dm, dg = params.d_m, params.d_g

    h_curr = np.asarray(h_t, dtype=np.float64)
    h_pre_vec: Optional[np.ndarray] = None
    if params.dividend_uses_lagged_h:
        base_prev = h_prev if h_prev is not None else params.mu_h
        h_pre_vec = np.asarray(base_prev, dtype=np.float64)
    Dm_series, Dg_series = diag_vols_series(
        Gamma0=params.Gamma0,
        Gamma1=params.Gamma1,
        H=h_curr[None, :],
        d_m=dm,
        d_g=dg,
        e_div_ix=int(params.e_div_ix),
        use_lag_for_div=params.dividend_uses_lagged_h,
        h_pre=h_pre_vec,
    )
    diag_m = Dm_series[0]
    diag_g = Dg_series[0]

    m_t = m[t]
    g_t = g[t]
    m_tp1 = m[t + 1]
    g_tp1 = g[t + 1]

    mean_m = params.mu_m + params.Phi_m @ m_t + params.Phi_mg @ g_t + params.Phi_mh @ h_curr
    mean_g = params.mu_g + params.Phi_gm @ m_t + params.Phi_g @ g_t + params.Phi_gh @ h_curr

    r_m = m_tp1 - mean_m
    r_g = g_tp1 - mean_g
    r = np.concatenate([r_m, r_g])

    Sig_mm = params.Sigma_m * diag_m[None, :]
    var_m = Sig_mm @ Sig_mm.T

    cov_mg = Sig_mm @ params.Sigma_gm.T

    Sig_gg = params.Sigma_g * diag_g[None, :]
    var_g = params.Sigma_gm @ params.Sigma_gm.T + Sig_gg @ Sig_gg.T

    top = np.concatenate([var_m, cov_mg], axis=1)
    bot = np.concatenate([cov_mg.T, var_g], axis=1)
    S = np.concatenate([top, bot], axis=0)
    S = 0.5 * (S + S.T) + 1e-12 * np.eye(dm + dg)

    zeros = np.zeros(dm + dg, dtype=np.float64)
    return logmvnorm(r, zeros, S)


# --------------------------
# PG-AS main
# --------------------------
@dataclass
class PGASDiagnostics:
    ess: np.ndarray
    loglik: float
    resample_count: int
    accept_path: bool


def draw_h_pgas(
    m: np.ndarray,
    g: np.ndarray,
    params: VolParams,
    h_ref: np.ndarray,
    J: int = 300,
    ess_threshold: float = 0.5,
    seed: Optional[int] = None,
    use_ancestor_sampling: bool = True,
) -> Tuple[np.ndarray, PGASDiagnostics]:
    """Particle Gibbs with ancestor sampling for ``h_{0:T}``."""
    if m.shape[0] != g.shape[0]:
        raise ValueError("m and g must share the same length")
    if h_ref.shape != (m.shape[0], params.d_h):
        raise ValueError("h_ref must have shape (T+1, d_h)")

    params.precompute()
    rng = np.random.default_rng(seed)

    T = m.shape[0] - 1
    if T < 1:
        raise ValueError("Need at least one transition (T >= 1)")
    dh = params.d_h

    H = np.zeros((T + 1, J, dh), dtype=np.float64)
    A = np.full((T + 1, J), -1, dtype=np.int64)
    LW = np.zeros((T, J), dtype=np.float64)
    W = np.zeros((T, J), dtype=np.float64)
    ess = np.zeros(T, dtype=np.float64)

    H[0, 0, :] = h_ref[0]
    L0 = safe_cholesky(params.h0_cov, 1e-12)
    for j in range(1, J):
        z = rng.standard_normal(dh)
        H[0, j, :] = params.h0_mean + L0 @ z
    W_prev = np.full(J, 1.0 / J, dtype=np.float64)

    resample_count = 0
    loglik_sum = 0.0

    Phi_h_T = params.Phi_h.T

    for t in range(T):
        if use_ancestor_sampling:
            means = params.mu_h[None, :] + H[t, :, :] @ Phi_h_T
            log_as = np.empty(J, dtype=np.float64)
            for k in range(J):
                log_as[k] = np.log(W_prev[k] + 1e-300) + logmvnorm(
                    h_ref[t + 1], means[k], params.Sh
                )
            log_as -= logsumexp(log_as)
            as_probs = np.exp(log_as)
            as_probs /= np.sum(as_probs)
            a0 = rng.choice(J, p=as_probs)
        else:
            a0 = rng.choice(J, p=W_prev)

        A[t + 1, 0] = a0
        H[t + 1, 0, :] = h_ref[t + 1]

        ess_prev = 1.0 / np.sum(W_prev ** 2)
        do_resample = bool(ess_prev < ess_threshold * J)

        if do_resample:
            resample_count += 1
            idx = systematic_resampling(W_prev, rng, J)
            for j in range(1, J):
                A[t + 1, j] = idx[j]
        else:
            for j in range(1, J):
                A[t + 1, j] = j

        for j in range(1, J):
            parent = A[t + 1, j]
            mean = params.mu_h + params.Phi_h @ H[t, parent, :]
            z = rng.standard_normal(dh)
            H[t + 1, j, :] = mean + params.Sh_chol @ z

        LW_t = np.empty(J, dtype=np.float64)
        for j in range(J):
            if params.dividend_uses_lagged_h:
                if t == 0:
                    h_prev = params.mu_h
                else:
                    parent = A[t, j]
                    h_prev = params.mu_h if parent < 0 else H[t - 1, parent, :]
            else:
                h_prev = None
            LW_t[j] = log_obs_density_t(t, H[t, j, :], m, g, params, h_prev=h_prev)
        lw = LW_t - logsumexp(LW_t)
        W_t = np.exp(lw)
        W_t /= np.sum(W_t)

        LW[t, :] = lw
        W[t, :] = W_t
        ess[t] = 1.0 / np.sum(W_t ** 2)
        loglik_sum += logsumexp(LW_t)

        W_prev = W_t.copy()

    terminal_index = rng.choice(J, p=W[T - 1, :])
    h_path = np.zeros((T + 1, dh), dtype=np.float64)
    h_path[T, :] = H[T, terminal_index, :]
    k = terminal_index
    for t in range(T, 0, -1):
        k = A[t, k]
        h_path[t - 1, :] = H[t - 1, k, :]

    diagnostics = PGASDiagnostics(
        ess=ess,
        loglik=loglik_sum,
        resample_count=resample_count,
        accept_path=True,
    )
    return h_path, diagnostics


def draw_h_pgas_componentwise(
    m: np.ndarray,
    g: np.ndarray,
    params: VolParams,
    h_ref: np.ndarray,
    J: int = 300,
    ess_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, PGASDiagnostics]]:
    """Convenience wrapper that cycles through volatility dimensions individually."""
    rng = np.random.default_rng(seed)
    T = m.shape[0] - 1
    dh = params.d_h
    h_curr = h_ref.copy()
    diags_per_dim: Dict[str, PGASDiagnostics] = {}
    for i in range(dh):
        h_curr, diag = draw_h_pgas(
            m=m,
            g=g,
            params=params,
            h_ref=h_curr,
            J=J,
            ess_threshold=ess_threshold,
            seed=rng.integers(1 << 31),
            use_ancestor_sampling=True,
        )
        diags_per_dim[f"h{i}"] = diag
    return h_curr, diags_per_dim
