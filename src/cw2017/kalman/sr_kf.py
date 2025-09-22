"""Square-root Kalman filter tailored to the CW2017 sampler."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import lax

Array = jnp.ndarray

_JITTER = 1e-10
_LOG_2PI = jnp.log(2.0 * jnp.pi)


def compute_dg_diag_sq(Gamma0_g: Array, Gamma1_g: Array, h_vec: Array) -> Array:
    exponent = Gamma0_g + h_vec @ Gamma1_g.T
    exponent = jnp.clip(exponent, -20.0, 20.0)
    return jnp.exp(exponent)


def symmetrize_psd(matrix: Array) -> Array:
    return (matrix + matrix.T) * 0.5


def cholesky_psd(matrix: Array) -> Array:
    matrix = symmetrize_psd(matrix)
    dim = matrix.shape[0]
    return jsp.cholesky(matrix + _JITTER * jnp.eye(dim, dtype=matrix.dtype), lower=True)


def _qr_lower(stack: Array) -> Array:
    _, r_mat = jnp.linalg.qr(stack.T, mode="reduced")
    sqrt_factor = r_mat.T
    diag = jnp.diag(sqrt_factor)
    sign = jnp.where(diag < 0.0, -1.0, 1.0)
    return sqrt_factor * sign


def sr_predict(S_prev: Array, F_t: Array, Q_chol_t: Array) -> Array:
    q_aug = Q_chol_t + _JITTER * jnp.eye(Q_chol_t.shape[0], dtype=S_prev.dtype)
    fs = F_t @ S_prev
    stacked = jnp.concatenate([fs, q_aug], axis=1)
    return _qr_lower(stacked)


def sr_update(
    a_pred: Array,
    S_pred: Array,
    H_t: Array,
    R_chol_t: Array,
    innovation: Array,
) -> Tuple[Array, Array, Array, jnp.float64]:
    r_aug = R_chol_t + _JITTER * jnp.eye(R_chol_t.shape[0], dtype=S_pred.dtype)
    HP = H_t @ S_pred
    stacked = jnp.concatenate([HP, r_aug], axis=1)
    S_yy = _qr_lower(stacked)

    solved_innov = jsp.solve_triangular(S_yy, innovation, lower=True)
    quad_form = jnp.dot(solved_innov, solved_innov)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(S_yy), _JITTER)))
    obs_dim = innovation.shape[0]
    loglik_inc = -0.5 * (obs_dim * _LOG_2PI + log_det + quad_form)

    PHt = S_pred @ HP.T
    tmp = jsp.solve_triangular(S_yy, PHt.T, lower=True)
    gain_t = jsp.solve_triangular(S_yy.T, tmp, lower=False)
    kalman_gain = gain_t.T

    posterior_mean = a_pred + kalman_gain @ innovation
    innovation_effect = kalman_gain @ HP
    gain_noise = kalman_gain @ r_aug
    stack_post = jnp.concatenate([S_pred - innovation_effect, gain_noise], axis=1)
    posterior_sqrt = _qr_lower(stack_post)
    return posterior_mean, posterior_sqrt, S_yy, loglik_inc


def sr_kf_loglik(
    params3a: Any,
    fixed: Dict[str, Array],
    y_t: Array,
    m_t: Array,
    h_t: Array,
    measurement_terms: Tuple[Array, Array, Array, Array, Array],
    build_HR: Callable[[Any, Dict[str, Array]], Tuple[Array, Array]],
    build_FQ: Callable[[Any, Dict[str, Array], Array, Array, int], Tuple[Array, Array, Array]],
) -> Tuple[jnp.float64, Dict[str, Array]]:
    """Run the SR-Kalman filter with time-varying process noise."""

    y_t = jnp.asarray(y_t, dtype=jnp.float64)
    m_t = jnp.asarray(m_t, dtype=jnp.float64)
    h_t = jnp.asarray(h_t, dtype=jnp.float64)

    A0, A1, B, M0Q, M1Q = measurement_terms
    A0_vec = jnp.asarray(A0, dtype=jnp.float64).reshape(-1)
    A1_mat = jnp.asarray(A1, dtype=jnp.float64)
    M0_vec = jnp.asarray(M0Q, dtype=jnp.float64).reshape(-1)
    M1_mat = jnp.asarray(M1Q, dtype=jnp.float64)
    mu_g_qu = jnp.asarray(fixed["mu_g^{Q,u}"], dtype=jnp.float64)
    obs_offset = A0_vec + A1_mat @ (M0_vec + M1_mat @ mu_g_qu)
    y_t_tilde = y_t - obs_offset

    H_t, R_chol_t = build_HR(params3a, fixed)

    Ng = params3a.Sigma_g.shape[0]
    Gamma0_g = params3a.Gamma0[-Ng:]
    Gamma1_g = params3a.Gamma1[-Ng:, :]

    h0 = h_t[0]
    diag_sq0 = compute_dg_diag_sq(Gamma0_g, Gamma1_g, h0)
    Sigma_scaled0 = params3a.Sigma_g * diag_sq0[None, :]
    cov0 = Sigma_scaled0 @ params3a.Sigma_g.T
    S0 = cholesky_psd(cov0)
    mean0 = params3a.PhiP_blocks.Phi_gh @ h0

    def step(carry, inputs):
        mean_pred, sqrt_pred = carry
        t_idx, y_curr, m_curr, h_curr = inputs
        F_t, Q_chol_t, a_const = build_FQ(params3a, fixed, m_curr, h_curr, t_idx)
        innovation = y_curr - H_t @ mean_pred
        a_post, S_post, S_yy, ll_inc = sr_update(mean_pred, sqrt_pred, H_t, R_chol_t, innovation)
        mean_next = a_const + F_t @ a_post
        sqrt_next = sr_predict(S_post, F_t, Q_chol_t)

        finite_flags = jnp.array(
            [
                jnp.all(jnp.isfinite(arr))
                for arr in (mean_pred, sqrt_pred, innovation, a_post, S_post, S_yy)
            ],
            dtype=jnp.bool_,
        )
        nan_flag = jnp.logical_not(jnp.all(finite_flags))

        outputs = {
            "filtered_mean": a_post,
            "filtered_sqrt": S_post,
            "innovation": innovation,
            "innovation_sqrt": S_yy,
            "loglik": ll_inc,
            "nan_flag": nan_flag,
        }
        return (mean_next, sqrt_next), outputs

    time_index = jnp.arange(y_t_tilde.shape[0])
    (_, _), outputs = lax.scan(
        step,
        (mean0, S0),
        (time_index, y_t_tilde, m_t, h_t),
    )

    loglik = jnp.sum(outputs["loglik"], dtype=jnp.float64)
    nan_detected = jnp.any(outputs["nan_flag"])
    loglik = jnp.where(nan_detected, -jnp.inf, loglik)

    aux: Dict[str, Array] = {
        "filtered_means": outputs["filtered_mean"],
        "filtered_sqrt_covs": outputs["filtered_sqrt"],
        "innovations": outputs["innovation"],
        "innovation_sqrt": outputs["innovation_sqrt"],
        "nan_detected": nan_detected,
    }
    return loglik, aux


__all__ = [
    "sr_kf_loglik",
    "sr_predict",
    "sr_update",
    "symmetrize_psd",
    "cholesky_psd",
    "compute_dg_diag_sq",
]
