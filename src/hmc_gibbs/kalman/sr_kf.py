"""Square-root Kalman filter implementation used across the sampler.

The filter follows a numerically robust formulation that operates on
Cholesky factors instead of full covariance matrices. This avoids forming
matrix inverses explicitly and keeps the updates stable even when the system
is ill conditioned.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import lax

from ..typing import Array

_JITTER = 1e-10
_LOG_2PI = jnp.log(2.0 * jnp.pi)


def symmetrize_psd(matrix: Array) -> Array:
    """Symmetrise ``matrix`` ensuring the result remains PSD."""

    return (matrix + matrix.T) * 0.5


def _qr_lower(stack: Array) -> Array:
    """Return a lower-triangular Cholesky factor via QR decomposition."""

    _, r_mat = jnp.linalg.qr(stack.T, mode="reduced")
    sqrt_factor = r_mat.T
    diag = jnp.diag(sqrt_factor)
    sign = jnp.where(diag < 0.0, -1.0, 1.0)
    return sqrt_factor * sign


def sr_predict(S_prev: Array, F_t: Array, Q_chol_t: Array) -> Array:
    """Predictive square-root covariance update."""

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
    """Measurement update returning posterior mean/covariance and log-lik increment."""

    r_aug = R_chol_t + _JITTER * jnp.eye(R_chol_t.shape[0], dtype=S_pred.dtype)
    HP = H_t @ S_pred
    stacked = jnp.concatenate([HP, r_aug], axis=1)
    S_yy = _qr_lower(stacked)

    solved_innov = jsp.solve_triangular(S_yy, innovation, lower=True)
    quad_form = jnp.dot(solved_innov, solved_innov)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(S_yy), min=_JITTER)))
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


def _get_field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container[name]
    return getattr(container, name)


def _get_builder(params: Any, name: str) -> Any:
    fns = _get_field(params, "fns")
    if isinstance(fns, Mapping):
        return fns[name]
    return getattr(fns, name)


def sr_kf_loglik(
    params: Any,
    h_t: Array,
    y_t: Array,
    m_t: Array,
) -> Tuple[jnp.float64, Dict[str, Array]]:
    """Run a numerically stable SR-Kalman filter.

    Parameters
    ----------
    params:
        Container with ``a0``, ``S0`` and callable builders ``build_HR`` and ``build_FQ``.
    h_t, y_t, m_t:
        Time series inputs with leading dimension ``T``.
    """

    y_t = jnp.asarray(y_t, dtype=jnp.float64)
    h_t = jnp.asarray(h_t, dtype=jnp.float64)
    m_t = jnp.asarray(m_t, dtype=jnp.float64)

    a_prev = jnp.asarray(_get_field(params, "a0"), dtype=jnp.float64)
    S_prev = jnp.asarray(_get_field(params, "S0"), dtype=jnp.float64)

    build_HR = _get_builder(params, "build_HR")
    build_FQ = _get_builder(params, "build_FQ")

    time_index = jnp.arange(y_t.shape[0])

    def step(carry, inputs):
        mean_prev, sqrt_prev = carry
        t_idx, h_curr, y_curr, m_curr = inputs
        H_t, R_chol_t = build_HR(t_idx, params, h_curr, m_curr)
        F_t, Q_chol_t = build_FQ(t_idx, params, h_curr, m_curr)

        H_t = jnp.asarray(H_t, dtype=jnp.float64)
        R_chol_t = jnp.asarray(R_chol_t, dtype=jnp.float64)
        F_t = jnp.asarray(F_t, dtype=jnp.float64)
        Q_chol_t = jnp.asarray(Q_chol_t, dtype=jnp.float64)

        a_pred = F_t @ mean_prev
        S_pred = sr_predict(sqrt_prev, F_t, Q_chol_t)
        innovation = y_curr - H_t @ a_pred
        a_post, S_post, S_yy, ll_inc = sr_update(a_pred, S_pred, H_t, R_chol_t, innovation)

        finite_flags = jnp.array(
            [
                jnp.all(jnp.isfinite(arr))
                for arr in (a_pred, S_pred, innovation, a_post, S_post, S_yy)
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
        return (a_post, S_post), outputs

    (_, _), outputs = lax.scan(
        step,
        (a_prev, S_prev),
        (time_index, h_t, y_t, m_t),
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
    "symmetrize_psd",
    "sr_predict",
    "sr_update",
]
