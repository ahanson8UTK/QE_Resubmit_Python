"""Square-root Kalman filter tailored to the CW2017 sampler."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import lax

from .design import build_block3a_design
from .utils import cholesky_psd, symmetrize_psd

Array = jnp.ndarray

_JITTER = 1e-10
_LOG_2PI = jnp.log(2.0 * jnp.pi)


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
    build_HR: Callable[[Any, Dict[str, Array]], Tuple[Array, Array]] | None,
    build_FQ: Callable[[Any, Dict[str, Array], Array, Array, int], Tuple[Array, Array, Array]] | None,
) -> Tuple[jnp.float64, Dict[str, Array]]:
    """Run the SR-Kalman filter with time-varying process noise."""

    del build_HR, build_FQ  # The design fully specifies the LDS inputs.

    design = build_block3a_design(params3a, fixed, measurement_terms, y_t, m_t, h_t, ())

    observations = jnp.asarray(design.observations, dtype=jnp.float64)
    H_t = jnp.asarray(design.measurement_matrix, dtype=jnp.float64)
    R_chol_t = jnp.asarray(design.measurement_noise_chol, dtype=jnp.float64)
    transition = jnp.asarray(design.transition_matrix, dtype=jnp.float64)
    offsets = jnp.asarray(design.state_offsets, dtype=jnp.float64)
    Q_chols = jnp.asarray(design.process_noise_chol, dtype=jnp.float64)
    mean0 = jnp.asarray(design.initial_mean, dtype=jnp.float64)
    S0 = jnp.asarray(design.initial_sqrt_cov, dtype=jnp.float64)

    def step(carry, inputs):
        mean_pred, sqrt_pred = carry
        y_curr, offset_curr, Q_chol_curr = inputs
        innovation = y_curr - H_t @ mean_pred
        a_post, S_post, S_yy, ll_inc = sr_update(mean_pred, sqrt_pred, H_t, R_chol_t, innovation)
        mean_next = offset_curr + transition @ a_post
        sqrt_next = sr_predict(S_post, transition, Q_chol_curr)

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

    (_, _), outputs = lax.scan(
        step,
        (mean0, S0),
        (observations, offsets, Q_chols),
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
        "design": design,
    }
    return loglik, aux


__all__ = [
    "sr_kf_loglik",
    "sr_predict",
    "sr_update",
    "symmetrize_psd",
    "cholesky_psd",
]
