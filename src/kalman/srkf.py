"""Square-root Kalman filtering with exact-row handling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Dict

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from utils.linalg import safe_cholesky

Array = jax.Array

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from .ffbs import Step


def _as_arrays(step: Any) -> tuple[Array, ...]:
    """Return the state-space matrices as float64 JAX arrays."""

    if hasattr(step, "as_arrays"):
        return step.as_arrays()  # type: ignore[return-value]
    return tuple(jnp.asarray(item, dtype=jnp.float64) for item in step)


def _get_step(steps: Callable[[int], Any] | Sequence[Any], t: int) -> Any:
    """Return the ``t``-th step regardless of container type."""

    if callable(steps):  # type: ignore[call-arg]
        return steps(t)
    return steps[t]


def _predict_cov_chol(L_prev: Array, T: Array, R: Array, Q_chol: Array) -> Array:
    """Return the Cholesky factor of the one-step-ahead covariance."""

    A_left = T @ L_prev
    A_right = R @ Q_chol
    pre_array = jnp.concatenate([A_left, A_right], axis=1)
    _, R_factor = jnp.linalg.qr(pre_array.T, mode="reduced")
    return R_factor.T


def _partition_measurement(H: Array, tol: float) -> tuple[Array, Array]:
    """Return boolean masks for exact and noisy measurement rows."""

    diag = jnp.diag(jnp.asarray(H, dtype=jnp.float64))
    exact_mask = jnp.abs(diag) <= tol
    noisy_mask = jnp.logical_not(exact_mask)
    return exact_mask, noisy_mask


def _equality_update(m: Array, L: Array, Z: Array, d: Array, y: Array) -> tuple[Array, Array]:
    """Condition ``(m, L)`` on exact (noise-free) linear observations."""

    if Z.shape[0] == 0:
        return m, L

    resid = jnp.asarray(y, dtype=jnp.float64) - (Z @ m + d)
    U = Z @ L
    S = U @ U.T
    S = 0.5 * (S + S.T)
    S_chol = safe_cholesky(S)

    temp = cho_solve((S_chol, True), U)
    K = L @ temp.T

    m_upd = m + K @ resid

    P = L @ L.T
    I = jnp.eye(P.shape[0], dtype=P.dtype)
    IZ = I - K @ Z
    P_upd = IZ @ P @ IZ.T
    P_upd = 0.5 * (P_upd + P_upd.T)
    L_upd = safe_cholesky(P_upd)
    return m_upd, L_upd


def _noisy_update(
    m: Array,
    L: Array,
    Z: Array,
    d: Array,
    y: Array,
    H: Array,
) -> tuple[Array, Array, Array, Array]:
    """Kalman update for noisy measurement rows."""

    if Z.shape[0] == 0:
        zero = jnp.zeros((0,), dtype=jnp.float64)
        zero_mat = jnp.zeros((0, 0), dtype=jnp.float64)
        return m, L, zero, zero_mat

    resid = jnp.asarray(y, dtype=jnp.float64) - (Z @ m + d)

    U = Z @ L
    S = jnp.asarray(H, dtype=jnp.float64) + U @ U.T
    S = 0.5 * (S + S.T)
    S_chol = safe_cholesky(S)

    temp = cho_solve((S_chol, True), U)
    K = L @ temp.T

    m_upd = m + K @ resid

    P = L @ L.T
    I = jnp.eye(P.shape[0], dtype=P.dtype)
    IZ = I - K @ Z
    P_upd = IZ @ P @ IZ.T + K @ H @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)
    L_upd = safe_cholesky(P_upd)
    return m_upd, L_upd, resid, S_chol


def kf_forward_sr(
    y: Array,
    steps: Callable[[int], Any] | Sequence[Any],
    m0: Array,
    L0: Array,
    *,
    exact_tol: float = 0.0,
) -> Dict[str, Array]:
    """Run the square-root Kalman filter with exact-row handling."""

    y_arr = jnp.asarray(y, dtype=jnp.float64)
    m = jnp.asarray(m0, dtype=jnp.float64)
    L = jnp.asarray(L0, dtype=jnp.float64)

    Tlen = y_arr.shape[0]

    pred_means: list[Array] = []
    pred_chols: list[Array] = []
    filt_means: list[Array] = []
    filt_chols: list[Array] = []
    innovs: list[Array] = []
    innov_chols: list[Array] = []
    loglik_terms: list[Array] = []
    exact_masks: list[Array] = []
    noisy_masks: list[Array] = []

    for t in range(Tlen):
        step = _get_step(steps, t)
        Z, d, H, T_mat, c, R, Q = _as_arrays(step)

        pred_means.append(m)
        pred_chols.append(L)

        exact_mask, noisy_mask = _partition_measurement(H, exact_tol)
        exact_masks.append(exact_mask)
        noisy_masks.append(noisy_mask)

        Z_exact = Z[exact_mask, :]
        d_exact = d[exact_mask]
        y_exact = y_arr[t][exact_mask]

        Z_noisy = Z[noisy_mask, :]
        d_noisy = d[noisy_mask]
        y_noisy = y_arr[t][noisy_mask]
        H_noisy = H[noisy_mask, :][:, noisy_mask]

        m_eq, L_eq = _equality_update(m, L, Z_exact, d_exact, y_exact)
        m_filt, L_filt, innov, S_chol = _noisy_update(
            m_eq, L_eq, Z_noisy, d_noisy, y_noisy, H_noisy
        )

        filt_means.append(m_filt)
        filt_chols.append(L_filt)
        innovs.append(innov)
        innov_chols.append(S_chol)

        if innov.shape[0] == 0:
            loglik_terms.append(jnp.array(0.0, dtype=jnp.float64))
        else:
            alpha = solve_triangular(S_chol, innov, lower=True)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(S_chol)))
            dim = innov.shape[0]
            quad = alpha @ alpha
            term = -0.5 * (logdet + quad + dim * jnp.log(2.0 * jnp.pi))
            loglik_terms.append(term)

        Q_chol = safe_cholesky(Q)
        L = _predict_cov_chol(L_filt, T_mat, R, Q_chol)
        m = T_mat @ m_filt + c

    m_pred = jnp.stack(pred_means)
    L_pred = jnp.stack(pred_chols)
    m_filt = jnp.stack(filt_means)
    L_filt = jnp.stack(filt_chols)
    innov_arr = jnp.stack(innovs)
    S_chol_arr = jnp.stack(innov_chols)
    exact_mask_arr = jnp.stack(exact_masks)
    noisy_mask_arr = jnp.stack(noisy_masks)

    P_pred = jnp.einsum("tij,tkj->tik", L_pred, L_pred)
    P_filt = jnp.einsum("tij,tkj->tik", L_filt, L_filt)
    innov_cov = jnp.einsum("tij,tkj->tik", S_chol_arr, S_chol_arr)

    loglik = jnp.sum(jnp.stack(loglik_terms)) if loglik_terms else jnp.array(0.0, dtype=jnp.float64)

    cache: Dict[str, Array] = {
        "m_pred": m_pred,
        "L_pred": L_pred,
        "m": m_filt,
        "L": L_filt,
        "P_pred": P_pred,
        "P": P_filt,
        "innov": innov_arr,
        "S_chol": S_chol_arr,
        "innov_cov": innov_cov,
        "loglik": loglik,
        "exact_mask": exact_mask_arr,
        "noisy_mask": noisy_mask_arr,
    }

    cache["pred_mean"] = m_pred
    cache["pred_cov"] = P_pred
    cache["filt_mean"] = m_filt
    cache["filt_cov"] = P_filt

    return cache


__all__ = ["kf_forward_sr"]

