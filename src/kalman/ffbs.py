"""Kalman filtering and simulation smoothing utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from utils.linalg import chol_solve_spd, safe_cholesky

ArrayLike = Any


@dataclass(frozen=True)
class Step:
    """Container for time-varying state-space matrices."""

    Z: ArrayLike
    d: ArrayLike
    H: ArrayLike
    T: ArrayLike
    c: ArrayLike
    R: ArrayLike
    Q: ArrayLike

    def as_arrays(self) -> tuple[jax.Array, ...]:
        """Return the stored matrices as ``float64`` JAX arrays."""

        return tuple(jnp.asarray(item, dtype=jnp.float64) for item in (self.Z, self.d, self.H, self.T, self.c, self.R, self.Q))


def _get_step(steps: Callable[[int], Step] | Sequence[Step], t: int) -> Step:
    """Return the ``t``-th :class:`Step` regardless of container type."""

    if callable(steps):  # type: ignore[call-arg]
        return steps(t)
    return steps[t]


def kf_forward(
    y: ArrayLike,
    steps: Callable[[int], Step] | Sequence[Step],
    x0_mean: ArrayLike,
    x0_cov: ArrayLike,
) -> dict[str, jax.Array]:
    """Run the covariance-form Kalman filter for a time-varying LDS.

    Parameters
    ----------
    y:
        Observations with shape ``(T, n_y)``.
    steps:
        Sequence or callable that yields the matrices ``(Z_t, d_t, H_t, T_t, c_t, R_t, Q_t)``
        for time ``t``. The model follows Form I of Appendix B.1.1 in Creal and Wu
        (2017, *International Economic Review*), Eqs. (B.1)–(B.4) on pp. 26–27.
    x0_mean, x0_cov:
        Mean and covariance of the initial state ``x_0``.

    Returns
    -------
    dict of str -> Array
        Cache containing filtered and one-step-ahead moments and innovation statistics.
    """

    y_arr = jnp.asarray(y, dtype=jnp.float64)
    m = jnp.asarray(x0_mean, dtype=jnp.float64)
    P = jnp.asarray(x0_cov, dtype=jnp.float64)

    Tlen = y_arr.shape[0]

    pred_means = []
    pred_covs = []
    filt_means = []
    filt_covs = []
    innovs = []
    innov_covs = []

    for t in range(Tlen):
        step = _get_step(steps, t)
        Z, d, H, T_mat, c, R, Q = step.as_arrays()

        pred_means.append(m)
        pred_covs.append(P)

        y_hat = Z @ m + d
        v = y_arr[t] - y_hat
        S = Z @ P @ Z.T + H
        L = safe_cholesky(S)
        K = chol_solve_spd(L, Z @ P).T

        m_post = m + K @ v
        P_post = P - K @ Z @ P
        P_post = 0.5 * (P_post + P_post.T)

        filt_means.append(m_post)
        filt_covs.append(P_post)
        innovs.append(v)
        innov_covs.append(S)

        m = T_mat @ m_post + c
        P = T_mat @ P_post @ T_mat.T + R @ Q @ R.T
        P = 0.5 * (P + P.T)

    return {
        "pred_mean": jnp.stack(pred_means),
        "pred_cov": jnp.stack(pred_covs),
        "filt_mean": jnp.stack(filt_means),
        "filt_cov": jnp.stack(filt_covs),
        "innov": jnp.stack(innovs),
        "innov_cov": jnp.stack(innov_covs),
    }


def dk_sample(
    key: jax.random.KeyArray,
    y: ArrayLike,
    steps: Callable[[int], Step] | Sequence[Step],
    x0_mean: ArrayLike,
    x0_cov: ArrayLike,
    cache: dict[str, jax.Array],
) -> jax.Array:
    """Draw a state trajectory from the smoothing distribution.

    Implements the Durbin–Koopman simulation smoother (Form I) using the
    covariance-form Kalman filter moments derived in Appendix B.1.1 of Creal and
    Wu (2017, *International Economic Review*). The recursion matches Eqs.
    (B.5)–(B.8) on pp. 26–27.
    """

    del y, x0_mean, x0_cov

    filt_means = cache["filt_mean"]
    filt_covs = cache["filt_cov"]
    Tlen = filt_means.shape[0]
    n_state = filt_means.shape[1]

    samples = jnp.zeros((Tlen, n_state), dtype=jnp.float64)
    key_t = key

    key_t, key_draw = jax.random.split(key_t)
    L_T = safe_cholesky(filt_covs[-1])
    eps_T = jax.random.normal(key_draw, (n_state,), dtype=jnp.float64)
    samples = samples.at[-1].set(filt_means[-1] + L_T @ eps_T)

    for t in range(Tlen - 2, -1, -1):
        step = _get_step(steps, t)
        _, _, _, T_mat, c, R, Q = step.as_arrays()

        P_t = filt_covs[t]
        m_t = filt_means[t]

        P_pred = T_mat @ P_t @ T_mat.T + R @ Q @ R.T
        P_pred = 0.5 * (P_pred + P_pred.T)
        L_pred = safe_cholesky(P_pred)
        J = chol_solve_spd(L_pred, T_mat @ P_t).T

        m_pred = T_mat @ m_t + c
        mean_cond = m_t + J @ (samples[t + 1] - m_pred)
        Cov_cond = P_t - J @ P_pred @ J.T
        Cov_cond = 0.5 * (Cov_cond + Cov_cond.T)

        key_t, key_draw = jax.random.split(key_t)
        eps = jax.random.normal(key_draw, (n_state,), dtype=jnp.float64)
        L_cov = safe_cholesky(Cov_cond)
        samples = samples.at[t].set(mean_cond + L_cov @ eps)

    return samples
