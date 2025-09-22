"""Tests for covariance-form Kalman filter and DK smoother."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from kalman import Step, dk_sample, kf_forward

jax.config.update("jax_enable_x64", True)


def _build_time_varying_model():
    rng = np.random.default_rng(0)
    T = 8
    nx, ny = 3, 2

    steps: list[Step] = []
    x0_mean = np.zeros(nx)
    x0_cov = 0.25 * np.eye(nx)
    x = np.zeros((T + 1, nx))
    x[0] = x0_mean + np.linalg.cholesky(x0_cov) @ rng.normal(size=nx)

    y = np.zeros((T, ny))

    for t in range(T):
        Z = 0.3 * rng.normal(size=(ny, nx))
        d = 0.1 * rng.normal(size=ny)

        H_raw = rng.normal(size=(ny, ny))
        H = H_raw @ H_raw.T
        H += np.diag(np.array([1e-10, 0.2]))

        T_mat = np.eye(nx) + 0.1 * rng.normal(size=(nx, nx))
        c = 0.1 * rng.normal(size=nx)
        R = np.eye(nx)

        Q_raw = rng.normal(size=(nx, nx))
        Q = Q_raw @ Q_raw.T + 0.1 * np.eye(nx)

        steps.append(Step(Z=Z, d=d, H=H, T=T_mat, c=c, R=R, Q=Q))

        eta = np.linalg.cholesky(H) @ rng.normal(size=ny)
        y[t] = Z @ x[t] + d + eta

        eps = np.linalg.cholesky(Q) @ rng.normal(size=nx)
        x[t + 1] = T_mat @ x[t] + c + R @ eps

    return y, steps, x0_mean, x0_cov


def _numpy_reference_loglik(y: np.ndarray, steps: list[Step], x0_mean: np.ndarray, x0_cov: np.ndarray) -> float:
    m = x0_mean.copy()
    P = x0_cov.copy()
    loglik = 0.0
    ny = y.shape[1]

    for t, step in enumerate(steps):
        Z = np.asarray(step.Z, dtype=np.float64)
        d = np.asarray(step.d, dtype=np.float64)
        H = np.asarray(step.H, dtype=np.float64)
        T_mat = np.asarray(step.T, dtype=np.float64)
        c = np.asarray(step.c, dtype=np.float64)
        R = np.asarray(step.R, dtype=np.float64)
        Q = np.asarray(step.Q, dtype=np.float64)

        y_hat = Z @ m + d
        v = y[t] - y_hat

        S = Z @ P @ Z.T + H
        L = np.linalg.cholesky(S)

        ZP = Z @ P
        w = np.linalg.solve(L, ZP)
        S_inv_ZP = np.linalg.solve(L.T, w)
        K = S_inv_ZP.T

        m = m + K @ v
        P = P - K @ Z @ P
        P = 0.5 * (P + P.T)

        alpha = np.linalg.solve(L, v)
        loglik += -0.5 * (ny * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)

        m = T_mat @ m + c
        P = T_mat @ P @ T_mat.T + R @ Q @ R.T
        P = 0.5 * (P + P.T)

    return float(loglik)


def _loglik_from_cache(cache: dict[str, jax.Array]) -> float:
    innov = np.asarray(cache["innov"], dtype=np.float64)
    innov_cov = np.asarray(cache["innov_cov"], dtype=np.float64)
    ny = innov.shape[1]

    loglik = 0.0
    for v, S in zip(innov, innov_cov):
        L = np.linalg.cholesky(S)
        alpha = np.linalg.solve(L, v)
        loglik += -0.5 * (ny * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)
    return float(loglik)


def test_covariance_kf_matches_numpy():
    y, steps, x0_mean, x0_cov = _build_time_varying_model()

    cache = kf_forward(y, steps, x0_mean, x0_cov)
    loglik_ref = _numpy_reference_loglik(y, steps, x0_mean, x0_cov)
    loglik = _loglik_from_cache(cache)

    np.testing.assert_allclose(loglik, loglik_ref, rtol=1e-6, atol=1e-6)


def test_dk_sample_shape_and_finite():
    y, steps, x0_mean, x0_cov = _build_time_varying_model()
    cache = kf_forward(y, steps, x0_mean, x0_cov)

    key = jax.random.PRNGKey(42)
    draw = dk_sample(key, y, steps, x0_mean, x0_cov, cache)

    assert draw.shape == (y.shape[0], x0_mean.shape[0])
    assert jnp.all(jnp.isfinite(draw))
