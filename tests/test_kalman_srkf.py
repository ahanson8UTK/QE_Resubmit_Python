"""Tests for the square-root Kalman filter."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from kalman import Step, kf_forward
from kalman.ffbs import kf_forward_sr_wrapper

jax.config.update("jax_enable_x64", True)


def _simulate_model(seed: int = 0) -> tuple[np.ndarray, list[Step], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    T = 6
    nx, ny = 3, 2

    steps: list[Step] = []
    x0_mean = np.zeros(nx)
    x0_cov = 0.25 * np.eye(nx)
    x = np.zeros((T + 1, nx))
    x[0] = x0_mean + np.linalg.cholesky(x0_cov) @ rng.normal(size=nx)

    y = np.zeros((T, ny))

    for t in range(T):
        Z = 0.3 * rng.normal(size=(ny, nx))
        d = 0.05 * rng.normal(size=ny)

        H_raw = rng.normal(size=(ny, ny))
        H = H_raw @ H_raw.T + 0.1 * np.eye(ny)

        T_mat = np.eye(nx) + 0.1 * rng.normal(size=(nx, nx))
        c = 0.05 * rng.normal(size=nx)
        R = np.eye(nx)

        Q_raw = rng.normal(size=(nx, nx))
        Q = Q_raw @ Q_raw.T + 0.1 * np.eye(nx)

        steps.append(Step(Z=Z, d=d, H=H, T=T_mat, c=c, R=R, Q=Q))

        eta = np.linalg.cholesky(H) @ rng.normal(size=ny)
        y[t] = Z @ x[t] + d + eta

        eps = np.linalg.cholesky(Q) @ rng.normal(size=nx)
        x[t + 1] = T_mat @ x[t] + c + R @ eps

    return y, steps, x0_mean, x0_cov


def test_srkf_matches_covariance_without_exact_rows():
    y, steps, x0_mean, x0_cov = _simulate_model(seed=1)

    cache_cov = kf_forward(y, steps, x0_mean, x0_cov)
    cache_sr = kf_forward_sr_wrapper(y, steps, x0_mean, x0_cov, exact_tol=0.0)

    np.testing.assert_allclose(cache_sr["filt_mean"], cache_cov["filt_mean"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cache_sr["filt_cov"], cache_cov["filt_cov"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cache_sr["innov"], cache_cov["innov"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cache_sr["innov_cov"], cache_cov["innov_cov"], rtol=1e-6, atol=1e-6)


def test_srkf_handles_exact_rows():
    T = 5
    nx, ny = 3, 3

    Z = jnp.array(
        [
            [1.0, 0.2, -0.1],
            [0.0, 0.8, 0.3],
            [-0.2, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    d = jnp.zeros(ny, dtype=jnp.float64)
    H = jnp.diag(jnp.array([0.0, 0.05, 0.0], dtype=jnp.float64))
    T_mat = 0.9 * jnp.eye(nx, dtype=jnp.float64)
    c = jnp.zeros(nx, dtype=jnp.float64)
    R = jnp.eye(nx, dtype=jnp.float64)
    Q = 0.1 * jnp.eye(nx, dtype=jnp.float64)

    steps = [
        Step(Z=Z, d=d, H=H, T=T_mat, c=c, R=R, Q=Q) for _ in range(T)
    ]

    y_exact = jnp.array([0.75, -0.2], dtype=jnp.float64)
    y_noisy = jnp.array(0.1, dtype=jnp.float64)
    y = jnp.tile(jnp.array([y_exact[0], y_noisy, y_exact[1]], dtype=jnp.float64), (T, 1))

    x0_mean = jnp.zeros(nx, dtype=jnp.float64)
    x0_cov = 0.25 * jnp.eye(nx, dtype=jnp.float64)

    cache = kf_forward_sr_wrapper(y, steps, x0_mean, x0_cov, exact_tol=0.0)

    assert cache["innov"].shape == (T, 1)
    assert cache["S_chol"].shape == (T, 1, 1)
    assert jnp.isfinite(cache["loglik"]).item()

    filt_mean = cache["filt_mean"]
    exact_mask = jnp.isclose(jnp.diag(H), 0.0)
    Z_exact = Z[exact_mask, :]
    d_exact = d[exact_mask]
    y_exact_full = y[:, exact_mask]

    recon = jnp.einsum("ij,tj->ti", Z_exact, filt_mean) + d_exact
    np.testing.assert_allclose(recon, np.asarray(y_exact_full), rtol=1e-6, atol=1e-6)
