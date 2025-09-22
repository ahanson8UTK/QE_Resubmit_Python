"""Linear algebra utilities for covariance block sampling."""

from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator
from scipy import linalg


def safe_cholesky(matrix: np.ndarray, *, jitter: float = 1e-12, max_attempts: int = 6) -> np.ndarray:
    """Return the lower Cholesky factor using adaptive jitter."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    attempt = 0
    current = np.array(matrix, dtype=np.float64, copy=True)
    jitter_scale = 1.0
    while attempt < max_attempts:
        try:
            return np.linalg.cholesky(current)
        except np.linalg.LinAlgError:
            diag_indices = np.diag_indices_from(current)
            current[diag_indices] += jitter * jitter_scale
            jitter_scale *= 10.0
            attempt += 1
    raise np.linalg.LinAlgError("Cholesky decomposition failed even after jitter")


def triangular_solve_unit_lower(tril: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``tril @ x = rhs`` for unit-lower-triangular ``tril``."""

    if tril.ndim != 2 or tril.shape[0] != tril.shape[1]:
        raise ValueError("tril must be square")
    return linalg.solve_triangular(tril, rhs, lower=True, unit_diagonal=True, check_finite=False)


def _bartlett_factor(rng: Generator, df: float, dim: int) -> np.ndarray:
    if df <= dim - 1:
        raise ValueError("degrees of freedom must exceed dim - 1")
    a = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        a[i, i] = math.sqrt(rng.chisquare(df - i))
        for j in range(i):
            a[i, j] = rng.normal()
    return a


def iw_sample(rng: Generator, df: float, scale: np.ndarray) -> np.ndarray:
    """Draw from ``IW(df, scale)`` using Bartlett decomposition."""

    if scale.ndim != 2 or scale.shape[0] != scale.shape[1]:
        raise ValueError("scale must be square")
    dim = scale.shape[0]
    scale = np.array(scale, dtype=np.float64, copy=True)
    chol_scale = safe_cholesky(scale)
    inv_chol = linalg.solve_triangular(chol_scale, np.eye(dim), lower=True, check_finite=False)
    bartlett = _bartlett_factor(rng, df, dim)
    mat = inv_chol @ bartlett
    wishart_sample = mat @ mat.T
    chol_wishart = safe_cholesky(wishart_sample)
    inv_sample = linalg.cho_solve((chol_wishart, True), np.eye(dim))
    return (inv_sample + inv_sample.T) / 2.0


def matrix_normal_sample(
    rng: Generator,
    mean: np.ndarray,
    row_cov: np.ndarray,
    col_cov: np.ndarray,
) -> np.ndarray:
    """Sample from ``MN(mean, row_cov, col_cov)``."""

    mean = np.array(mean, dtype=np.float64, copy=False)
    row_cov = np.array(row_cov, dtype=np.float64, copy=False)
    col_cov = np.array(col_cov, dtype=np.float64, copy=False)

    if mean.ndim != 2:
        raise ValueError("mean must be a matrix")
    if row_cov.ndim != 2 or row_cov.shape[0] != row_cov.shape[1]:
        raise ValueError("row_cov must be square")
    if col_cov.ndim != 2 or col_cov.shape[0] != col_cov.shape[1]:
        raise ValueError("col_cov must be square")
    if mean.shape[0] != row_cov.shape[0]:
        raise ValueError("row dimension mismatch")
    if mean.shape[1] != col_cov.shape[0]:
        raise ValueError("column dimension mismatch")

    chol_row = safe_cholesky(row_cov)
    chol_col = safe_cholesky(col_cov)
    z = rng.standard_normal(size=mean.shape)
    sample = mean + chol_row @ z @ chol_col.T
    return sample


def ensure_spd_inverse(matrix: np.ndarray) -> np.ndarray:
    """Compute the inverse of an SPD matrix via Cholesky solves."""

    chol = safe_cholesky(matrix)
    inv = linalg.cho_solve((chol, True), np.eye(matrix.shape[0]))
    return (inv + inv.T) / 2.0
