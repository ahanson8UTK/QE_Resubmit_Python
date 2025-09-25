"""Shared numerical utilities for the CW2017 Kalman routines."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsp

Array = jnp.ndarray

_JITTER = 1e-10


def compute_diag_scales(Gamma0: Array, Gamma1: Array, h_vec: Array) -> Array:
    """Return ``exp(0.5 * (Gamma0 + Gamma1 h_t))`` with clipping for stability."""

    h_arr = jnp.asarray(h_vec, dtype=jnp.float64)
    Gamma0_arr = jnp.asarray(Gamma0, dtype=jnp.float64)
    Gamma1_arr = jnp.asarray(Gamma1, dtype=jnp.float64)

    if h_arr.ndim == 1:
        exponent = Gamma0_arr + h_arr @ Gamma1_arr.T
    else:
        exponent = Gamma0_arr + h_arr @ Gamma1_arr.T
    exponent = jnp.clip(exponent, -20.0, 20.0)
    return jnp.exp(0.5 * exponent)


def symmetrize_psd(matrix: Array) -> Array:
    return (matrix + matrix.T) * 0.5


def cholesky_psd(matrix: Array) -> Array:
    matrix = symmetrize_psd(matrix)
    dim = matrix.shape[0]
    jitter_eye = _JITTER * jnp.eye(dim, dtype=matrix.dtype)
    return jsp.cholesky(matrix + jitter_eye, lower=True)


__all__ = ["compute_diag_scales", "symmetrize_psd", "cholesky_psd"]

