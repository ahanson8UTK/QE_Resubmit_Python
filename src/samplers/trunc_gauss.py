"""Truncated Gaussian sampling routines."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from utils.linalg import safe_cholesky

Array = jax.Array


def sample_halfspace_trunc_normal(
    key: jax.Array,
    mean: Array | jnp.ndarray,
    cov: Array | jnp.ndarray,
    a: Array | jnp.ndarray,
    b: float | Array | jnp.ndarray,
) -> Array:
    """Draw ``X ~ N(mean, cov)`` subject to ``aᵀX + b < 0``.

    Parameters
    ----------
    key:
        PRNG key used for sampling.
    mean:
        Mean vector of the Gaussian distribution.
    cov:
        Symmetric positive-definite covariance matrix of the Gaussian.
    a:
        Coefficients describing the half-space constraint.
    b:
        Offset describing the half-space constraint.

    Returns
    -------
    jax.Array
        A sample from the truncated Gaussian distribution.
    """

    mean_vec = jnp.atleast_1d(mean).astype(jnp.float64)
    a_vec = jnp.atleast_1d(a).astype(jnp.float64)
    cov_mat = jnp.asarray(cov, dtype=jnp.float64)
    L = safe_cholesky(cov_mat)

    c = L.T @ a_vec
    cnorm = jnp.linalg.norm(c)

    def _nonbinding(_: None) -> Array:
        z = jax.random.normal(key, shape=mean_vec.shape, dtype=jnp.float64)
        return mean_vec + L @ z

    def _binding(_: None) -> Array:
        k1, k2 = jax.random.split(key)

        denom = jnp.maximum(cnorm, 1e-300)
        kappa = -(a_vec @ mean_vec + jnp.asarray(b, dtype=jnp.float64)) / denom
        Phi = norm.cdf(kappa)
        Phi = jnp.clip(Phi, min=1e-300, max=1.0 - 1e-12)

        u_base = jax.random.uniform(
            k1,
            shape=(),
            minval=1e-12,
            maxval=1.0 - 1e-12,
            dtype=jnp.float64,
        )
        u_uniform = Phi * u_base
        u_std = norm.ppf(jnp.clip(u_uniform, min=1e-300, max=1.0 - 1e-12))
        u = u_std * cnorm

        c_hat = c / denom
        z = jax.random.normal(k2, shape=mean_vec.shape, dtype=jnp.float64)
        projection = jnp.dot(z, c_hat)
        z_perp = z - projection * c_hat

        return mean_vec + L @ (u * c_hat + z_perp)

    return jax.lax.cond(cnorm < 1e-14, _nonbinding, _binding, operand=None)


__all__ = ["sample_halfspace_trunc_normal"]
