from __future__ import annotations

"""Utilities for constructing Q-measure covariances."""

from typing import Tuple

import numpy as np

Array = np.ndarray


def split_diag_mg(diag_vec: Array, d_m: int, d_g: int) -> Tuple[Array, Array]:
    """Split a length-``(d_m + d_g)`` vector into ``(D_m, D_g)`` diagonal vectors."""

    if diag_vec.shape != (d_m + d_g,):
        raise ValueError("diag_vec must have shape (d_m + d_g,)")
    return diag_vec[:d_m], diag_vec[d_m:]


def compute_sigma_g_Q(
    Sigma_g: Array,
    Gamma0: Array,
    Gamma1: Array,
    mu_h_bar: Array,
    d_m: int,
    d_g: int,
) -> Array:
    """Return ``Sigma_g^Q = Sigma_g diag(\bar D_g)`` without inverting matrices.

    The long-run diagonal scaling obeys

    ``diag([\bar D_m; \bar D_g]) = exp((Gamma0 + Gamma1 mu_h_bar) / 2)``.

    Parameters
    ----------
    Sigma_g:
        Unit lower-triangular matrix for ``g`` innovations under ``P``.
    Gamma0, Gamma1, mu_h_bar:
        Long-run volatility mapping parameters.
    d_m, d_g:
        Dimensions for the macro and growth blocks.
    """

    xp = np
    dtype = np.float64
    try:
        Sigma_g_arr = np.asarray(Sigma_g, dtype=dtype)
        Gamma0_arr = np.asarray(Gamma0, dtype=dtype)
        Gamma1_arr = np.asarray(Gamma1, dtype=dtype)
        mu_h_arr = np.asarray(mu_h_bar, dtype=dtype)
    except Exception as err:  # pragma: no cover - only hit under JAX tracing
        try:
            from jax.errors import TracerArrayConversionError
        except ModuleNotFoundError as import_err:  # pragma: no cover - safety guard
            raise err from import_err
        if not isinstance(err, TracerArrayConversionError):
            raise
        import jax.numpy as jnp

        xp = jnp
        dtype = jnp.float64
        Sigma_g_arr = jnp.asarray(Sigma_g, dtype=dtype)
        Gamma0_arr = jnp.asarray(Gamma0, dtype=dtype)
        Gamma1_arr = jnp.asarray(Gamma1, dtype=dtype)
        mu_h_arr = jnp.asarray(mu_h_bar, dtype=dtype)

    if Sigma_g_arr.shape != (d_g, d_g):
        raise ValueError("Sigma_g must be square with shape (d_g, d_g)")
    if Gamma0_arr.shape != (d_m + d_g,):
        raise ValueError("Gamma0 must have length d_m + d_g")
    if Gamma1_arr.shape != (d_m + d_g, mu_h_arr.shape[0]):
        raise ValueError("Gamma1 must have shape (d_m + d_g, d_h)")

    z_all = Gamma0_arr + Gamma1_arr @ mu_h_arr
    diag_all = xp.exp(0.5 * z_all)
    _, Dg_bar = split_diag_mg(diag_all, d_m, d_g)
    return Sigma_g_arr @ xp.diag(Dg_bar)


def _assert_spd_condition() -> None:
    """Sanity check ensuring the implied covariance is SPD for valid inputs."""

    d_m, d_g = 1, 2
    Sigma_g = np.array([[1.1, 0.0], [0.2, 0.9]], dtype=np.float64)
    Gamma0 = np.array([0.0, 0.1, -0.05], dtype=np.float64)
    Gamma1 = np.array([[0.05], [0.02], [0.01]], dtype=np.float64)
    mu_h_bar = np.array([0.3], dtype=np.float64)

    Sigma_g_Q = compute_sigma_g_Q(Sigma_g, Gamma0, Gamma1, mu_h_bar, d_m, d_g)
    cov = Sigma_g_Q @ Sigma_g_Q.T
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0.0)


_assert_spd_condition()

