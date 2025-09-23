"""JAX implementation of the volatility diagonal mapping."""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

Array = jnp.ndarray


def diag_vols_series(
    Gamma0: Array,
    Gamma1: Array,
    H: Array,
    d_m: int,
    d_g: int,
    e_div_ix: int,
    use_lag_for_div: bool = True,
    h_pre: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Mirror of :func:`common.vol_mapping.diag_vols_series` using ``jax.numpy``."""

    H_arr = jnp.asarray(H, dtype=jnp.float64)
    Gamma0_arr = jnp.asarray(Gamma0, dtype=jnp.float64)
    Gamma1_arr = jnp.asarray(Gamma1, dtype=jnp.float64)

    z_all = Gamma0_arr[None, :] + H_arr @ Gamma1_arr.T
    diag_all = jnp.exp(0.5 * z_all)

    Dm = diag_all[:, :d_m]
    Dg = diag_all[:, d_m:]

    if use_lag_for_div and H_arr.shape[0] > 0:
        if h_pre is None:
            h_pre_arr = H_arr[0]
        else:
            h_pre_arr = jnp.asarray(h_pre, dtype=jnp.float64)
        H_prev = jnp.vstack([h_pre_arr[None, :], H_arr[:-1, :]])
        row = e_div_ix
        z_row = Gamma0_arr[row] + H_prev @ Gamma1_arr[row, :]
        Dm = Dm.at[:, row].set(jnp.exp(0.5 * z_row))

    return Dm, Dg


__all__ = ["diag_vols_series"]
