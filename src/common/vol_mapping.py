"""Volatility diagonal construction helpers.

The baseline mapping follows Creal & Wu (2017) Eq. (3)::

    diag([D_{m,t}; D_{g,t}]) = exp{ 0.5 * (Gamma0 + Gamma1 h_t) }.

For the dividend state we override the timing so that the dividend row of
``D_{m,t}`` evaluates at ``h_{t-1}``, reflecting the driven-volatility
convention discussed around Eqs. (1)–(4). All other rows continue to use
``h_t`` directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


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
    """Return ``diag(D_{m,t})`` and ``diag(D_{g,t})`` for ``t = 0, …, T-1``.

    Parameters
    ----------
    Gamma0, Gamma1:
        Volatility mapping parameters with shapes ``(d_m + d_g,)`` and
        ``(d_m + d_g, d_h)`` respectively.
    H:
        Sequence of volatility states ``h_t`` with shape ``(T, d_h)``.
    d_m, d_g:
        Dimensions of the macro and yield blocks.
    e_div_ix:
        Index of the log-dividend growth row within the macro block.
    use_lag_for_div:
        When ``True``, the dividend row uses ``h_{t-1}`` instead of ``h_t``.
    h_pre:
        Optional pre-sample value used as ``h_{-1}`` for ``t = 0`` when the
        dividend row uses lagged volatility. Defaults to ``H[0]`` if omitted.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays with shapes ``(T, d_m)`` and ``(T, d_g)`` containing the
        diagonal elements of ``D_{m,t}`` and ``D_{g,t}``.
    """

    T, d_h = H.shape
    if Gamma0.shape != (d_m + d_g,):  # pragma: no cover - sanity guard
        raise AssertionError("Gamma0 must have length d_m + d_g")
    if Gamma1.shape != (d_m + d_g, d_h):  # pragma: no cover - sanity guard
        raise AssertionError("Gamma1 must have shape (d_m + d_g, d_h)")
    if not (0 <= e_div_ix < d_m):  # pragma: no cover - sanity guard
        raise AssertionError("e_div_ix must index the macro block")

    z_all = Gamma0[None, :] + H @ Gamma1.T
    diag_all = np.exp(0.5 * z_all)

    Dm = diag_all[:, :d_m].copy()
    Dg = diag_all[:, d_m:]

    if use_lag_for_div and T > 0:
        if h_pre is None:
            h_pre = H[0]
        H_prev = np.vstack([np.asarray(h_pre)[None, :], H[:-1, :]])
        row = e_div_ix
        z_row = Gamma0[row] + H_prev @ Gamma1[row, :].T
        Dm[:, row] = np.exp(0.5 * z_row)

    return Dm, Dg


__all__ = ["diag_vols_series"]
