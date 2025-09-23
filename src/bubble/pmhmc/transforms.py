"""Parameter space transforms for the bubble PM-HMC module."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .types import BubbleParams, BubbleParamsUnconstrained

ArrayLike = float | NDArray[np.float_]


def _as_native_type(value: NDArray[np.float_], original: ArrayLike) -> ArrayLike:
    """Return ``value`` as a Python float when ``original`` was scalar."""
    if np.isscalar(original):
        return float(value)
    if isinstance(original, np.ndarray) and value.shape == ():
        return float(value)
    return value


def softplus(x: ArrayLike) -> ArrayLike:
    """Numerically stable softplus transform.

    Parameters
    ----------
    x:
        Input scalar or array.

    Returns
    -------
    ArrayLike
        Softplus of the input with the same shape as ``x``.
    """

    x_arr = np.asarray(x, dtype=float)
    out = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)
    return _as_native_type(out, x)


def inv_softplus(y: ArrayLike) -> ArrayLike:
    """Inverse of :func:`softplus` on the positive real line."""

    y_arr = np.asarray(y, dtype=float)
    out = np.log(np.expm1(y_arr))
    return _as_native_type(out, y)


def tanh_constrain(x: ArrayLike) -> ArrayLike:
    """Map real values to ``(-1, 1)`` via the hyperbolic tangent."""

    x_arr = np.asarray(x, dtype=float)
    out = np.tanh(x_arr)
    return _as_native_type(out, x)


def artanh_unconstrain(y: ArrayLike) -> ArrayLike:
    """Inverse of :func:`tanh_constrain` with domain ``(-1, 1)``."""

    y_arr = np.asarray(y, dtype=float)
    if np.any(np.abs(y_arr) >= 1.0):
        raise ValueError("Input to artanh_unconstrain must have absolute value < 1.")
    out = np.arctanh(y_arr)
    return _as_native_type(out, y)


def to_unit_ball(z: Iterable[float]) -> NDArray[np.float_]:
    """Project unconstrained values to the open unit ball.

    The mapping ``v -> v / (1 + ||v||)`` ensures that the resulting vector
    has Euclidean norm strictly less than one.  This is convenient for
    correlation-style parameters that must satisfy
    ``||rho_bm||^2 + ||rho_bg||^2 < 1``.
    """

    z_arr = np.asarray(list(z), dtype=float)
    norm = np.linalg.norm(z_arr)
    scale = 1.0 / (1.0 + norm)
    return z_arr * scale


def from_unit_ball(v: Iterable[float]) -> NDArray[np.float_]:
    """Inverse transformation for :func:`to_unit_ball`."""

    v_arr = np.asarray(list(v), dtype=float)
    norm = np.linalg.norm(v_arr)
    if norm >= 1.0:
        raise ValueError("Input to from_unit_ball must lie strictly inside the unit ball.")
    scale = 1.0 / (1.0 - norm)
    return v_arr * scale


def constrain_params(params: BubbleParamsUnconstrained) -> BubbleParams:
    """Map unconstrained parameters onto the interpretable constrained space."""

    rho_concat = np.concatenate((params.z_rho_bm, params.z_rho_bg))
    rho = to_unit_ball(rho_concat)
    d_m = params.z_rho_bm.shape[0]
    rho_bm = rho[:d_m]
    rho_bg = rho[d_m:]

    return BubbleParams(
        B0=float(softplus(params.z_B0)),
        mu_b=float(params.z_mu_b),
        phi_b=float(tanh_constrain(params.z_phi_b)),
        sigma_h=float(softplus(params.z_sigma_h)),
        rho_bm=rho_bm,
        rho_bg=rho_bg,
    )


def unconstrain_params(params: BubbleParams) -> BubbleParamsUnconstrained:
    """Map constrained parameters back to an unconstrained representation."""

    rho_concat = np.concatenate((params.rho_bm, params.rho_bg))
    rho_z = from_unit_ball(rho_concat)
    d_m = params.rho_bm.shape[0]
    rho_bm_z = rho_z[:d_m]
    rho_bg_z = rho_z[d_m:]

    return BubbleParamsUnconstrained(
        z_B0=float(inv_softplus(params.B0)),
        z_mu_b=float(params.mu_b),
        z_phi_b=float(artanh_unconstrain(params.phi_b)),
        z_sigma_h=float(inv_softplus(params.sigma_h)),
        z_rho_bm=rho_bm_z,
        z_rho_bg=rho_bg_z,
    )


__all__ = [
    "softplus",
    "inv_softplus",
    "tanh_constrain",
    "artanh_unconstrain",
    "to_unit_ball",
    "from_unit_ball",
    "constrain_params",
    "unconstrain_params",
]
