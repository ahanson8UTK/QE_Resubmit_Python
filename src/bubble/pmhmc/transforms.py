"""Parameter space transforms for the bubble PM-HMC module."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .types import BubbleParams, BubbleParamsUnconstrained

ArrayLike = float | NDArray[np.float_]

PHI_B_LOWER = -0.999
PHI_B_UPPER = 0.999


def _softplus_raw(x: ArrayLike) -> NDArray[np.float_]:
    """Raw softplus without minimum clipping."""

    x_arr = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)


def softplus(x: ArrayLike, min: float = 1e-6) -> NDArray[np.float_]:
    """Numerically stable softplus transform with a lower bound."""

    x_arr = np.asarray(x, dtype=float)
    out = _softplus_raw(x_arr)
    if min > 0.0:
        out = np.maximum(out, min)
    return out


def inv_softplus(y: ArrayLike, min: float = 1e-6) -> NDArray[np.float_]:
    """Inverse of :func:`softplus` on the positive real line."""

    y_arr = np.asarray(y, dtype=float)
    y_adj = np.maximum(y_arr, min)
    out = y_adj + np.log1p(-np.exp(-y_adj))
    return out


def tanh_to_interval(
    z: ArrayLike, a: float = PHI_B_LOWER, b: float = PHI_B_UPPER
) -> NDArray[np.float_]:
    """Map real values to the open interval ``(a, b)`` using ``tanh``."""

    if not a < b:
        raise ValueError("Lower bound must be strictly less than upper bound.")
    z_arr = np.asarray(z, dtype=float)
    tanh_z = np.tanh(z_arr)
    half_range = 0.5 * (b - a)
    center = 0.5 * (a + b)
    out = center + half_range * tanh_z
    return out


def _log_sigmoid(x: NDArray[np.float_]) -> NDArray[np.float_]:
    """Stable computation of ``log(sigmoid(x))``."""

    x_arr = np.asarray(x, dtype=float)
    raw = -_softplus_raw(-x_arr)
    return raw


def renorm_rho(
    z_rho_bm: NDArray[np.float_], z_rho_bg: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_], float]:
    """Shrink unconstrained vectors to the open unit ball.

    The mapping ``v -> v / sqrt(1 + ||v||^2)`` ensures that the resulting
    vector has Euclidean norm strictly less than one.  For robustness we
    treat the unconstrained ``z`` variables as primary and therefore ignore
    the Jacobian adjustment of this shrinkage, effectively viewing it as part
    of the parameterization.  The returned log-Jacobian contribution is zero
    with this convention.
    """

    v = np.concatenate((np.asarray(z_rho_bm, dtype=float), np.asarray(z_rho_bg, dtype=float)))
    norm_sq = np.dot(v, v)
    scale = 1.0 / np.sqrt(1.0 + norm_sq)
    u = v * scale
    d_m = z_rho_bm.shape[0]
    rho_bm = u[:d_m].copy()
    rho_bg = u[d_m:].copy()
    log_jac = 0.0
    return rho_bm, rho_bg, log_jac


def unconstrained_to_constrained(
    u: BubbleParamsUnconstrained,
) -> Tuple[BubbleParams, float]:
    """Map unconstrained parameters to their constrained counterparts."""

    B0 = softplus(u.z_B0)
    sigma_h = softplus(u.z_sigma_h)
    phi_b = tanh_to_interval(u.z_phi_b, PHI_B_LOWER, PHI_B_UPPER)
    rho_bm, rho_bg, log_jac_rho = renorm_rho(u.z_rho_bm, u.z_rho_bg)

    params = BubbleParams(
        B0=float(B0),
        mu_b=float(u.z_mu_b),
        phi_b=float(phi_b),
        sigma_h=float(sigma_h),
        rho_bm=rho_bm,
        rho_bg=rho_bg,
    )

    z_B0 = np.asarray(u.z_B0, dtype=float)
    z_sigma_h = np.asarray(u.z_sigma_h, dtype=float)
    z_phi_b = np.asarray(u.z_phi_b, dtype=float)

    log_jacobian = float(np.sum(_log_sigmoid(z_B0)))
    log_jacobian += float(np.sum(_log_sigmoid(z_sigma_h)))

    tanh_phi = np.tanh(z_phi_b)
    half_range = 0.5 * (PHI_B_UPPER - PHI_B_LOWER)
    tanh_sq = np.clip(tanh_phi**2, 0.0, 1.0 - 1e-12)
    log_jacobian += float(np.sum(np.log(half_range) + np.log1p(-tanh_sq)))
    log_jacobian += log_jac_rho

    return params, log_jacobian


__all__ = [
    "softplus",
    "inv_softplus",
    "tanh_to_interval",
    "renorm_rho",
    "unconstrained_to_constrained",
]

