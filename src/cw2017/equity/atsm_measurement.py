"""Exact affine term-structure measurement equations for CW2017."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import lax

Array = jnp.ndarray

_EPS = 1e-10


def _as_array(matrix: Array) -> Array:
    arr = jnp.asarray(matrix, dtype=jnp.float64)
    return arr


def _extract_lambda(LgQ: Dict[str, Array]) -> Array:
    lam = LgQ.get("Lambda", LgQ)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    if lam.ndim == 2:
        lam = jnp.diag(lam)
    return lam


def _extract_sigma(LgQ: Dict[str, Array]) -> Array:
    sigma = LgQ.get("Sigma", LgQ)
    return jnp.asarray(sigma, dtype=jnp.float64)


def _prepare_b_row(QgQ: Array, lam: Array) -> Tuple:
    Q = _as_array(QgQ)
    Q_inv = jsp.solve(Q, jnp.eye(Q.shape[0], dtype=jnp.float64))
    e1 = jnp.zeros(Q.shape[0], dtype=jnp.float64).at[0].set(1.0)
    e1Q = e1 @ Q

    def geom_sum(power: int) -> Array:
        lam_pow = jnp.power(lam, power)
        denom = 1.0 - lam
        ratio = jnp.where(jnp.abs(denom) < _EPS, power, (1.0 - lam_pow) / denom)
        return ratio

    def b_row(power: int) -> Array:
        geom = geom_sum(power)
        power_f = jnp.asarray(power, dtype=jnp.float64)
        return (e1Q * geom) @ Q_inv / power_f

    return b_row, Q, Q_inv


def compute_B(QgQ: Array, LgQ: Dict[str, Array], maturities: Iterable[int]) -> Array:
    """Return the measurement matrix ``B`` across the supplied maturities."""

    lam = _extract_lambda(LgQ)
    b_row, _, _ = _prepare_b_row(QgQ, lam)
    maturities = jnp.asarray(list(maturities), dtype=jnp.int32)
    return jax.vmap(b_row)(maturities)


def compute_A0_A1(
    B: Array,
    LgQ: Dict[str, Array],
    QgQ: Array,
    maturities: Iterable[int],
) -> Tuple[Array, Array]:
    """Compute ``A0`` and ``A1`` using the exact ATSM formulas."""

    lam = _extract_lambda(LgQ)
    sigma = _extract_sigma(LgQ)
    cov = sigma @ sigma.T

    maturities_list = tuple(maturities)
    maturities = jnp.asarray(maturities_list, dtype=jnp.int32)
    max_maturity = int(max(maturities_list)) if maturities_list else 0

    b_row, Q, Q_inv = _prepare_b_row(QgQ, lam)
    horizons = jnp.arange(1, max_maturity, dtype=jnp.int32)
    if horizons.size == 0:
        horizons = jnp.array([1], dtype=jnp.int32)
    b_history = jax.vmap(b_row)(horizons)

    i_vals = jnp.arange(1, b_history.shape[0] + 1, dtype=jnp.float64)
    quad_terms = jnp.einsum("ij,jk,ik->i", b_history, cov, b_history)
    weighted_quads = (i_vals ** 2) * quad_terms
    cumsum_quads = jnp.cumsum(weighted_quads)

    weighted_b = b_history * i_vals[:, None]
    cumsum_weighted_b = jnp.cumsum(weighted_b, axis=0)

    transform = jnp.eye(Q.shape[0], dtype=jnp.float64) - Q @ jnp.diag(lam) @ Q_inv

    def lookup_scalar(count: jnp.ndarray) -> Array:
        zero = jnp.array(0.0, dtype=jnp.float64)

        def true_fn(idx):
            return cumsum_quads[idx]

        return lax.cond(count > 0, true_fn, lambda _: zero, count - 1)

    def lookup_vector(count: jnp.ndarray) -> Array:
        zero_vec = jnp.zeros(B.shape[1], dtype=jnp.float64)

        def true_fn(idx):
            return cumsum_weighted_b[idx]

        return lax.cond(count > 0, true_fn, lambda _: zero_vec, count - 1)

    def a_terms(maturity: jnp.ndarray) -> Tuple[Array, Array]:
        count = maturity - 1
        maturity_f = maturity.astype(jnp.float64)
        sum_i2 = lookup_scalar(count)
        sum_ib = lookup_vector(count)
        a0 = jnp.where(maturity > 0, -0.5 * sum_i2 / maturity_f, 0.0)
        a1_row = jnp.where(maturity > 0, sum_ib @ transform / maturity_f, 0.0)
        return a0, a1_row

    a0_vals, a1_rows = jax.vmap(a_terms)(maturities)
    return a0_vals[:, None], a1_rows


def build_M0_M1(
    B: Array,
    LgQ: Dict[str, Array],
    maturities: Iterable[int],
) -> Tuple[Array, Array]:
    """Construct the ``M0^Q`` and ``M1^Q`` terms with ``n* = 60``."""

    lam = _extract_lambda(LgQ)
    sigma = _extract_sigma(LgQ)
    QgQ = LgQ.get("QgQ")
    if QgQ is None:
        raise ValueError("LgQ mapping must contain the 'QgQ' matrix for M0 construction")

    b_row, _, _ = _prepare_b_row(QgQ, lam)
    nstar = 60
    horizons = jnp.arange(1, nstar, dtype=jnp.int32)
    b_history = jax.vmap(b_row)(horizons)
    cov = sigma @ sigma.T
    quad_terms = jnp.einsum("ij,jk,ik->i", b_history, cov, b_history)
    i_vals = jnp.arange(1, nstar, dtype=jnp.float64)
    weighted_sum = jnp.sum((i_vals ** 2) * quad_terms)
    m0_first = -0.5 * weighted_sum / nstar

    M0 = jnp.zeros((B.shape[1], 1), dtype=jnp.float64)
    M0 = M0.at[0, 0].set(m0_first)
    M1 = jnp.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    return M0, M1


def build_measurement_terms(
    params3a,
    fixed: Dict[str, Array],
    maturities: Iterable[int],
):
    """Wrapper returning ``(A0, A1, B, M0^Q, M1^Q)`` for the Kalman filter."""

    QgQ = _as_array(fixed["Q_g^Q"])
    Lambda_gQ = fixed["Lambda_g^Q"]
    Sigma_gQ = _as_array(fixed["Sigma_g^Q"])
    measurement_dict = {"Lambda": Lambda_gQ, "Sigma": Sigma_gQ, "QgQ": QgQ}
    B = compute_B(QgQ, measurement_dict, maturities)
    A0, A1 = compute_A0_A1(B, measurement_dict, QgQ, maturities)
    M0Q, M1Q = build_M0_M1(B, measurement_dict, maturities)
    return A0, A1, B, M0Q, M1Q


__all__ = [
    "compute_B",
    "compute_A0_A1",
    "build_M0_M1",
    "build_measurement_terms",
]
