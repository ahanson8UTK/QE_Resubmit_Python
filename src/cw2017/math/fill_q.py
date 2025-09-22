r"""Utilities for constructing the eigenvector matrix ``Q`` in CW2017.

The original MATLAB code released alongside Creal & Wu (2017) constructs the
matrix ``Q`` by filling a sparse pattern with free parameters and then applying
several deterministic adjustments.  The :func:`fill_Q_jax` function is a direct
translation of Appendix B.2.2 (step 1) with ``vTau_star`` fixed to ``60`` as
specified by the user.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

Array = jnp.ndarray

_EPS = 1e-10


def _mask_structure(info: Dict[str, int]):
    N = int(info["Nstates"])
    Nm = int(info["Nm"])
    Ng = int(info["Ng"])
    Nmg = Nm + Ng

    mask = [[False for _ in range(N)] for _ in range(N)]
    rows: list[int] = []
    cols: list[int] = []
    for i in range(N):
        if i == Nm + 1:
            continue
        for j in range(N):
            if i == j:
                continue
            if i >= Nmg and j < Nmg:
                continue
            if i == Nm and j == Nm + 1:
                continue
            mask[i][j] = True
            rows.append(i)
            cols.append(j)
    return mask, rows, cols


def _build_q_mask(info: Dict[str, int]) -> Array:
    mask, _, _ = _mask_structure(info)
    return jnp.asarray(mask, dtype=jnp.bool_)


def count_qs_entries(info: Dict[str, int]) -> int:
    """Return the number of free parameters needed for ``Q`` construction."""

    _, rows, _ = _mask_structure(info)
    return len(rows)


def _geom_tilde(eigs: Array, vTau_star: int) -> Array:
    powers = jnp.power(eigs, vTau_star)
    numerator = 1.0 - powers
    denom = 1.0 - eigs
    ratio = jnp.where(jnp.abs(denom) < _EPS, vTau_star, numerator / denom)
    return ratio / vTau_star


def fill_Q_jax(eigs_diag: Array, qs: Array, info: Dict[str, int]) -> Array:
    r"""Construct the eigenvector matrix ``Q`` used in the VAR parameterisation.

    Parameters
    ----------
    eigs_diag:
        Length-``N`` vector containing the diagonal elements of ``\Lambda``.
    qs:
        Vector supplying the free off-diagonal elements of ``Q`` following the
        row-major traversal defined by the MATLAB mask.
    info:
        Dictionary with entries ``Nm``, ``Ng``, ``Nh``, ``Nstates`` and
        ``vTau_star``.  The latter is fixed to ``60`` according to the user
        specification.

    Returns
    -------
    Array
        The full ``Q`` matrix with unit diagonal.

    Notes
    -----
    The implementation mirrors the MATLAB logic line-by-line:

    1. ``eigs_tilde = diag((1/vTau_star) * (I - eigs^vTau_star) * (I - eigs)^-1)``
       computed element-wise while guarding against division by zero.
    2. Build the mask matrix described in Sec. A.2 by starting from an all-ones
       matrix, zeroing the diagonal, the lower-left block below row ``Nm+Ng``
       and two additional entries ``(Nm, Nm+1)`` and ``(Nm+1, :)``.
    3. Fill the off-diagonal entries of ``Q`` in row-major order using ``qs``.
    4. Apply the post-processing tweaks from the MATLAB implementation:
       ``Q[Nmg-1, :] = Q[Nmg-2, :] * eigs_tilde`` (element-wise) and
       ``Q[Nm+1, Nm+2] = 1 / eigs_tilde[Nm+2]``.
    5. Finally add the identity matrix so that ``q_ii = 1``.
    """

    eigs_diag = jnp.asarray(eigs_diag, dtype=jnp.float64)
    qs = jnp.asarray(qs, dtype=jnp.float64)

    vTau_star = int(info.get("vTau_star", 60))
    Nm = int(info["Nm"])
    Ng = int(info["Ng"])
    N = int(info["Nstates"])
    Nmg = Nm + Ng

    expected = count_qs_entries(info)
    if qs.shape[0] != expected:
        raise ValueError(
            "qs has incorrect size: expected %d, received %d" % (expected, qs.shape[0])
        )

    eigs_tilde = _geom_tilde(eigs_diag, vTau_star)

    mask, rows_list, cols_list = _mask_structure(info)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    rows = jnp.asarray(rows_list, dtype=jnp.int32)
    cols = jnp.asarray(cols_list, dtype=jnp.int32)
    Q = jnp.zeros((N, N), dtype=jnp.float64)
    Q = Q.at[rows, cols].set(qs)

    Q = Q.at[Nmg - 1, :].set(Q[Nmg - 2, :] * eigs_tilde)
    Q = Q.at[Nm + 1, Nm + 2].set(1.0 / eigs_tilde[Nm + 2])
    Q = Q + jnp.eye(N, dtype=jnp.float64)
    return Q


__all__ = ["fill_Q_jax", "count_qs_entries"]
