"""Linear algebra utilities built on top of JAX."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

Array = jax.Array


def symmetrize(S: Array | jnp.ndarray) -> Array:
    """Return the symmetric part of ``S`` as a float64 array."""
    S64 = jnp.asarray(S, dtype=jnp.float64)
    return 0.5 * (S64 + S64.T)


def safe_cholesky(S: Array | jnp.ndarray, jitter: float = 1e-10, max_tries: int = 5) -> Array:
    """Return a lower-triangular ``L`` with ``S ≈ L Lᵀ``.

    Parameters
    ----------
    S:
        Symmetric positive definite matrix to factorise. The input is symmetrised
        to avoid numerical asymmetry issues and is always treated as ``float64``.
    jitter:
        Initial diagonal regularisation added when attempting the factorisation.
    max_tries:
        Maximum number of attempts made with exponentially increasing jitter.

    Returns
    -------
    jax.Array
        Lower-triangular Cholesky factor of ``S``.

    Raises
    ------
    np.linalg.LinAlgError
        If the matrix cannot be factorised after ``max_tries`` attempts.
    """

    if max_tries < 1:
        raise ValueError("max_tries must be at least 1")

    S_sym = symmetrize(S)
    if S_sym.ndim != 2 or S_sym.shape[0] != S_sym.shape[1]:
        raise ValueError("S must be a square matrix")
    n = S_sym.shape[0]

    eye = jnp.eye(n, dtype=jnp.float64)
    eps = 0.0
    for _ in range(max_tries):
        try:
            return jnp.linalg.cholesky(S_sym + (eps + jitter) * eye)
        except Exception:  # pragma: no cover - JAX raises custom errors
            eps = 10.0 * (eps + jitter)
    raise np.linalg.LinAlgError("Cholesky failed")


def chol_solve_spd(L: Array | jnp.ndarray, B: Array | jnp.ndarray) -> Array:
    """Solve ``(L Lᵀ) X = B`` for ``X`` using the provided Cholesky factor ``L``."""
    L64 = jnp.asarray(L, dtype=jnp.float64)
    B64 = jnp.asarray(B, dtype=jnp.float64)
    y = solve_triangular(L64, B64, lower=True, trans="N")
    x = solve_triangular(L64, y, lower=True, trans="T")
    return x


def tri_solve(
    L: Array | jnp.ndarray,
    B: Array | jnp.ndarray,
    *,
    lower: bool = True,
    trans: str | int = "N",
) -> Array:
    """Solve a triangular system using float64 precision."""
    L64 = jnp.asarray(L, dtype=jnp.float64)
    B64 = jnp.asarray(B, dtype=jnp.float64)
    return solve_triangular(L64, B64, lower=lower, trans=trans)
