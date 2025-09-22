"""Selectors and structured matrix constructors."""

from __future__ import annotations

import jax.numpy as jnp


def e_last(d: int) -> jnp.ndarray:
    """Row selector for the last element of a length-``d`` vector."""
    if d < 1:
        raise ValueError("d must be positive")
    return jnp.eye(d, dtype=jnp.float64)[-1, :]


def e1(d: int) -> jnp.ndarray:
    """Row selector for the first element of a length-``d`` vector."""
    if d < 1:
        raise ValueError("d must be positive")
    return jnp.eye(d, dtype=jnp.float64)[0, :]


def block_diag(*blocks: jnp.ndarray) -> jnp.ndarray:
    """Block-diagonal concatenation of matrices using zeros for off-diagonals."""
    if not blocks:
        raise ValueError("block_diag requires at least one block")

    arrays = [jnp.asarray(block) for block in blocks]
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError("All blocks must be two-dimensional")

    dtype = jnp.result_type(*arrays)
    total_rows = sum(arr.shape[0] for arr in arrays)
    total_cols = sum(arr.shape[1] for arr in arrays)
    out = jnp.zeros((total_rows, total_cols), dtype=dtype)

    row_offset = 0
    col_offset = 0
    for arr in arrays:
        h, w = arr.shape
        out = out.at[row_offset : row_offset + h, col_offset : col_offset + w].set(
            jnp.asarray(arr, dtype=dtype)
        )
        row_offset += h
        col_offset += w

    return out
