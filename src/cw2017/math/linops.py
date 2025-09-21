"""Linear operator utilities.

TODOs:
- implement :func:`apply_block_diag` for structured multiplications
- implement :func:`solve_lower_triangular` with efficient triangular solves
- add caching for repeated Kronecker products
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from ..typing import Array


def apply_block_diag(blocks: Sequence[Array], x: Array) -> Array:
    """Apply block-diagonal matrices to ``x``.

    Parameters
    ----------
    blocks:
        Sequence of square matrices forming the diagonal blocks.
    x:
        Input array whose leading dimension matches the number of blocks.
    """

    raise NotImplementedError("apply_block_diag remains to be implemented")


def solve_lower_triangular(triangular: Array, b: Array) -> Array:
    """Solve ``triangular @ x = b`` for ``x``.

    Notes
    -----
    This is a placeholder that will be replaced with a numerically stable
    forward-substitution routine once the state-space structure is finalised.
    """

    raise NotImplementedError("Triangular solve is pending implementation")


__all__ = ["apply_block_diag", "solve_lower_triangular"]
