import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from utils.linalg import chol_solve_spd, safe_cholesky
from utils.selectors import block_diag


def test_safe_cholesky_matches_numpy():
    rng = np.random.default_rng(123)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.5 * np.eye(4)
    B = rng.standard_normal((4, 3))

    S_jax = jnp.asarray(S, dtype=jnp.float64)
    B_jax = jnp.asarray(B, dtype=jnp.float64)

    L = safe_cholesky(S_jax)
    assert L.shape == (4, 4)
    assert jnp.allclose(L, jnp.tril(L))

    L_np = np.linalg.cholesky(S)
    np.testing.assert_allclose(np.array(L), L_np, atol=1e-8)

    x_jax = chol_solve_spd(L, B_jax)
    x_np = np.linalg.solve(S, B)
    np.testing.assert_allclose(np.array(x_jax), x_np, atol=1e-8)


def test_block_diag_three_blocks():
    b1 = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    b2 = jnp.array([[5.0, 6.0, 7.0]], dtype=jnp.float64)
    b3 = jnp.array([[8.0], [9.0], [10.0]], dtype=jnp.float64)

    result = block_diag(b1, b2, b3)

    expected = jnp.zeros((6, 6), dtype=jnp.float64)
    expected = expected.at[0:2, 0:2].set(b1)
    expected = expected.at[2:3, 2:5].set(b2)
    expected = expected.at[3:6, 5:6].set(b3)

    assert result.shape == (6, 6)
    np.testing.assert_allclose(np.array(result), np.array(expected))
