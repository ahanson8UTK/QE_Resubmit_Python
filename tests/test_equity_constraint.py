"""Unit tests for equity constraint construction."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from equity_constraint import compute_M0Q_from_B_and_Sigma_gQ, equity_coeffs


def test_compute_m0q_matches_scalar_bruteforce():
    rng = np.random.default_rng(0)
    n_star = 5
    d_g = 3
    B_rows = rng.standard_normal((n_star - 1, d_g))
    Sigma_gQ = rng.standard_normal((d_g, d_g))

    result = compute_M0Q_from_B_and_Sigma_gQ(B_rows, Sigma_gQ, n_star=n_star)

    GG = Sigma_gQ @ Sigma_gQ.T
    inner = 0.0
    for i in range(1, n_star):
        b = B_rows[i - 1]
        inner += (i**2) * float(b @ GG @ b)
    expected = np.array([-0.5 * inner / n_star, 0.0, 0.0], dtype=np.float64)

    assert result.shape == (3,)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12, atol=1e-12)


def test_equity_coeffs_returns_finite_a_and_b_and_inequality():
    rng = np.random.default_rng(1)
    d_m = 3
    d_g = 3
    d_h = 2

    Phi_m = 0.1 * rng.standard_normal((d_m, d_m))
    Phi_mg = 0.1 * rng.standard_normal((d_m, d_g))
    Phi_mh = 0.1 * rng.standard_normal((d_m, d_h))
    Phi_gQ = 0.05 * rng.standard_normal((d_g, d_g))

    Sigma_gQ = rng.standard_normal((d_g, d_g))
    Ah = rng.standard_normal((d_h, d_h))
    Sigma_h = Ah @ Ah.T + np.eye(d_h)
    Sigma_hm = rng.standard_normal((d_h, d_m))
    Sigma_hg = rng.standard_normal((d_h, d_g))
    mu_h_bar = rng.standard_normal(d_h)

    bar_mm = np.eye(d_m)
    bar_mg = rng.standard_normal((d_m, d_g))
    bar_mh = rng.standard_normal((d_m, d_h))
    bar_hh = np.eye(d_h)

    n_star = 60
    B_rows = rng.standard_normal((n_star - 1, d_g))
    M0Q = compute_M0Q_from_B_and_Sigma_gQ(B_rows, Sigma_gQ, n_star=n_star)
    M1Q = rng.standard_normal((d_g, 2))

    Sg_free = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)

    fixed = {
        "Phi_m": jnp.asarray(Phi_m, dtype=jnp.float64),
        "Phi_mg": jnp.asarray(Phi_mg, dtype=jnp.float64),
        "Phi_mh": jnp.asarray(Phi_mh, dtype=jnp.float64),
        "Phi_gQ": jnp.asarray(Phi_gQ, dtype=jnp.float64),
        "Sigma_gQ": jnp.asarray(Sigma_gQ, dtype=jnp.float64),
        "Sigma_h": jnp.asarray(Sigma_h, dtype=jnp.float64),
        "Sigma_hm": jnp.asarray(Sigma_hm, dtype=jnp.float64),
        "Sigma_hg": jnp.asarray(Sigma_hg, dtype=jnp.float64),
        "mu_h_bar": jnp.asarray(mu_h_bar, dtype=jnp.float64),
        "bar": {
            "mm": jnp.asarray(bar_mm, dtype=jnp.float64),
            "mg": jnp.asarray(bar_mg, dtype=jnp.float64),
            "mh": jnp.asarray(bar_mh, dtype=jnp.float64),
            "hh": jnp.asarray(bar_hh, dtype=jnp.float64),
        },
        "M0Q": M0Q,
        "M1Q": jnp.asarray(M1Q, dtype=jnp.float64),
        "Sg_free": Sg_free,
    }

    cfg = {"dims": {"d_m": d_m}}

    a, b = equity_coeffs(fixed, cfg)

    expected_length = d_m + Sg_free.shape[0] + M1Q.shape[1]
    assert a.shape == (expected_length,)
    assert b.shape == ()
    assert jnp.all(jnp.isfinite(a))
    assert jnp.isfinite(b)

    norm_sq = float(a @ a)
    if norm_sq > 1e-10:
        scale = (abs(float(b)) + 1.0) / (norm_sq + 1e-12) + 1.0
        mu = -scale * a
    else:
        mu = -jnp.ones_like(a) * (abs(float(b)) + 1.0)

    value = float(a @ mu + b)
    assert value < -1e-6
