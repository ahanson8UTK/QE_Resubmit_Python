"""Unit tests for equity constraint construction."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from equity_constraint import compute_M0Q_from_B_and_Sigma_gQ, equity_coeffs
from tsm.q_covariance import compute_sigma_g_Q


def test_compute_m0q_matches_scalar_bruteforce():
    rng = np.random.default_rng(0)
    n_star = 5
    d_m = 2
    d_g = 3
    d_h = 4
    B_rows = rng.standard_normal((n_star - 1, d_g))
    Sigma_g = rng.standard_normal((d_g, d_g))
    Sigma_g = np.tril(Sigma_g)
    np.fill_diagonal(Sigma_g, np.abs(np.diag(Sigma_g)) + 0.5)
    Gamma0 = rng.standard_normal(d_m + d_g)
    Gamma1 = rng.standard_normal((d_m + d_g, d_h))
    mu_h_bar = rng.standard_normal(d_h)

    result = compute_M0Q_from_B_and_Sigma_gQ(
        B_rows,
        Sigma_g=Sigma_g,
        Gamma0=Gamma0,
        Gamma1=Gamma1,
        mu_h_bar=mu_h_bar,
        n_star=n_star,
    )

    Sigma_g_Q = compute_sigma_g_Q(Sigma_g, Gamma0, Gamma1, mu_h_bar, d_m, d_g)
    GG = Sigma_g_Q @ Sigma_g_Q.T
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

    Sigma_g = rng.standard_normal((d_g, d_g))
    Sigma_g = np.tril(Sigma_g)
    np.fill_diagonal(Sigma_g, np.abs(np.diag(Sigma_g)) + 0.5)
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
    Gamma0 = rng.standard_normal(d_m + d_g)
    Gamma1 = rng.standard_normal((d_m + d_g, d_h))
    M0Q = compute_M0Q_from_B_and_Sigma_gQ(
        B_rows,
        Sigma_g=Sigma_g,
        Gamma0=Gamma0,
        Gamma1=Gamma1,
        mu_h_bar=mu_h_bar,
        n_star=n_star,
    )
    M1Q = rng.standard_normal((d_g, 2))

    Sg_free = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)

    fixed = {
        "Phi_m": jnp.asarray(Phi_m, dtype=jnp.float64),
        "Phi_mg": jnp.asarray(Phi_mg, dtype=jnp.float64),
        "Phi_mh": jnp.asarray(Phi_mh, dtype=jnp.float64),
        "Phi_gQ": jnp.asarray(Phi_gQ, dtype=jnp.float64),
        "Sigma_g": jnp.asarray(Sigma_g, dtype=jnp.float64),
        "Sigma_h": jnp.asarray(Sigma_h, dtype=jnp.float64),
        "Sigma_hm": jnp.asarray(Sigma_hm, dtype=jnp.float64),
        "Sigma_hg": jnp.asarray(Sigma_hg, dtype=jnp.float64),
        "mu_h_bar": jnp.asarray(mu_h_bar, dtype=jnp.float64),
        "Gamma0": jnp.asarray(Gamma0, dtype=jnp.float64),
        "Gamma1": jnp.asarray(Gamma1, dtype=jnp.float64),
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
