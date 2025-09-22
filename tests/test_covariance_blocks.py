"""Unit tests for covariance block draws with equity constraint."""

from __future__ import annotations

import numpy as np
import pytest

from tsm.covariance_blocks import (
    CovarianceBlockPriors,
    EquityConstraintParams,
    MatrixNormalInverseWishartPrior,
    TriangularRegressionPrior,
    draw_sigma_gm_and_eps_g,
    draw_sigma_h_block_with_constraint,
    draw_sigma_m,
)
from tsm.equity_constraint import equity_margin


D_M = 5
D_G = 3
D_H = 7
T_SMALL = 200


def _make_setup(seed: int = 0):
    rng = np.random.default_rng(seed)

    Dm = rng.uniform(0.8, 1.2, size=(T_SMALL, D_M))
    Dg = rng.uniform(0.8, 1.2, size=(T_SMALL, D_G))

    Sigma_m_true = np.eye(D_M)
    for i in range(1, D_M - 1):
        Sigma_m_true[i, :i] = 0.15 * rng.standard_normal(i)
    Sigma_m_true[-1, :-1] = 0.0
    Sigma_m_true[-1, -1] = 1.0

    Sigma_g = np.eye(D_G)
    for i in range(1, D_G):
        Sigma_g[i, :i] = 0.1 * rng.standard_normal(i)

    Sigma_gm_true = 0.05 * rng.standard_normal((D_G, D_M))
    base_h = 0.08 * rng.standard_normal((D_H, D_H))
    Sigma_h_true = np.tril(base_h)
    diag = np.abs(np.diag(Sigma_h_true)) + 0.5
    np.fill_diagonal(Sigma_h_true, diag)
    Sigma_hm_true = 0.05 * rng.standard_normal((D_H, D_M))
    Sigma_hg_true = 0.05 * rng.standard_normal((D_H, D_G))

    eps_m = rng.standard_normal((T_SMALL, D_M))
    eps_g = rng.standard_normal((T_SMALL, D_G))
    eps_h = rng.standard_normal((T_SMALL, D_H))

    r_m = np.empty((T_SMALL, D_M))
    r_g = np.empty((T_SMALL, D_G))
    r_h = np.empty((T_SMALL, D_H))
    for t in range(T_SMALL):
        r_m[t] = Sigma_m_true @ (Dm[t] * eps_m[t])
        r_g[t] = Sigma_gm_true @ eps_m[t] + Sigma_g @ (Dg[t] * eps_g[t])
        r_h[t] = (
            Sigma_hm_true @ eps_m[t]
            + Sigma_hg_true @ eps_g[t]
            + Sigma_h_true @ eps_h[t]
        )

    row_covs_m = [np.zeros((0, 0))]
    for i in range(1, D_M):
        row_covs_m.append(np.eye(i) * 100.0)
    sigma_m_prior = TriangularRegressionPrior(row_covariances=row_covs_m)

    row_covs_gm = [np.eye(D_M) * 100.0 for _ in range(D_G)]
    sigma_gm_prior = TriangularRegressionPrior(row_covariances=row_covs_gm)

    M0 = np.zeros((D_H, D_M + D_G))
    V0 = np.eye(D_M + D_G) * 10.0
    S0 = np.eye(D_H) * 1e-6
    sigma_h_prior = MatrixNormalInverseWishartPrior(M0=M0, V0=V0, S0=S0, nu0=D_H + 5.0)

    priors = CovarianceBlockPriors(
        rng=rng,
        sigma_m=sigma_m_prior,
        sigma_gm=sigma_gm_prior,
        sigma_h=sigma_h_prior,
    )

    Phi_m = 0.05 * rng.standard_normal((D_M, D_M))
    Phi_mg = 0.05 * rng.standard_normal((D_M, D_G))
    Phi_mh = 0.05 * rng.standard_normal((D_M, D_H))
    Phi_h = 0.02 * rng.standard_normal((D_H, D_H))
    Phi_g_Q = 0.03 * rng.standard_normal((D_G, D_G))

    theta_m = np.zeros(D_M)
    theta_m[0] = -5.0
    theta_g = np.zeros(D_G)
    theta_g_Q = np.zeros(D_G)
    mu_m_bar = np.zeros(D_M)
    mu_m_bar[0] = 1.0
    mu_g_u_bar = np.zeros(D_G)
    mu_g_Q_u_bar = np.zeros(D_G)
    mu_h_bar = np.zeros(D_H)
    Gamma0 = 0.01 * rng.standard_normal(D_M + D_G)
    Gamma1 = 0.01 * rng.standard_normal((D_M + D_G, D_H))
    M0_Q = np.zeros(D_G)
    M1_Q = np.eye(D_G)

    equity_params = EquityConstraintParams(
        theta_m=theta_m,
        theta_g=theta_g,
        theta_g_Q=theta_g_Q,
        mu_m_bar=mu_m_bar,
        mu_g_u_bar=mu_g_u_bar,
        mu_g_Q_u_bar=mu_g_Q_u_bar,
        mu_h_bar=mu_h_bar,
        Phi_m=Phi_m,
        Phi_mg=Phi_mg,
        Phi_mh=Phi_mh,
        Phi_h=Phi_h,
        M0_Q=M0_Q,
        M1_Q=M1_Q,
        Sigma_g=Sigma_g,
        Gamma0=Gamma0,
        Gamma1=Gamma1,
        Phi_g_Q=Phi_g_Q,
    )

    margin_true = equity_margin(
        equity_params.theta_m,
        equity_params.theta_g,
        equity_params.theta_g_Q,
        equity_params.mu_m_bar,
        equity_params.mu_g_u_bar,
        equity_params.mu_g_Q_u_bar,
        equity_params.Phi_m,
        equity_params.Phi_mg,
        equity_params.Phi_mh,
        equity_params.Phi_h,
        equity_params.M0_Q,
        equity_params.M1_Q,
        equity_params.Sigma_g,
        equity_params.Gamma0,
        equity_params.Gamma1,
        Sigma_hm_true,
        Sigma_hg_true,
        Sigma_h_true,
        equity_params.mu_h_bar,
        equity_params.Phi_g_Q,
    )
    if margin_true >= -1.0:
        theta_m = equity_params.theta_m.copy()
        theta_m[0] -= margin_true + 2.0
        equity_params.theta_m = theta_m

    return {
        "r_m": r_m,
        "Dm": Dm,
        "r_g": r_g,
        "Dg": Dg,
        "r_h": r_h,
        "Sigma_g": Sigma_g,
        "priors": priors,
        "equity_params": equity_params,
    }


def test_covariance_draws_shapes_and_constraints():
    setup = _make_setup(seed=123)

    Sigma_m, eps_m = draw_sigma_m(setup["r_m"], setup["Dm"], setup["priors"])
    Sigma_gm, eps_g = draw_sigma_gm_and_eps_g(
        setup["r_g"],
        setup["Dg"],
        setup["Sigma_g"],
        eps_m,
        setup["priors"],
    )
    Sigma_hm, Sigma_hg, Sigma_h, diagnostics = draw_sigma_h_block_with_constraint(
        setup["r_h"],
        eps_m,
        eps_g,
        setup["equity_params"],
        setup["priors"],
        max_rejects=100,
    )

    assert np.allclose(Sigma_m, np.tril(Sigma_m))
    assert np.allclose(np.diag(Sigma_m), 1.0)
    assert np.allclose(Sigma_m[-1, :-1], 0.0)

    assert np.isfinite(eps_m).all()
    assert np.isfinite(eps_g).all()
    var_eps_m = np.var(eps_m, axis=0)
    var_eps_g = np.var(eps_g, axis=0)
    assert np.all(np.abs(var_eps_m - 1.0) < 0.35)
    assert np.all(np.abs(var_eps_g - 1.0) < 0.35)

    assert np.allclose(Sigma_h, np.tril(Sigma_h))
    assert np.all(np.diag(Sigma_h) > 0.0)

    margin = equity_margin(
        setup["equity_params"].theta_m,
        setup["equity_params"].theta_g,
        setup["equity_params"].theta_g_Q,
        setup["equity_params"].mu_m_bar,
        setup["equity_params"].mu_g_u_bar,
        setup["equity_params"].mu_g_Q_u_bar,
        setup["equity_params"].Phi_m,
        setup["equity_params"].Phi_mg,
        setup["equity_params"].Phi_mh,
        setup["equity_params"].Phi_h,
        setup["equity_params"].M0_Q,
        setup["equity_params"].M1_Q,
        setup["equity_params"].Sigma_g,
        setup["equity_params"].Gamma0,
        setup["equity_params"].Gamma1,
        Sigma_hm,
        Sigma_hg,
        Sigma_h,
        setup["equity_params"].mu_h_bar,
        setup["equity_params"].Phi_g_Q,
    )
    assert margin < 0.0

    assert diagnostics["constraint_rejects"] >= 0
    assert diagnostics["constraint_used_fallback"] is False


def test_constraint_rejection_increments(monkeypatch):
    setup = _make_setup(seed=456)

    Sigma_m, eps_m = draw_sigma_m(setup["r_m"], setup["Dm"], setup["priors"])
    Sigma_gm, eps_g = draw_sigma_gm_and_eps_g(
        setup["r_g"],
        setup["Dg"],
        setup["Sigma_g"],
        eps_m,
        setup["priors"],
    )

    true_margin = equity_margin
    counter = {"calls": 0}

    def fake_margin(
        theta_m,
        theta_g,
        theta_g_Q,
        mu_m_bar,
        mu_g_u_bar,
        mu_g_Q_u_bar,
        Phi_m,
        Phi_mg,
        Phi_mh,
        Phi_h,
        M0_Q,
        M1_Q,
        Sigma_g,
        Gamma0,
        Gamma1,
        Sigma_hm,
        Sigma_hg,
        Sigma_h,
        mu_h_bar,
        Phi_g_Q,
    ):
        counter["calls"] += 1
        if counter["calls"] == 1:
            return 1.0
        return true_margin(
            theta_m,
            theta_g,
            theta_g_Q,
            mu_m_bar,
            mu_g_u_bar,
            mu_g_Q_u_bar,
            Phi_m,
            Phi_mg,
            Phi_mh,
            Phi_h,
            M0_Q,
            M1_Q,
            Sigma_g,
            Gamma0,
            Gamma1,
            Sigma_hm,
            Sigma_hg,
            Sigma_h,
            mu_h_bar,
            Phi_g_Q,
        )

    monkeypatch.setattr("tsm.covariance_blocks.equity_margin", fake_margin)

    Sigma_hm, Sigma_hg, Sigma_h, diagnostics = draw_sigma_h_block_with_constraint(
        setup["r_h"],
        eps_m,
        eps_g,
        setup["equity_params"],
        setup["priors"],
        max_rejects=100,
    )

    assert counter["calls"] >= 2
    assert diagnostics["constraint_rejects"] >= 1
    assert diagnostics["constraint_used_fallback"] is False

    margin = true_margin(
        setup["equity_params"].theta_m,
        setup["equity_params"].theta_g,
        setup["equity_params"].theta_g_Q,
        setup["equity_params"].mu_m_bar,
        setup["equity_params"].mu_g_u_bar,
        setup["equity_params"].mu_g_Q_u_bar,
        setup["equity_params"].Phi_m,
        setup["equity_params"].Phi_mg,
        setup["equity_params"].Phi_mh,
        setup["equity_params"].Phi_h,
        setup["equity_params"].M0_Q,
        setup["equity_params"].M1_Q,
        setup["equity_params"].Sigma_g,
        setup["equity_params"].Gamma0,
        setup["equity_params"].Gamma1,
        Sigma_hm,
        Sigma_hg,
        Sigma_h,
        setup["equity_params"].mu_h_bar,
        setup["equity_params"].Phi_g_Q,
    )
    assert margin < 0.0
