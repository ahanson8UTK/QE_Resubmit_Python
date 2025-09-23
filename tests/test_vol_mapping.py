import numpy as np

from common.vol_mapping import diag_vols_series


def test_dividend_row_is_lagged():
    rng = np.random.default_rng(0)
    T, d_m, d_g, d_h = 5, 5, 3, 7
    e_div_ix = d_m - 1

    Gamma0 = rng.normal(size=d_m + d_g)
    Gamma1 = rng.normal(size=(d_m + d_g, d_h))
    H = rng.normal(size=(T, d_h))
    mu_h = rng.normal(size=(d_h,))

    Dm, Dg = diag_vols_series(Gamma0, Gamma1, H, d_m, d_g, e_div_ix, True, mu_h)

    H_prev = np.vstack([mu_h[None, :], H[:-1, :]])
    z_last = Gamma0[e_div_ix] + H_prev @ Gamma1[e_div_ix].T
    manual = np.exp(0.5 * z_last)

    np.testing.assert_allclose(Dm[:, e_div_ix], manual)

    z_all = Gamma0[None, :] + H @ Gamma1.T
    baseline_dm = np.exp(0.5 * z_all[:, :d_m])
    baseline_dg = np.exp(0.5 * z_all[:, d_m:])
    keep_mask = np.ones(d_m, dtype=bool)
    keep_mask[e_div_ix] = False
    np.testing.assert_allclose(Dm[:, keep_mask], baseline_dm[:, keep_mask])
    np.testing.assert_allclose(Dg, baseline_dg)


def test_no_lag_matches_baseline():
    rng = np.random.default_rng(1)
    T, d_m, d_g, d_h = 4, 3, 2, 5
    e_div_ix = 1

    Gamma0 = rng.normal(size=d_m + d_g)
    Gamma1 = rng.normal(size=(d_m + d_g, d_h))
    H = rng.normal(size=(T, d_h))

    Dm, Dg = diag_vols_series(
        Gamma0,
        Gamma1,
        H,
        d_m,
        d_g,
        e_div_ix,
        use_lag_for_div=False,
    )

    z_all = Gamma0[None, :] + H @ Gamma1.T
    baseline_dm = np.exp(0.5 * z_all[:, :d_m])
    baseline_dg = np.exp(0.5 * z_all[:, d_m:])

    np.testing.assert_allclose(Dm, baseline_dm)
    np.testing.assert_allclose(Dg, baseline_dg)
