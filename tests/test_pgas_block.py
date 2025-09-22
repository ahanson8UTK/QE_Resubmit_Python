import numpy as np

from pg.pgas import VolParams, draw_h_pgas


def test_pgas_shapes():
    T, d_m, d_g, d_h = 20, 5, 3, 2
    rng = np.random.default_rng(0)
    m = rng.normal(size=(T + 1, d_m))
    g = rng.normal(size=(T + 1, d_g))

    theta = dict(
        d_m=d_m,
        d_g=d_g,
        d_h=d_h,
        mu_m=np.zeros(d_m),
        mu_g=np.zeros(d_g),
        Phi_m=0.95 * np.eye(d_m),
        Phi_mg=np.zeros((d_m, d_g)),
        Phi_mh=np.zeros((d_m, d_h)),
        Phi_gm=np.zeros((d_g, d_m)),
        Phi_g=0.95 * np.eye(d_g),
        Phi_gh=np.zeros((d_g, d_h)),
        Sigma_m=np.eye(d_m),
        Sigma_gm=np.zeros((d_g, d_m)),
        Sigma_g=np.eye(d_g),
        Gamma0=np.zeros(d_m + d_g),
        Gamma1=np.zeros((d_m + d_g, d_h)),
        mu_h=np.zeros(d_h),
        Phi_h=0.95 * np.eye(d_h),
        Sigma_h=0.1 * np.eye(d_h),
        Sigma_hm=np.zeros((d_h, d_m)),
        Sigma_hg=np.zeros((d_h, d_g)),
        h0_mean=np.zeros(d_h),
        h0_cov=np.eye(d_h),
    )
    params = VolParams(**theta)
    params.precompute()
    h_ref = rng.normal(size=(T + 1, d_h))
    h_draw, diag = draw_h_pgas(m, g, params, h_ref, J=64, seed=1)

    assert h_draw.shape == (T + 1, d_h)
    assert diag.ess.shape == (T,)
