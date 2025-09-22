from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pg.pgas import VolParams, draw_h_pgas


def run_block3d_pg_as(
    data: Dict[str, np.ndarray],
    theta: Dict[str, np.ndarray],
    h_ref: np.ndarray,
    particles: int = 300,
    ess_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Draw ``h_{0:T}`` via PG-AS."""
    m = data["m"]
    g = data["g"]

    params = VolParams(
        d_m=theta["d_m"],
        d_g=theta["d_g"],
        d_h=theta["d_h"],
        mu_m=theta["mu_m"],
        mu_g=theta["mu_g"],
        Phi_m=theta["Phi_m"],
        Phi_mg=theta["Phi_mg"],
        Phi_mh=theta["Phi_mh"],
        Phi_gm=theta["Phi_gm"],
        Phi_g=theta["Phi_g"],
        Phi_gh=theta["Phi_gh"],
        Sigma_m=theta["Sigma_m"],
        Sigma_gm=theta["Sigma_gm"],
        Sigma_g=theta["Sigma_g"],
        Gamma0=theta["Gamma0"],
        Gamma1=theta["Gamma1"],
        mu_h=theta["mu_h"],
        Phi_h=theta["Phi_h"],
        Sigma_h=theta["Sigma_h"],
        Sigma_hm=theta["Sigma_hm"],
        Sigma_hg=theta["Sigma_hg"],
        h0_mean=theta["h0_mean"],
        h0_cov=theta["h0_cov"],
    )

    h_draw, diag = draw_h_pgas(
        m=m,
        g=g,
        params=params,
        h_ref=h_ref,
        J=particles,
        ess_threshold=ess_threshold,
        seed=seed,
        use_ancestor_sampling=True,
    )
    return {"h": h_draw, "diag": diag}
