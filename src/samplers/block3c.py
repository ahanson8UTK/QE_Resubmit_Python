"""Sampling block for the ``g`` factors and static means.

Implements the Form I recursion from Appendix B.1.1 of Creal & Wu (2017,
*International Economic Review*) with the Q-measure rotations described in
Appendix A.2.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from equity_constraint import equity_coeffs
from kalman import dk_sample, kf_forward
from models.ss_builders_3c import build_steps_augmented, build_steps_reduced
from samplers.trunc_gauss import sample_halfspace_trunc_normal
from utils.linalg import chol_solve_spd, safe_cholesky


def _loglike_from_cache(cache: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute the Gaussian log-likelihood from Kalman innovations."""

    innov = jnp.asarray(cache["innov"], dtype=jnp.float64)
    innov_cov = jnp.asarray(cache["innov_cov"], dtype=jnp.float64)

    def _per_period(v: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        L = safe_cholesky(s)
        solve = chol_solve_spd(L, v)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        dim = v.shape[0]
        return -0.5 * (logdet + v @ solve + dim * jnp.log(2.0 * jnp.pi))

    return jnp.sum(jax.vmap(_per_period)(innov, innov_cov))


def run_block3c_dk_draw(
    key: jax.random.KeyArray,
    state: Dict,
    fixed: Dict,
    data: Dict,
    cfg: Dict,
    priors: Dict,
) -> Tuple[Dict, Dict]:
    """Draw ``g_{1:T}`` and the static means using the Form I sampler of Appendix B.1.1."""

    y_obs = jnp.asarray(data["y"], dtype=jnp.float64)

    key_mu, key_g = jax.random.split(key)

    steps_aug, x0_mean_aug, x0_cov_aug, idx = build_steps_augmented(
        fixed, data, cfg, priors, state
    )
    cache_aug = kf_forward(y_obs, steps_aug, x0_mean_aug, x0_cov_aug)

    filt_mean_T = cache_aug["filt_mean"][-1]
    filt_cov_T = cache_aug["filt_cov"][-1]

    mu_mean = filt_mean_T[idx["static_all"]]
    mu_cov = filt_cov_T[idx["static_all"], idx["static_all"]]
    mu_cov = 0.5 * (mu_cov + mu_cov.T)

    a_vec, b_val = equity_coeffs(fixed, cfg)
    mu_draw = sample_halfspace_trunc_normal(key_mu, mu_mean, mu_cov, a_vec, b_val)

    d_m = int(cfg["dims"]["d_m"])
    mu_m = mu_draw[:d_m]
    mu_gu = mu_draw[d_m : d_m + 2]
    mu_gq = mu_draw[d_m + 2 : d_m + 4]

    steps_red, x0_mean_red, x0_cov_red = build_steps_reduced(
        fixed,
        data,
        cfg,
        mus={"mu_m": mu_m, "mu_gu": mu_gu, "mu_gQ_u": mu_gq},
    )
    cache_red = kf_forward(y_obs, steps_red, x0_mean_red, x0_cov_red)
    g_draw = dk_sample(key_g, y_obs, steps_red, x0_mean_red, x0_cov_red, cache_red)

    loglik = _loglike_from_cache(cache_red)
    max_abs_innov = jnp.max(jnp.abs(cache_red["innov"]))
    slack = a_vec @ mu_draw + b_val

    new_state = {
        "g_path": g_draw,
        "mu_m": mu_m,
        "mu_g_u": mu_gu,
        "mu_gQ_u": mu_gq,
    }
    diags = {
        "loglik": loglik,
        "max_abs_innov": max_abs_innov,
        "constraint_slack": slack,
    }
    return new_state, diags


__all__ = ["run_block3c_dk_draw"]
