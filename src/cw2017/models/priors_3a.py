"""Log-priors for the block 3a parameterisation."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .parameters import Params3aUnconstrained

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def _normal_logpdf(x: jnp.ndarray, mean: float, std: float) -> jnp.ndarray:
    var = std ** 2
    return -0.5 * ((x - mean) ** 2 / var + jnp.log(var) + _LOG_2PI)


def logprior_3a(u: Params3aUnconstrained, cfg: Dict[str, float]) -> jnp.float64:
    """Return the log-prior density evaluated at ``u``."""

    sigma_log_omega = cfg.get("sigma_log_omega", 1.0)
    log_omega_prior = jnp.sum(_normal_logpdf(u.log_omega, -4.0, sigma_log_omega))

    sigma_L = cfg.get("sigma_L", 0.3)
    L_prior = jnp.sum(_normal_logpdf(u.Lg_free, 0.0, sigma_L))

    sigma_lambda = cfg.get("sigma_lambda", 0.5)
    eig_prior = jnp.sum(_normal_logpdf(u.u_eigs_head, 0.0, sigma_lambda))
    eig_prior += jnp.sum(_normal_logpdf(u.u_eigs_tail, 0.0, sigma_lambda))

    sigma_q = jnp.sqrt(cfg.get("sigma_q_sq", 6000.0))
    qs_prior = jnp.sum(_normal_logpdf(u.qs_free, 0.0, sigma_q))

    phi11_prior = _normal_logpdf(u.phiQ_toprow_raw[0], 0.9, 0.02)
    phi12_prior = _normal_logpdf(u.phiQ_toprow_raw[1], 0.0, 0.02)
    phi13mag_prior = _normal_logpdf(u.phiQ_toprow_raw[2], 0.0, 1.0)
    phi13sign_prior = _normal_logpdf(u.phiQ_sign_raw, 0.0, 1.0)

    gamma0_prior = _normal_logpdf(u.gamma0_last_raw, 0.0, 1.0)

    sigma_gamma1 = cfg.get("sigma_gamma1", 250.0)
    gamma1_prior = jnp.sum(_normal_logpdf(u.gamma1_free_2elts, 0.0, sigma_gamma1))

    return (
        log_omega_prior
        + L_prior
        + eig_prior
        + qs_prior
        + phi11_prior
        + phi12_prior
        + phi13mag_prior
        + phi13sign_prior
        + gamma0_prior
        + gamma1_prior
    )


__all__ = ["logprior_3a"]
