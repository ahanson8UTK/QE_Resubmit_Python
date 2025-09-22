"""Log-density construction for block 3a."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import jax.numpy as jnp

from ..equity.atsm_measurement import build_measurement_terms
from ..kalman.sr_kf import cholesky_psd, compute_dg_diag_sq, sr_kf_loglik
from ..math.fill_q import count_qs_entries
from .parameters import Params3aUnconstrained, constrain_params3a
from .priors_3a import logprior_3a

Array = jnp.ndarray


def _asarray(x: Array) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def unpack_params3a_unconstrained(
    vec: Array,
    info: Dict[str, int],
    maturities: Iterable[int],
) -> Params3aUnconstrained:
    vec = _asarray(vec)
    dy = len(list(maturities))
    Nm = int(info["Nm"])
    Ng = int(info["Ng"])
    Nh = int(info["Nh"])
    qs_len = count_qs_entries(info)

    idx = 0
    log_omega = vec[idx : idx + dy]
    idx += dy

    Lg_free = vec[idx : idx + 3]
    idx += 3

    u_eigs_head = vec[idx : idx + Nm + Ng]
    idx += Nm + Ng

    u_eigs_tail = vec[idx : idx + Nh]
    idx += Nh

    qs_free = vec[idx : idx + qs_len]
    idx += qs_len

    phiQ_toprow_raw = vec[idx : idx + 3]
    idx += 3

    phiQ_sign_raw = vec[idx]
    idx += 1

    gamma0_last_raw = vec[idx]
    idx += 1

    gamma1_free_2elts = vec[idx : idx + 2]
    idx += 2

    if idx != vec.size:
        raise ValueError("Unexpected vector length for Params3aUnconstrained")

    return Params3aUnconstrained(
        log_omega=log_omega,
        Lg_free=Lg_free,
        u_eigs_head=u_eigs_head,
        u_eigs_tail=u_eigs_tail,
        qs_free=qs_free,
        phiQ_toprow_raw=phiQ_toprow_raw,
        phiQ_sign_raw=phiQ_sign_raw,
        gamma0_last_raw=gamma0_last_raw,
        gamma1_free_2elts=gamma1_free_2elts,
    )


def pack_params3a_unconstrained(
    u: Params3aUnconstrained,
    info: Dict[str, int],
    maturities: Iterable[int],
) -> Array:
    parts = [
        _asarray(u.log_omega).reshape(-1),
        _asarray(u.Lg_free).reshape(-1),
        _asarray(u.u_eigs_head).reshape(-1),
        _asarray(u.u_eigs_tail).reshape(-1),
        _asarray(u.qs_free).reshape(-1),
        _asarray(u.phiQ_toprow_raw).reshape(-1),
        jnp.atleast_1d(_asarray(u.phiQ_sign_raw)),
        jnp.atleast_1d(_asarray(u.gamma0_last_raw)),
        _asarray(u.gamma1_free_2elts).reshape(-1),
    ]
    return jnp.concatenate(parts, axis=0)


def make_logdensity_block3a(
    fixed: Dict[str, Array],
    data: Dict[str, Array],
    cfg: Dict[str, float],
    info: Dict[str, int],
    maturities: Iterable[int],
) -> Callable[[Array], jnp.float64]:
    y_t = _asarray(data["y_t"])
    m_t = _asarray(fixed["m_t"])
    h_t = _asarray(fixed["h_t"])

    def logdensity(u_vec: Array) -> jnp.float64:
        u = unpack_params3a_unconstrained(u_vec, info, maturities)
        params3a, log_det_jac = constrain_params3a(u, cfg, info)

        measurement_terms = build_measurement_terms(params3a, fixed, maturities)
        A0, A1, B, M0Q, M1Q = measurement_terms

        R_chol = jnp.diag(jnp.sqrt(params3a.Omega_diag))
        B_mat = _asarray(B)

        mu_g = _asarray(fixed["mu_g"])
        Ng = int(info["Ng"])
        Gamma0_g = params3a.Gamma0[-Ng:]
        Gamma1_g = params3a.Gamma1[-Ng:, :]
        Sigma_g = params3a.Sigma_g
        Phi_blocks = params3a.PhiP_blocks

        def build_HR_fn(_, __):
            return B_mat, R_chol

        def build_FQ_fn(_, __, m_curr, h_curr, _t_idx):
            diag_sq = compute_dg_diag_sq(Gamma0_g, Gamma1_g, h_curr)
            Sigma_scaled = Sigma_g * diag_sq[None, :]
            cov = Sigma_scaled @ Sigma_g.T
            Q_chol = cholesky_psd(cov)
            a_const = mu_g + Phi_blocks.Phi_gm @ m_curr + Phi_blocks.Phi_gh @ h_curr
            return Phi_blocks.Phi_gg, Q_chol, a_const

        loglik, _ = sr_kf_loglik(
            params3a,
            fixed,
            y_t,
            m_t,
            h_t,
            measurement_terms,
            build_HR_fn,
            build_FQ_fn,
        )
        log_prior = logprior_3a(u, cfg.get("priors_3a", {}))
        return log_prior + log_det_jac + loglik

    return logdensity


__all__ = [
    "make_logdensity_block3a",
    "pack_params3a_unconstrained",
    "unpack_params3a_unconstrained",
]
