"""Gibbs conditional log-densities and sampling utilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn

from ..kalman import smoother, sr_kf
from ..particle import pgibbs
from ..equity import constraints as equity_constraints
from ..utils.logging import WorkUnitLogger

work_logger = WorkUnitLogger()


def _tile_theta(theta: jnp.ndarray, dim: int) -> jnp.ndarray:
    reps = (dim + theta.size - 1) // theta.size
    tiled = jnp.tile(theta, reps)
    return tiled[:dim]


def _stick_breaking_descending(raw: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Map unconstrained reals to a positive, strictly decreasing vector."""

    if raw.size == 0:
        return jnp.asarray([], dtype=jnp.float64)

    raw = jnp.asarray(raw, dtype=jnp.float64)
    radius = jnp.asarray(radius, dtype=jnp.float64)
    eps = jnp.array(1e-6, dtype=jnp.float64)

    positive_steps = jnn.softplus(raw) + eps
    total = jnp.sum(positive_steps)
    scaled = positive_steps / (total + eps)
    cumulative = jnp.cumsum(scaled)

    margin = radius * jnp.array(1e-3, dtype=jnp.float64)
    available = radius - margin
    ordered = available * (1.0 - cumulative) + margin
    return ordered


def _map_q_measure_eigs(raw: jnp.ndarray, rho_max: float) -> jnp.ndarray:
    """Transform unconstrained eigenvalues onto the stability region."""

    raw = jnp.asarray(raw, dtype=jnp.float64)
    rho = jnp.asarray(rho_max, dtype=jnp.float64)
    if raw.ndim != 1:
        raise ValueError("Eigenvalue parameter must be a one-dimensional vector")

    leading = rho * jnp.tanh(raw[0])
    if raw.size == 1:
        return jnp.atleast_1d(leading)

    tail = _stick_breaking_descending(raw[1:], rho)
    return jnp.concatenate([jnp.atleast_1d(leading), tail], axis=0)


def transform_q_measure_eigs(
    raw: jnp.ndarray,
    rho_max: float,
) -> Tuple[jnp.ndarray, jnp.float64]:
    """Return Q-measure eigenvalues and log-Jacobian of the restriction map."""

    map_fn = lambda x: _map_q_measure_eigs(x, rho_max)
    eigs = map_fn(raw)
    jacobian = jax.jacfwd(map_fn)(raw)
    sign, logabsdet = jnp.linalg.slogdet(jacobian)
    logabsdet = jnp.where(sign == 0, -jnp.inf, logabsdet)
    return eigs, logabsdet


def logpost_block3a_theta(
    theta_block: jnp.ndarray,
    fixed: Dict[str, Any],
    data: Dict[str, Any],
) -> jnp.float64:
    """Minimal working log-posterior for block 3a parameters."""

    theta_block = jnp.asarray(theta_block, dtype=jnp.float64)
    h_t = fixed["h_t"]
    y_t = data["y_t"]
    m_t = data["m_t"]

    params_template = data["params"]
    params = dict(params_template)
    params["theta"] = theta_block

    def build_HR(t, params_obj, _h, _m):
        theta = jnp.asarray(params_obj["theta"], dtype=jnp.float64)
        obs_dim = y_t.shape[1]
        state_dim = params_obj["a0"].shape[0]
        scales = 0.5 + 0.5 * jnp.tanh(_tile_theta(theta, obs_dim))
        eye = jnp.eye(state_dim, dtype=jnp.float64)
        H = eye[:obs_dim, :] * scales[:, None]
        R_diag = 0.1 + 0.05 * scales
        R_chol = jnp.diag(R_diag)
        return H, R_chol

    def build_FQ(t, params_obj, _h, _m):
        theta = jnp.asarray(params_obj["theta"], dtype=jnp.float64)
        state_dim = params_obj["a0"].shape[0]
        scales = 0.8 + 0.2 * jnp.tanh(_tile_theta(theta[::-1], state_dim))
        F = jnp.eye(state_dim, dtype=jnp.float64) * scales
        Q_diag = 0.05 + 0.02 * scales
        Q_chol = jnp.diag(Q_diag)
        return F, Q_chol

    params["fns"] = {"build_HR": build_HR, "build_FQ": build_FQ}

    log_prior = -0.5 * jnp.sum((theta_block / 3.0) ** 2)
    log_prior -= theta_block.size * 0.5 * jnp.log(2.0 * jnp.pi * 9.0)

    loglik, _ = sr_kf.sr_kf_loglik(params=params, h_t=h_t, y_t=y_t, m_t=m_t)
    return log_prior + loglik


def make_logpost_block3a_batched(
    fixed: Dict[str, Any],
    data: Dict[str, Any],
) -> Any:
    """Return a chain-batched logdensity for block 3a."""

    single = lambda theta: logpost_block3a_theta(theta, fixed, data)
    return jax.vmap(single)


def logpost_block1_params(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> jnp.float64:
    """Stub log posterior for block 3a."""

    work_logger.incr(kalman_evals=1)
    _ = sr_kf.sr_kf_loglik(params=fixed, h_t=data.get("h_t"), y_t=data.get("y_t"), m_t=data.get("m_t"))
    return jnp.array(0.0, dtype=jnp.float64)


def logpost_block2_q_eigs(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> jnp.float64:
    """Log posterior for the Q-measure eigenvalues in block 3b."""

    rho_max = float(data.get("rho_max", 0.995))
    theta_block = jnp.asarray(theta_block, dtype=jnp.float64)
    eigs, log_jac = transform_q_measure_eigs(theta_block, rho_max)
    log_jac = jnp.where(jnp.isfinite(log_jac), log_jac, -jnp.inf)

    params_template = data["params"]
    params = dict(params_template)
    params["lambda_q"] = eigs

    theta_fixed = jnp.asarray(params_template.get("theta", jnp.zeros_like(eigs)), dtype=jnp.float64)
    y_t = jnp.asarray(data["y_t"], dtype=jnp.float64)
    m_t = jnp.asarray(data["m_t"], dtype=jnp.float64)
    h_t = jnp.asarray(fixed["h_t"], dtype=jnp.float64)

    def build_HR(t, params_obj, _h, _m):
        obs_dim = y_t.shape[1]
        state_dim = params_obj["a0"].shape[0]
        scales = 0.5 + 0.5 * jnp.tanh(_tile_theta(theta_fixed, obs_dim))
        H = jnp.eye(state_dim, dtype=jnp.float64)[:obs_dim, :]
        H = H * scales[:, None]
        R_diag = 0.08 + 0.04 * scales
        return H, jnp.diag(R_diag)

    def build_FQ(t, params_obj, _h, _m):
        lam = jnp.asarray(params_obj["lambda_q"], dtype=jnp.float64)
        state_dim = lam.size
        F = jnp.eye(state_dim, dtype=jnp.float64) * lam
        innovations = 0.05 + 0.02 * (1.0 - lam**2)
        Q_diag = jnp.clip(innovations, a_min=1e-6)
        return F, jnp.diag(jnp.sqrt(Q_diag))

    params["fns"] = {"build_HR": build_HR, "build_FQ": build_FQ}

    prior_scale = float(data.get("prior_scale_q", 0.25))
    log_prior = -0.5 * jnp.sum((eigs / prior_scale) ** 2)
    log_prior -= eigs.size * 0.5 * jnp.log(2.0 * jnp.pi * prior_scale**2)

    work_logger.incr(kalman_evals=1)
    loglik, _ = sr_kf.sr_kf_loglik(params=params, h_t=h_t, y_t=y_t, m_t=m_t)
    logpost = log_prior + log_jac + loglik
    return jnp.where(jnp.isfinite(logpost), logpost, -jnp.inf)


def draw_block2b_g_and_means(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw g and means using the Durbin–Koopman smoother and apply equity constraints."""

    work_logger.incr(kalman_evals=1)
    g_sample, means = smoother.dk_simulation_smoother(
        params=fixed,
        h_t=data.get("h_t"),
        y_t=data.get("y_t"),
        m_t=data.get("m_t"),
        key=data.get("key"),
    )
    constrained_means = equity_constraints.solve_linear_equity_constraint(theta_block, means)
    return g_sample, constrained_means


def pgibbs_h(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> jnp.ndarray:
    """Particle Gibbs update for the latent ``h`` block."""

    work_logger.incr(particle_passes=1)
    return pgibbs.pgibbs_sample_h(
        params=fixed,
        g_t=data.get("g_t"),
        y_t=data.get("y_t"),
        key=data.get("key"),
        num_particles=int(data.get("num_particles", 8)),
    )


def conjugate_cov_updates(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for conjugate covariance draws respecting Σ_m restrictions."""

    work_logger.incr(kalman_evals=1)
    return {"sigma_m": jnp.zeros((2, 2), dtype=jnp.float64)}


__all__ = [
    "work_logger",
    "logpost_block3a_theta",
    "make_logpost_block3a_batched",
    "logpost_block1_params",
    "logpost_block2_q_eigs",
    "transform_q_measure_eigs",
    "draw_block2b_g_and_means",
    "pgibbs_h",
    "conjugate_cov_updates",
]
