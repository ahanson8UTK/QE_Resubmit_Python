"""Gibbs conditional log-densities and sampling utilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp

from ..kalman import smoother, sr_kf
from ..particle import pgibbs
from ..equity import constraints as equity_constraints
from ..utils.logging import WorkUnitLogger

work_logger = WorkUnitLogger()


def logpost_block1_params(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> jnp.float64:
    """Stub log posterior for block 3a."""

    work_logger.incr(kalman_evals=1)
    _ = sr_kf.sr_kf_loglik(params=fixed, h_t=data.get("h_t"), y_t=data.get("y_t"), m_t=data.get("m_t"))
    return jnp.array(0.0, dtype=jnp.float64)


def logpost_block2_q_eigs(theta_block: Any, fixed: Dict[str, Any], data: Dict[str, Any]) -> jnp.float64:
    """Stub log posterior for block 3b eigenvalues."""

    work_logger.incr(kalman_evals=1)
    return jnp.array(0.0, dtype=jnp.float64)


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
    "logpost_block1_params",
    "logpost_block2_q_eigs",
    "draw_block2b_g_and_means",
    "pgibbs_h",
    "conjugate_cov_updates",
]
