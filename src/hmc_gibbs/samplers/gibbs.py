"""Minimal Gibbs driver with a smoke-test 3a block update."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..config import AppConfig
from ..models.conditionals import logpost_block3a_theta, logpost_block2_q_eigs, transform_q_measure_eigs
from ..reporting import save_results, summarize
from ..samplers.hmc_block import adapt_block_chees, hmc_block_step
from ..utils.logging import setup_logging

BLOCK_SEQUENCE = ["3a", "3b", "3c", "3d", "3e", "4", "5", "6"]


@dataclass
class SmokeState:
    """State container for the smoke-test Gibbs sweep."""

    theta_block3a: jnp.ndarray
    theta_block3b: jnp.ndarray
    hmc_state_3a: Optional[Any] = None
    tuned_params_3a: Optional[Dict[str, Any]] = None
    diagnostics_3a: Optional[Dict[str, Any]] = None
    hmc_state_3b: Optional[Any] = None
    tuned_params_3b: Optional[Dict[str, Any]] = None
    diagnostics_3b: Optional[Dict[str, Any]] = None
    reject_streak_3b: int = 0
    sweep: int = 0


def _regularize_mass_matrix(params: Dict[str, Any], regularization: float) -> Dict[str, Any]:
    """Return a copy of ``params`` with a heavily regularised mass matrix."""

    if "inverse_mass_matrix" not in params:
        return params

    inv_mass = jnp.asarray(params["inverse_mass_matrix"], dtype=jnp.float64)
    reg = jnp.asarray(regularization, dtype=jnp.float64)
    if inv_mass.ndim == 1:
        inv_mass = inv_mass + reg
    else:
        inv_mass = inv_mass + reg * jnp.eye(inv_mass.shape[0], dtype=inv_mass.dtype)

    new_params = dict(params)
    new_params["inverse_mass_matrix"] = inv_mass
    return new_params


def run_one_sweep_smoke(
    rng_key: jax.Array,
    state: SmokeState,
    cfg: Dict[str, Any],
    data: Dict[str, Any],
) -> Tuple[jax.Array, SmokeState, List[Dict[str, float]]]:
    """Execute Block 3a and Block 3b HMC updates and record diagnostics."""

    fixed = data["fixed"]
    single_logdensity = lambda theta: logpost_block3a_theta(theta, fixed, data)

    metrics: List[Dict[str, float]] = []
    start_time = time.time()
    metrics_3a: Dict[str, float] = {"block": "3a"}

    info: Dict[str, Any]
    if state.tuned_params_3a is None or state.hmc_state_3a is None:
        warmup_key, rng_key = jax.random.split(rng_key)
        last_states, tuned_params, diag = adapt_block_chees(
            warmup_key,
            state.theta_block3a,
            single_logdensity,
            cfg.get("warmup", {}),
        )
        state = replace(
            state,
            hmc_state_3a=last_states,
            tuned_params_3a=tuned_params,
            diagnostics_3a=diag,
            theta_block3a=last_states.position,
        )
        metrics_3a["warmup_acceptance_hmean"] = float(diag["acceptance_hmean"])
        metrics_3a["warmup_ebfmi"] = float(diag["ebfmi"])
        metrics_3a["step_size"] = float(diag["final_step_size"])
        metrics_3a["trajectory_length"] = float(diag["final_trajectory_length"])
        info = {"acceptance": jnp.ones_like(state.theta_block3a[..., 0])}
    else:
        assert state.tuned_params_3a is not None and state.hmc_state_3a is not None
        sample_key, rng_key = jax.random.split(rng_key)
        new_state, info = hmc_block_step(sample_key, state.hmc_state_3a, state.tuned_params_3a)
        state = replace(
            state,
            theta_block3a=new_state.position,
            hmc_state_3a=new_state,
        )

    accept = jnp.asarray(info["acceptance"], dtype=jnp.float64)
    metrics_3a["acceptance"] = float(jnp.mean(accept))
    metrics_3a["duration_sec"] = float(time.time() - start_time)
    metrics_3a["step_size"] = float(state.tuned_params_3a["step_size"])
    metrics_3a.setdefault("warmup_ebfmi", float("nan"))
    metrics_3a.setdefault("warmup_acceptance_hmean", float("nan"))
    metrics_3a["trajectory_length"] = float(
        state.diagnostics_3a.get("final_trajectory_length")
        if state.diagnostics_3a
        else float("nan")
    )

    metrics.append(metrics_3a)

    # ----- Block 3b -----
    metrics_3b: Dict[str, float] = {"block": "3b"}
    start_time = time.time()
    logdensity3b = lambda lam: logpost_block2_q_eigs(lam, fixed, data)
    warm_cfg_3b = cfg.get("block3b", {}).get("warmup", cfg.get("warmup", {}))
    mass_reg = float(cfg.get("block3b", {}).get("mass_regularization", 5.0))
    jitter_scale = float(cfg.get("block3b", {}).get("step_size_jitter", 0.1))
    boundary_thresh = float(cfg.get("block3b", {}).get("boundary_threshold", 0.98))
    traj_shrink = float(cfg.get("block3b", {}).get("trajectory_shrink", 0.5))
    reject_streak_cap = int(cfg.get("block3b", {}).get("max_reject_streak", 3))
    reject_shrink = float(cfg.get("block3b", {}).get("reject_step_size_shrink", 0.8))
    rho_max = float(data.get("rho_max", 0.995))

    if state.tuned_params_3b is None or state.hmc_state_3b is None:
        warmup_key, rng_key = jax.random.split(rng_key)
        last_states, tuned_params, diag = adapt_block_chees(
            warmup_key,
            state.theta_block3b,
            logdensity3b,
            warm_cfg_3b,
        )
        tuned_params = _regularize_mass_matrix(dict(tuned_params), mass_reg)
        state = replace(
            state,
            hmc_state_3b=last_states,
            tuned_params_3b=tuned_params,
            diagnostics_3b=diag,
            theta_block3b=last_states.position,
            reject_streak_3b=0,
        )
        metrics_3b["warmup_acceptance_hmean"] = float(diag["acceptance_hmean"])
        metrics_3b["warmup_ebfmi"] = float(diag["ebfmi"])
        metrics_3b["step_size"] = float(diag["final_step_size"])
        metrics_3b["trajectory_length"] = float(diag["final_trajectory_length"])
        info_b = {"acceptance": jnp.ones_like(state.theta_block3b[..., 0])}
    else:
        sample_key, rng_key = jax.random.split(rng_key)
        jitter_key, sample_key = jax.random.split(sample_key)
        tuned_params = dict(state.tuned_params_3b)
        step_size = jnp.asarray(tuned_params["step_size"], dtype=jnp.float64)
        if jitter_scale > 0.0:
            jitter = 1.0 + jitter_scale * (jax.random.uniform(jitter_key, step_size.shape) - 0.5) * 2.0
            tuned_params["step_size"] = step_size * jitter

        eigen_fn = lambda raw: transform_q_measure_eigs(raw, rho_max)[0]
        eigvals = jax.vmap(eigen_fn)(state.theta_block3b)
        max_abs = jnp.max(jnp.abs(eigvals))
        if max_abs >= boundary_thresh * rho_max:
            traj = jnp.asarray(tuned_params.get("trajectory_length", 1.0), dtype=jnp.float64)
            tuned_params["trajectory_length"] = traj * traj_shrink

        new_state, info_b = hmc_block_step(sample_key, state.hmc_state_3b, tuned_params)
        accept_b = jnp.asarray(info_b["acceptance"], dtype=jnp.float64)
        mean_accept = float(jnp.mean(accept_b))
        streak = state.reject_streak_3b + 1 if mean_accept < 0.2 else 0
        tuned_store = dict(state.tuned_params_3b)
        if streak >= reject_streak_cap:
            tuned_store["step_size"] = jnp.asarray(tuned_store["step_size"]) * reject_shrink
            streak = 0
        state = replace(
            state,
            theta_block3b=new_state.position,
            hmc_state_3b=new_state,
            tuned_params_3b=tuned_store,
            reject_streak_3b=streak,
        )

    if "acceptance" in info_b:
        acceptances = jnp.asarray(info_b["acceptance"], dtype=jnp.float64)
        metrics_3b["acceptance"] = float(jnp.mean(acceptances))
    else:
        metrics_3b["acceptance"] = 1.0

    metrics_3b.setdefault("step_size", float(state.tuned_params_3b["step_size"]))
    metrics_3b.setdefault(
        "trajectory_length",
        float(
            state.diagnostics_3b.get("final_trajectory_length")
            if state.diagnostics_3b
            else float("nan")
        ),
    )
    metrics_3b.setdefault("warmup_ebfmi", float("nan"))
    metrics_3b.setdefault("warmup_acceptance_hmean", float("nan"))
    metrics_3b["duration_sec"] = float(time.time() - start_time)

    state = replace(state, sweep=state.sweep + 1)
    metrics.append(metrics_3b)

    return rng_key, state, metrics


def run_smoke_sweeps(
    rng_key: jax.Array,
    state: SmokeState,
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    num_sweeps: int,
) -> Tuple[jax.Array, SmokeState, List[Dict[str, float]]]:
    """Run ``num_sweeps`` of the smoke Gibbs sampler."""

    metrics: List[Dict[str, float]] = []
    for _ in range(num_sweeps):
        rng_key, state, block_records = run_one_sweep_smoke(rng_key, state, cfg, data)
        for record in block_records:
            record["sweep"] = state.sweep
            metrics.append(record)
    return rng_key, state, metrics


def run_gibbs(rng_key: jax.Array, config: AppConfig, results_root: Path) -> Dict[str, object]:
    """Execute a placeholder Gibbs sweep (legacy entry point)."""

    setup_logging(level=config.logging.level, rich_tracebacks=config.logging.rich_tracebacks)
    run_id = save_results.build_run_id(config.run.run_id_prefix)
    summary_dir = save_results.prepare_run_directory(results_root, run_id)

    block_metrics: List[Dict[str, float]] = []
    for block in BLOCK_SEQUENCE:
        block_metrics.append(
            {
                "block": block,
                "acceptance": 1.0,
                "duration_sec": 0.0,
            }
        )

    summarize.write_summary(summary_dir, block_metrics)
    save_results.write_metadata(
        summary_dir,
        {
            "run_id": run_id,
            "chains": config.run.chains,
            "seed": config.run.seed,
        },
    )
    return {"run_id": run_id, "summary": block_metrics}


__all__ = [
    "run_gibbs",
    "BLOCK_SEQUENCE",
    "SmokeState",
    "run_one_sweep_smoke",
    "run_smoke_sweeps",
]
