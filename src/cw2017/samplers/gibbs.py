"""Minimal Gibbs driver with a smoke-test 3a block update."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..config import AppConfig
from ..models.conditionals import logpost_block3a_theta
from ..reporting import save_results, summarize
from ..samplers.hmc_block import adapt_block_chees, hmc_block_step
from ..utils.logging import setup_logging

BLOCK_SEQUENCE = ["3a", "3b", "3c", "3d", "3e", "4", "5", "6"]


@dataclass
class SmokeState:
    """State container for the smoke-test Gibbs sweep."""

    theta_block3a: jnp.ndarray
    hmc_state_3a: Optional[Any] = None
    tuned_params_3a: Optional[Dict[str, Any]] = None
    diagnostics_3a: Optional[Dict[str, Any]] = None
    sweep: int = 0


def run_one_sweep_smoke(
    rng_key: jax.Array,
    state: SmokeState,
    cfg: Dict[str, Any],
    data: Dict[str, Any],
) -> Tuple[jax.Array, SmokeState, Dict[str, float]]:
    """Execute a single Block 3a HMC update and record diagnostics."""

    fixed = data["fixed"]
    single_logdensity = lambda theta: logpost_block3a_theta(theta, fixed, data)

    metrics: Dict[str, float] = {"block": "3a"}
    start_time = time.time()

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
        metrics["warmup_acceptance_hmean"] = float(diag["acceptance_hmean"])
        metrics["warmup_ebfmi"] = float(diag["ebfmi"])
        metrics["step_size"] = float(diag["final_step_size"])
        metrics["trajectory_length"] = float(diag["final_trajectory_length"])

    assert state.tuned_params_3a is not None and state.hmc_state_3a is not None

    sample_key, rng_key = jax.random.split(rng_key)
    new_state, info = hmc_block_step(sample_key, state.hmc_state_3a, state.tuned_params_3a)
    theta_samples = new_state.position
    state = replace(
        state,
        theta_block3a=theta_samples,
        hmc_state_3a=new_state,
        sweep=state.sweep + 1,
    )

    accept = jnp.asarray(info["acceptance"], dtype=jnp.float64)
    metrics["acceptance"] = float(jnp.mean(accept))
    metrics["duration_sec"] = float(time.time() - start_time)
    metrics["step_size"] = float(state.tuned_params_3a["step_size"])
    metrics.setdefault("warmup_ebfmi", float("nan"))
    metrics.setdefault("warmup_acceptance_hmean", float("nan"))
    metrics["trajectory_length"] = float(
        state.diagnostics_3a.get("final_trajectory_length")
        if state.diagnostics_3a
        else float("nan")
    )

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
        rng_key, state, record = run_one_sweep_smoke(rng_key, state, cfg, data)
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
