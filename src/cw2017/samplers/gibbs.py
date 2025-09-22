"""Minimal Gibbs-style wrapper for the block 3a HMC update."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp

from ..models.conditionals import make_logdensity_block3a
from ..math.fill_q import count_qs_entries
from ..samplers.hmc_block import adapt_block_chees, hmc_block_step

Array = jnp.ndarray


@dataclass
class Block3aSamplerState:
    """Container tracking kernels, positions and diagnostics for block 3a."""

    positions: Dict[str, Array] = field(default_factory=dict)
    kernels: Dict[str, Any] = field(default_factory=dict)
    tuned_params: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _param_dimension(info: Dict[str, int], maturities: Iterable[int]) -> int:
    dy = len(list(maturities))
    Nm = int(info["Nm"])
    Ng = int(info["Ng"])
    Nh = int(info["Nh"])
    qs_len = count_qs_entries(info)
    return dy + 3 + (Nm + Ng) + Nh + qs_len + 3 + 1 + 1 + 2


def run_block3a_hmc_step(
    rng_key: jax.Array,
    state: Block3aSamplerState,
    fixed: Dict[str, Any],
    data: Dict[str, Any],
    cfg: Dict[str, Any],
    warmup_cfg: Dict[str, Any],
    info: Dict[str, int],
    maturities: Iterable[int],
) -> Tuple[jax.Array, Block3aSamplerState, Dict[str, float]]:
    """Advance the block 3a parameters using an HMC step."""

    logdensity = make_logdensity_block3a(fixed, data, cfg, info, maturities)
    dim = _param_dimension(info, maturities)

    positions = state.positions.get("3a")
    if positions is None:
        num_chains = int(cfg.get("num_chains", 2))
        positions = jnp.zeros((num_chains, dim), dtype=jnp.float64)
        state.positions["3a"] = positions

    kernel_state = state.kernels.get("3a")
    tuned_params = state.tuned_params.get("3a")
    diagnostics = state.diagnostics.get("3a")
    metrics: Dict[str, float] = {}

    if tuned_params is None or kernel_state is None:
        warm_key, rng_key = jax.random.split(rng_key)
        warm_cfg = dict(warmup_cfg or {})
        warm_cfg.setdefault("target_accept", 0.651)
        kernel_state, tuned_params, warm_diag = adapt_block_chees(
            warm_key,
            positions,
            logdensity,
            warm_cfg,
        )
        state.positions["3a"] = kernel_state.position
        state.kernels["3a"] = kernel_state
        state.tuned_params["3a"] = tuned_params
        state.diagnostics["3a"] = warm_diag
        metrics.update(
            step_type="warmup",
            acceptance_hmean=float(warm_diag["acceptance_hmean"]),
            ebfmi=float(warm_diag["ebfmi"]),
            step_size=float(warm_diag["final_step_size"]),
            trajectory_length=float(warm_diag["final_trajectory_length"]),
        )
    else:
        sample_key, rng_key = jax.random.split(rng_key)
        new_state, info_dict = hmc_block_step(sample_key, kernel_state, tuned_params)
        state.positions["3a"] = new_state.position
        state.kernels["3a"] = new_state
        state.diagnostics["3a"] = {"last_info": info_dict}
        acceptance = jnp.asarray(info_dict["acceptance"], dtype=jnp.float64)
        metrics.update(
            step_type="sample",
            acceptance=float(jnp.mean(acceptance)),
        )

    return rng_key, state, metrics


__all__ = ["Block3aSamplerState", "run_block3a_hmc_step"]
