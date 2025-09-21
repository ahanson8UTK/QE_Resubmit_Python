"""Block-wise warm-up scheduling for the Gibbs sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import jax

from . import hmc_block
from ..typing import Array, PRNGKey


@dataclass
class StabilityThresholds:
    acceptance_lower: float
    acceptance_upper: float
    log_step_tolerance: float
    log_traj_tolerance: float
    ebfmi_min: float
    divergence_rate_max: float


@dataclass
class BlockWarmupConfig:
    sweeps: int
    inner_transitions: int
    target_acceptance: float
    max_rewarm_sweeps: int
    thresholds: StabilityThresholds

    @property
    def num_steps(self) -> int:
        return self.inner_transitions


@dataclass
class WarmupResult:
    kernels: Dict[str, hmc_block.BlockKernel]
    diagnostics: Dict[str, Any]


def run_warmup(
    rng_key: PRNGKey,
    blocks: Iterable[str],
    init_positions: Dict[str, Array],
    logdensity_fns: Dict[str, Any],
    config: BlockWarmupConfig,
) -> WarmupResult:
    """Run placeholder warm-up cycles returning tuned kernels."""

    tuned_kernels: Dict[str, hmc_block.BlockKernel] = {}
    diagnostics: Dict[str, Any] = {}
    key = rng_key
    for name in blocks:
        key, subkey = jax.random.split(key)
        _, kernel, diag = hmc_block.adapt_block_chees(
            subkey, init_positions[name], logdensity_fns[name], config
        )
        tuned_kernels[name] = kernel
        diagnostics[name] = diag
    return WarmupResult(kernels=tuned_kernels, diagnostics=diagnostics)


__all__ = ["StabilityThresholds", "BlockWarmupConfig", "WarmupResult", "run_warmup"]
