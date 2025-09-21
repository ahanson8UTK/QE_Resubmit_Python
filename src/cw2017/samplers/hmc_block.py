"""Per-block HMC kernels with ChEES warm-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import optax

from ..typing import Array, PRNGKey


def _import_blackjax():
    try:  # pragma: no cover - import failure should be explicit
        import blackjax  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency should be present
        raise RuntimeError("blackjax must be installed to use HMC kernels") from exc
    return blackjax


@dataclass
class BlockKernel:
    """Container for a tuned HMC kernel."""

    logdensity_fn: Callable[[Array], jnp.float64]
    parameters: Dict[str, Any]


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _harmonic_mean(values: Array) -> jnp.float64:
    eps = 1e-9
    clipped = jnp.clip(values, min=eps)
    return jnp.array(values.size, dtype=jnp.float64) / jnp.sum(1.0 / clipped)


def _compute_ebfmi(energies: Array) -> jnp.float64:
    if energies.shape[0] <= 1:
        return jnp.nan
    diffs = jnp.diff(energies, axis=0)
    numerator = jnp.var(energies, axis=0)
    denominator = jnp.mean(diffs**2, axis=0) + 1e-12
    ebfmi = numerator / denominator
    return jnp.mean(ebfmi)


def adapt_block_chees(
    rng_key: PRNGKey,
    init_positions: Array,
    logdensity_fn: Callable[[Array], jnp.float64],
    warmup_cfg: Any,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Run ChEES adaptation to tune a dynamic HMC kernel."""

    _ = _import_blackjax()
    from blackjax.adaptation.chees_adaptation import chees_adaptation

    num_chains = init_positions.shape[0]
    warmup_steps = int(_cfg_value(warmup_cfg, "num_warmup_steps", 50))
    target_accept = float(_cfg_value(warmup_cfg, "target_accept", 0.651))
    jitter_amount = float(_cfg_value(warmup_cfg, "jitter_amount", 1.0))
    decay_rate = float(_cfg_value(warmup_cfg, "decay_rate", 0.5))
    lr = float(_cfg_value(warmup_cfg, "adam_lr", 0.02))
    init_step_size = float(_cfg_value(warmup_cfg, "initial_step_size", 0.1))

    init_positions = jnp.asarray(init_positions, dtype=jnp.float64)
    adaptation = chees_adaptation(
        logdensity_fn,
        num_chains=num_chains,
        target_acceptance_rate=target_accept,
        jitter_amount=jitter_amount,
        decay_rate=decay_rate,
    )
    optim = optax.adam(lr)
    results, info = adaptation.run(rng_key, init_positions, init_step_size, optim, warmup_steps)

    tuned_params = dict(results.parameters)
    tuned_params["logdensity_fn"] = logdensity_fn

    acceptance = info.info.acceptance_rate.astype(jnp.float64)
    energies = info.info.energy.astype(jnp.float64)
    harmonic_accept = _harmonic_mean(acceptance.reshape(-1))
    ebfmi = _compute_ebfmi(energies)
    traj = info.adaptation_state.trajectory_length.astype(jnp.float64)
    final_traj = traj[-1] if traj.shape[0] else jnp.nan

    diagnostics = {
        "acceptance_trace": acceptance,
        "acceptance_hmean": harmonic_accept,
        "ebfmi": ebfmi,
        "final_step_size": jnp.asarray(tuned_params["step_size"], dtype=jnp.float64),
        "final_trajectory_length": final_traj,
    }
    return results.state, tuned_params, diagnostics


def hmc_block_step(
    rng_key: PRNGKey,
    state: Any,
    tuned_params: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Array]]:
    """Advance one dynamic HMC step using tuned parameters."""

    blackjax = _import_blackjax()
    logdensity_fn = tuned_params["logdensity_fn"]
    kernel_params = {k: v for k, v in tuned_params.items() if k != "logdensity_fn"}
    kernel = blackjax.dynamic_hmc(logdensity_fn, **kernel_params)

    num_chains = state.position.shape[0]
    keys = jax.random.split(rng_key, num_chains)
    step = jax.vmap(kernel.step, in_axes=(0, 0))
    new_state, info = step(keys, state)
    diagnostics = {
        "acceptance": info.acceptance_rate,
        "energy": info.energy,
        "is_divergent": info.is_divergent,
        "num_integration_steps": info.num_integration_steps,
    }
    return new_state, diagnostics


__all__ = ["BlockKernel", "adapt_block_chees", "hmc_block_step"]
