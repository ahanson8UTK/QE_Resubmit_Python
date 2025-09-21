"""Per-block HMC kernels with ChEES warm-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import optax

from ..typing import Array, PRNGKey


def _import_blackjax():
    try:
        import blackjax  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency should be present
        raise RuntimeError("blackjax must be installed to use HMC kernels") from exc
    return blackjax


@dataclass
class BlockKernel:
    """Container for a tuned HMC kernel."""

    logdensity_fn: Callable[[Array], jnp.float64]
    parameters: Dict[str, Any]


def _get_cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return default


def adapt_block_chees(
    rng_key: PRNGKey,
    init_positions: Array,
    logdensity_fn: Callable[[Array], jnp.float64],
    warmup_cfg: Any,
) -> Tuple[Array, BlockKernel, Dict[str, Any]]:
    """Run ChEES adaptation to tune an HMC kernel."""

    blackjax = _import_blackjax()
    num_chains = init_positions.shape[0]
    num_steps = int(_get_cfg_value(warmup_cfg, "num_steps", 10))
    target_acceptance = float(_get_cfg_value(warmup_cfg, "target_acceptance", 0.651))
    chees = blackjax.adaptation.chees_adaptation(
        logdensity_fn,
        num_chains=num_chains,
        target_acceptance_rate=target_acceptance,
        jitter_amount=1.0,
        decay_rate=0.5,
        optimizer=optax.adam(0.02),
    )
    adaptation_state, last_states, parameters = chees.run(rng_key, init_positions, num_steps)
    tuned_kernel = BlockKernel(logdensity_fn=logdensity_fn, parameters=parameters)
    diagnostics = {
        "acceptance_rate": getattr(adaptation_state, "acceptance_rate", None),
        "step_size": parameters.get("step_size"),
    }
    return last_states, tuned_kernel, diagnostics


def hmc_block_step(rng_key: PRNGKey, state: Any, tuned_kernel: BlockKernel) -> Tuple[Any, Dict[str, Any]]:
    """Advance one dynamic HMC step using the tuned kernel."""

    blackjax = _import_blackjax()
    kernel = blackjax.dynamic_hmc(tuned_kernel.logdensity_fn, **tuned_kernel.parameters)
    new_state, info = kernel.step(rng_key, state)
    return new_state, {"acceptance": getattr(info, "acceptance_rate", None)}


__all__ = ["BlockKernel", "adapt_block_chees", "hmc_block_step"]
