"""Sampler diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class BlockDiagnostics:
    acceptance_rate: float | None
    divergences: int
    ess_per_second: float | None


def compute_ess_per_second(samples: jnp.ndarray) -> float:
    """Placeholder ESS/sec computation."""

    return 0.0


__all__ = ["BlockDiagnostics", "compute_ess_per_second"]
