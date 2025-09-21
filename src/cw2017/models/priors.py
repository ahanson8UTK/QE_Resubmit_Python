"""Prior distribution scaffolds for model parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ..typing import Array


@dataclass
class Prior:
    """Simple log-density interface."""

    def logpdf(self, value: Array) -> jnp.float64:
        raise NotImplementedError


@dataclass
class NormalPrior(Prior):
    mean: float
    scale: float

    def logpdf(self, value: Array) -> jnp.float64:
        return -0.5 * jnp.sum(((value - self.mean) / self.scale) ** 2)


__all__ = ["Prior", "NormalPrior"]
