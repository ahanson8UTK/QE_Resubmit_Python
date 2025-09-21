"""Bubble component model definitions.

TODOs:
- marginalise observation variance under Jeffreys prior
- integrate with pseudo-marginal likelihood estimator
- expose control variates for variance reduction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ..typing import Array


@dataclass
class BubbleModelParams:
    """Placeholder parameters for the bubble observation equation."""

    mean: float


def bubble_loglik(params: BubbleModelParams, data: Array) -> jnp.float64:
    """Compute the marginal log-likelihood under the Gaussian bubble model stub."""

    return jnp.array(0.0, dtype=jnp.float64)


__all__ = ["BubbleModelParams", "bubble_loglik"]
