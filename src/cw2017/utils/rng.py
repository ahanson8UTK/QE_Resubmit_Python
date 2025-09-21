"""Random number helper utilities for NumPy and JAX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import jax
import numpy as np

from ..typing import PRNGKey


@dataclass
class RNGManager:
    """Jointly manage NumPy and JAX RNG states for reproducibility."""

    seed: int
    numpy_rng: np.random.Generator = field(init=False, repr=False)
    key: PRNGKey = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.numpy_rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)

    def split(self, count: int = 1) -> Tuple[PRNGKey, ...]:
        keys = jax.random.split(self.key, count)
        self.key = keys[-1]
        return tuple(keys[:-1]) if count > 1 else (keys[0],)


__all__ = ["RNGManager"]
