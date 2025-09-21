"""Shared typing aliases for the cw2017 package."""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar

import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = jax.Array
LogDensityFn = Callable[[Array], jnp.float64]


class LogDensityWithGrad(Protocol):
    """Protocol for callable returning value and gradient."""

    def __call__(self, x: Array) -> tuple[jnp.float64, Array]:
        ...


T = TypeVar("T")

__all__ = ["Array", "PRNGKey", "LogDensityFn", "LogDensityWithGrad", "T"]
