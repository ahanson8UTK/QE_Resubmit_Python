"""JAX configuration and helper utilities used throughout the package."""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Iterator

os.environ.setdefault("JAX_USE_PJRT_C_API_ON_CPU", "0")

import jax
import jax.numpy as jnp

from ..typing import Array

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
jax.default_matmul_precision("high")


def vmap(fn: Callable[..., Array], *args, **kwargs) -> Callable[..., Array]:
    """Thin wrapper over :func:`jax.vmap` to keep imports centralised."""

    return jax.vmap(fn, *args, **kwargs)


def vjit(fn: Callable[..., Array]) -> Callable[..., Array]:
    """Wrapper adding :func:`jax.jit` with ``static_argnums=()`` by default."""

    return jax.jit(fn, static_argnums=())


@contextlib.contextmanager
def nan_guard(params: Array) -> Iterator[None]:
    """Context manager that raises if NaNs/Infs appear in JAX computations."""

    yield
    if not jnp.all(jnp.isfinite(params)):
        raise FloatingPointError("Detected non-finite parameters after JAX call")


__all__ = ["vmap", "vjit", "nan_guard"]
