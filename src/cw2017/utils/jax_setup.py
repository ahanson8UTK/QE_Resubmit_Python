"""JAX configuration and helper utilities used throughout the package."""

from __future__ import annotations

import os
from typing import Callable

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


def nan_guard(name: str, *arrays: Array) -> None:
    """Raise with diagnostics if any array contains NaNs or infs."""

    for arr in arrays:
        tensor = jnp.asarray(arr)
        if jnp.any(~jnp.isfinite(tensor)):
            stats = {
                "min": float(jnp.nanmin(tensor)),
                "max": float(jnp.nanmax(tensor)),
                "mean": float(jnp.nanmean(tensor)),
            }
            raise RuntimeError(f"{name}: detected non-finite values with stats {stats}")


__all__ = ["vmap", "vjit", "nan_guard"]
