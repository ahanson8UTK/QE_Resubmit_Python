"""Pseudo-marginal likelihood estimators for the bubble block.

TODOs:
- implement unbiased log-likelihood estimator with variance control
- support correlated pseudo-marginal updates for efficiency
- add gradient estimators or fall back to MH-within-Gibbs when unavailable
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from ..typing import Array, PRNGKey


def unbiased_loglik_and_grad_estimator(params: Any, data: Array, key: PRNGKey, control_variates: Any | None = None) -> Tuple[jnp.float64, Array | None]:
    """Return a noisy log-likelihood estimate and optional gradient."""

    noise = jax.random.normal(key, dtype=jnp.float64)
    return noise, None


def pm_hmc_step(state: Any, params: Any, data: Array, key: PRNGKey) -> Tuple[Any, dict]:
    """Placeholder pseudo-marginal HMC or MH step."""

    loglik_hat, grad_hat = unbiased_loglik_and_grad_estimator(params, data, key)
    info = {"loglik_hat": loglik_hat, "used_gradient": grad_hat is not None}
    return state, info


__all__ = ["unbiased_loglik_and_grad_estimator", "pm_hmc_step"]
