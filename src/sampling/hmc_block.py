"""Generic NUTS block driver with change-of-variable transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


@dataclass
class HMCConfig:
    """Configuration for a single HMC-within-Gibbs block."""

    target_accept: float = 0.85
    max_treedepth: int = 10
    dense_mass: bool = False
    warmup_long: int = 400
    warmup_refresh: int = 50
    num_draws: int = 1
    chees_batch: int = 50
    chees_tol: float = 0.05
    init_step_size: float = 0.1


@dataclass
class HMCCache:
    """Cache holding adapted step-size and mass matrix."""

    step_size: float | None = None
    inv_mass_matrix: jnp.ndarray | None = None
    dense_mass: bool = False


def _ebfmi(kinetic_energies: jnp.ndarray) -> float:
    """Compute the energy Bayesian fraction of missing information."""

    if kinetic_energies.size < 2:
        return float("nan")
    diffs = jnp.diff(kinetic_energies)
    numerator = jnp.mean(diffs**2)
    denominator = jnp.var(kinetic_energies) + 1e-12
    return float(numerator / denominator)


def _ess_batch_means(samples: jnp.ndarray) -> float:
    """Cheap batch-means effective sample size estimate."""

    if samples.ndim == 1:
        series = samples
    else:
        series = samples.reshape(samples.shape[0], -1)
    n = series.shape[0]
    if n < 20:
        return float(n)
    m = int(jnp.sqrt(n))
    b = n // m
    trimmed = series[: b * m]
    batch_means = trimmed.reshape(m, b, -1).mean(axis=1)
    overall_var = jnp.var(trimmed, axis=0)
    batch_var = jnp.var(batch_means, axis=0)
    ess = n * overall_var / (b * batch_var + 1e-12)
    return float(jnp.mean(ess))


class NUTSBlock:
    """Wraps a conditional log-posterior defined on unconstrained parameters."""

    def __init__(
        self,
        logprob_fn: Callable[[PyTree, Dict[str, Any]], Tuple[jnp.ndarray, Dict[str, Any]]],
        transform: Callable[[PyTree], Tuple[PyTree, float]],
        inv_transform: Callable[[PyTree], PyTree],
        cfg: HMCConfig,
    ) -> None:
        self.logprob_fn = logprob_fn
        self.transform = transform
        self.inv_transform = inv_transform
        self.cfg = cfg

    def _logprob_u(self, theta_u: PyTree, context: Dict[str, Any]) -> jnp.ndarray:
        theta_c, logabsdet = self.transform(theta_u)
        logprob, _ = self.logprob_fn(theta_c, context)
        return logprob + logabsdet

    def warmup_and_draw(
        self,
        init_theta_c: PyTree,
        cache: HMCCache | None,
        context: Dict[str, Any],
        key: jax.Array,
    ) -> Tuple[PyTree, Dict[str, Any], HMCCache]:
        """Run warm-up (with cache reuse) and draw ``num_draws`` samples."""

        theta_u0 = self.inv_transform(init_theta_c)
        logprob = lambda th: self._logprob_u(th, context)

        use_dense = self.cfg.dense_mass or (cache.dense_mass if cache else False)
        init_step = cache.step_size if cache and cache.step_size is not None else self.cfg.init_step_size

        n_adapt = self.cfg.warmup_long if cache is None or cache.step_size is None else self.cfg.warmup_refresh
        n_adapt = max(int(n_adapt), 0)

        adapt_key, draw_key = jax.random.split(key)
        adapt = blackjax.window_adaptation(
            blackjax.nuts,
            logprob,
            is_mass_matrix_diagonal=not use_dense,
            initial_step_size=float(init_step),
            target_acceptance_rate=self.cfg.target_accept,
            max_num_doublings=self.cfg.max_treedepth,
        )
        adapt_res, adapt_info = adapt.run(adapt_key, theta_u0, n_adapt if n_adapt > 0 else 1)
        warmup_state = adapt_res.state
        params = adapt_res.parameters

        energies = jnp.asarray(adapt_info.info.energy)
        ebfmi = _ebfmi(energies)
        warmup_accept = float(jnp.mean(jnp.asarray(adapt_info.info.acceptance_rate)))

        nuts_kernel = blackjax.nuts(
            logprob,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            max_num_doublings=params.get("max_num_doublings", self.cfg.max_treedepth),
        )

        draws_u = []
        accepts = []
        divergences = []
        depths = []
        state_draw = warmup_state
        for _ in range(max(1, int(self.cfg.num_draws))):
            draw_key, step_key = jax.random.split(draw_key)
            state_draw, info = nuts_kernel.step(step_key, state_draw)
            draws_u.append(jnp.asarray(state_draw.position))
            accepts.append(float(info.acceptance_rate))
            divergences.append(bool(info.is_divergent))
            depths.append(int(info.num_trajectory_expansions))

        draws_arr = jnp.stack(draws_u, axis=0)
        ess_proxy = _ess_batch_means(draws_arr)

        theta_samples = [self.transform(u)[0] for u in draws_u]
        theta_last = theta_samples[-1]

        inv_mass = jnp.asarray(params["inverse_mass_matrix"])
        if use_dense:
            inv_mass_np = np.asarray(inv_mass)
            eigs_inv = np.linalg.eigvalsh(inv_mass_np)
        else:
            eigs_inv = np.asarray(inv_mass)
        eigs_mass = 1.0 / np.clip(eigs_inv, 1e-12, None)

        diag = {
            "accept_rate": float(np.mean(accepts)) if accepts else float("nan"),
            "warmup_accept_rate": warmup_accept,
            "max_treedepth": int(np.max(depths)) if depths else 0,
            "divergences": int(np.sum(divergences)) if divergences else 0,
            "ebfmi": ebfmi,
            "ess_proxy": ess_proxy,
            "step_size": float(params["step_size"]),
            "mass_matrix_eigs": eigs_mass.tolist(),
            "warmup_steps": int(n_adapt),
            "dense_mass": bool(use_dense),
        }

        new_cache = HMCCache(
            step_size=float(params["step_size"]),
            inv_mass_matrix=jnp.asarray(params["inverse_mass_matrix"]),
            dense_mass=bool(use_dense),
        )
        return theta_last, diag, new_cache


__all__ = ["HMCConfig", "HMCCache", "NUTSBlock"]
