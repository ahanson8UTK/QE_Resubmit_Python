"""Sampling utilities for the bubble pseudo-marginal HMC module.

This module wires together the extended-target formulation required for
pseudo-marginal Hamiltonian Monte Carlo (PM-HMC) and provides a driver that
uses :mod:`blackjax` to simulate trajectories.  The key ideas implemented
here are

* deterministic construction of the auxiliary Gaussian random variables used
  by the importance sampler that delivers the unbiased likelihood estimate;
* evaluation of the extended log-target which includes prior, likelihood
  estimator, auxiliary Gaussian density, and the Jacobian of the
  unconstrained-to-constrained parameter transformation; and
* a light-weight driver that performs a short dual-averaging warm-up phase
  before drawing posterior samples from the extended target.

The implementation is written entirely in terms of JAX primitives so that it
can be JIT-compiled by :mod:`blackjax` and so that gradients can flow through
all pieces of the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from blackjax import diagnostics
from blackjax.mcmc import hmc as blackjax_hmc
from blackjax.optimizers import dual_averaging

from .types import (
    BubbleData,
    BubbleParams,
    BubbleParamsUnconstrained,
    PMHMCConfig,
    PMHMCResult,
)


ArrayDict = Dict[str, jnp.ndarray]


@dataclass(frozen=True)
class _ParameterStructure:
    """Index bookkeeping for flattening the parameter PyTree."""

    d_rho_bm: int
    d_rho_bg: int
    theta_size: int
    u_block_size: int
    total_size: int
    T: int
    N_is: int

    @property
    def u_size(self) -> int:
        return 2 * self.u_block_size

    @property
    def theta_slice(self) -> slice:
        return slice(0, self.theta_size)

    @property
    def u_slice(self) -> slice:
        return slice(self.theta_size, self.theta_size + self.u_size)

    @property
    def z_block_slice(self) -> slice:
        return slice(self.theta_size, self.theta_size + self.u_block_size)

    @property
    def eps_block_slice(self) -> slice:
        return slice(
            self.theta_size + self.u_block_size,
            self.theta_size + self.u_size,
        )


def pack_u(u_np: np.random.RandomState, T: int, N_is: int) -> Dict[str, jnp.ndarray]:
    """Create a deterministic bundle of auxiliary Gaussian draws."""

    z_b = u_np.standard_normal(size=(T, N_is))
    eps_b = u_np.standard_normal(size=(T, N_is))
    return {
        "z_b": jnp.asarray(z_b, dtype=jnp.float64),
        "eps_b": jnp.asarray(eps_b, dtype=jnp.float64),
    }


def _build_structure(init: BubbleParamsUnconstrained, data: BubbleData, cfg: PMHMCConfig) -> _ParameterStructure:
    d_rho_bm = int(np.size(init.z_rho_bm))
    d_rho_bg = int(np.size(init.z_rho_bg))
    theta_size = 4 + d_rho_bm + d_rho_bg
    u_block_size = int(data.T * cfg.N_is)
    total_size = theta_size + 2 * u_block_size
    return _ParameterStructure(
        d_rho_bm=d_rho_bm,
        d_rho_bg=d_rho_bg,
        theta_size=theta_size,
        u_block_size=u_block_size,
        total_size=total_size,
        T=data.T,
        N_is=cfg.N_is,
    )


def _params_to_vector(params: BubbleParamsUnconstrained) -> jnp.ndarray:
    pieces: Iterable[jnp.ndarray] = (
        jnp.atleast_1d(jnp.asarray(params.z_B0, dtype=jnp.float64)),
        jnp.atleast_1d(jnp.asarray(params.z_mu_b, dtype=jnp.float64)),
        jnp.atleast_1d(jnp.asarray(params.z_phi_b, dtype=jnp.float64)),
        jnp.atleast_1d(jnp.asarray(params.z_sigma_h, dtype=jnp.float64)),
        jnp.asarray(params.z_rho_bm, dtype=jnp.float64).reshape(-1),
        jnp.asarray(params.z_rho_bg, dtype=jnp.float64).reshape(-1),
    )
    return jnp.concatenate(tuple(pieces))


def _vector_to_unconstrained(theta_vec: jnp.ndarray, structure: _ParameterStructure) -> Dict[str, jnp.ndarray]:
    idx = 0
    z_B0 = theta_vec[idx]
    idx += 1
    z_mu_b = theta_vec[idx]
    idx += 1
    z_phi_b = theta_vec[idx]
    idx += 1
    z_sigma_h = theta_vec[idx]
    idx += 1

    if structure.d_rho_bm:
        z_rho_bm = theta_vec[idx : idx + structure.d_rho_bm]
    else:
        z_rho_bm = jnp.zeros((0,), dtype=theta_vec.dtype)
    idx += structure.d_rho_bm

    if structure.d_rho_bg:
        z_rho_bg = theta_vec[idx : idx + structure.d_rho_bg]
    else:
        z_rho_bg = jnp.zeros((0,), dtype=theta_vec.dtype)

    return {
        "z_B0": z_B0,
        "z_mu_b": z_mu_b,
        "z_phi_b": z_phi_b,
        "z_sigma_h": z_sigma_h,
        "z_rho_bm": z_rho_bm,
        "z_rho_bg": z_rho_bg,
    }


def _concat_rho(z_rho_bm: jnp.ndarray, z_rho_bg: jnp.ndarray) -> jnp.ndarray:
    if z_rho_bm.size and z_rho_bg.size:
        return jnp.concatenate((z_rho_bm, z_rho_bg))
    if z_rho_bm.size:
        return z_rho_bm
    if z_rho_bg.size:
        return z_rho_bg
    return jnp.zeros((0,), dtype=jnp.float64)


def _constrain_params(unconstrained: Dict[str, jnp.ndarray], structure: _ParameterStructure) -> Dict[str, jnp.ndarray]:
    z_B0 = unconstrained["z_B0"]
    z_mu_b = unconstrained["z_mu_b"]
    z_phi_b = unconstrained["z_phi_b"]
    z_sigma_h = unconstrained["z_sigma_h"]
    z_rho_bm = unconstrained["z_rho_bm"]
    z_rho_bg = unconstrained["z_rho_bg"]

    B0 = jax.nn.softplus(z_B0)
    mu_b = z_mu_b
    phi_b = jnp.tanh(z_phi_b)
    sigma_h = jax.nn.softplus(z_sigma_h)

    rho_concat = _concat_rho(z_rho_bm, z_rho_bg)
    rho_norm = jnp.linalg.norm(rho_concat)

    def _scale_fn(norm_val: jnp.ndarray) -> jnp.ndarray:
        return 1.0 / (1.0 + norm_val)

    scale = jax.lax.cond(
        rho_norm > 0.0,
        _scale_fn,
        lambda _: jnp.array(1.0, dtype=rho_concat.dtype),
        rho_norm,
    )
    rho = rho_concat * scale

    if structure.d_rho_bm:
        rho_bm = rho[: structure.d_rho_bm]
    else:
        rho_bm = jnp.zeros((0,), dtype=rho_concat.dtype)

    if structure.d_rho_bg:
        rho_bg = rho[structure.d_rho_bm : structure.d_rho_bm + structure.d_rho_bg]
    else:
        rho_bg = jnp.zeros((0,), dtype=rho_concat.dtype)

    return {
        "B0": B0,
        "mu_b": mu_b,
        "phi_b": phi_b,
        "sigma_h": sigma_h,
        "rho_bm": rho_bm,
        "rho_bg": rho_bg,
        "rho_norm_unconstrained": rho_norm,
    }


def _log_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return -jax.nn.softplus(-x)


def _log_det_jacobian(
    unconstrained: Dict[str, jnp.ndarray],
    constrained: Dict[str, jnp.ndarray],
    structure: _ParameterStructure,
) -> jnp.ndarray:
    log_jac = _log_sigmoid(unconstrained["z_B0"]) + _log_sigmoid(unconstrained["z_sigma_h"])
    log_jac = log_jac + jnp.log1p(-constrained["phi_b"] ** 2)
    rho_dim = structure.d_rho_bm + structure.d_rho_bg
    if rho_dim:
        log_jac = log_jac - (rho_dim + 1.0) * jnp.log1p(constrained["rho_norm_unconstrained"])
    return log_jac


def _vector_to_u_dict(u_vec: jnp.ndarray, structure: _ParameterStructure) -> ArrayDict:
    block = structure.u_block_size
    shape = (structure.T, structure.N_is)
    z_flat = u_vec[:block]
    eps_flat = u_vec[block: block + block]
    return {
        "z_b": jnp.reshape(z_flat, shape),
        "eps_b": jnp.reshape(eps_flat, shape),
    }


def _u_dict_to_vector(u_dict: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(
        (
            jnp.reshape(u_dict["z_b"], (-1,)),
            jnp.reshape(u_dict["eps_b"], (-1,)),
        )
    )


def _standard_normal_logpdf(vec: jnp.ndarray) -> jnp.ndarray:
    const = 0.5 * vec.size * jnp.log(2.0 * jnp.pi)
    return -0.5 * jnp.sum(vec**2) - const


def _extended_components(
    theta_vec: jnp.ndarray,
    u_vec: jnp.ndarray,
    data: BubbleData,
    cfg: PMHMCConfig,
    pre: Mapping[str, Any],
) -> tuple[
    Dict[str, jnp.ndarray],
    Dict[str, jnp.ndarray],
    ArrayDict,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    structure: _ParameterStructure = pre["structure"]
    unconstrained = _vector_to_unconstrained(theta_vec, structure)
    constrained = _constrain_params(unconstrained, structure)
    u_dict = _vector_to_u_dict(u_vec, structure)

    log_prior_fn = pre.get("log_prior_fn")
    if log_prior_fn is None:
        log_prior = jnp.array(0.0, dtype=theta_vec.dtype)
    else:
        log_prior = log_prior_fn(constrained, unconstrained, data, cfg, pre)

    loglik_hat_fn = pre.get("loglik_hat_fn")
    if loglik_hat_fn is None:
        raise ValueError("pre must define a 'loglik_hat_fn' callable.")
    loglik_hat = loglik_hat_fn(constrained, unconstrained, u_dict, data, cfg, pre)

    log_phi = _standard_normal_logpdf(u_vec)
    log_jac = _log_det_jacobian(unconstrained, constrained, structure)
    return unconstrained, constrained, u_dict, log_prior, loglik_hat, log_phi, log_jac


def log_target_extended(
    theta_u_vec: jnp.ndarray,
    u_vec: jnp.ndarray,
    data: BubbleData,
    cfg: PMHMCConfig,
    pre: Mapping[str, Any],
) -> jnp.ndarray:
    """Evaluate the log-density of the extended PM-HMC target."""

    _, _, _, log_prior, loglik_hat, log_phi, log_jac = _extended_components(
        theta_u_vec, u_vec, data, cfg, pre
    )
    return jnp.asarray(log_prior + loglik_hat + log_phi + log_jac, dtype=jnp.float64)


def _posterior_arrays(
    theta_draws: jnp.ndarray, structure: _ParameterStructure
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]:
    if theta_draws.size == 0:
        unconstrained = {
            "z_B0": jnp.zeros((0,), dtype=jnp.float64),
            "z_mu_b": jnp.zeros((0,), dtype=jnp.float64),
            "z_phi_b": jnp.zeros((0,), dtype=jnp.float64),
            "z_sigma_h": jnp.zeros((0,), dtype=jnp.float64),
            "z_rho_bm": jnp.zeros((0, structure.d_rho_bm), dtype=jnp.float64),
            "z_rho_bg": jnp.zeros((0, structure.d_rho_bg), dtype=jnp.float64),
        }
        constrained = {
            "B0": jnp.zeros((0,), dtype=jnp.float64),
            "mu_b": jnp.zeros((0,), dtype=jnp.float64),
            "phi_b": jnp.zeros((0,), dtype=jnp.float64),
            "sigma_h": jnp.zeros((0,), dtype=jnp.float64),
            "rho_bm": jnp.zeros((0, structure.d_rho_bm), dtype=jnp.float64),
            "rho_bg": jnp.zeros((0, structure.d_rho_bg), dtype=jnp.float64),
        }
        log_jac = jnp.zeros((0,), dtype=jnp.float64)
        return unconstrained, constrained, log_jac

    idx = 0
    z_B0 = theta_draws[:, idx]
    idx += 1
    z_mu_b = theta_draws[:, idx]
    idx += 1
    z_phi_b = theta_draws[:, idx]
    idx += 1
    z_sigma_h = theta_draws[:, idx]
    idx += 1

    rho_dim = structure.d_rho_bm + structure.d_rho_bg
    if rho_dim:
        rho_concat = theta_draws[:, idx : idx + rho_dim]
    else:
        rho_concat = jnp.zeros((theta_draws.shape[0], 0), dtype=theta_draws.dtype)

    unconstrained = {
        "z_B0": z_B0,
        "z_mu_b": z_mu_b,
        "z_phi_b": z_phi_b,
        "z_sigma_h": z_sigma_h,
        "z_rho_bm": rho_concat[:, : structure.d_rho_bm]
        if structure.d_rho_bm
        else jnp.zeros((theta_draws.shape[0], 0), dtype=theta_draws.dtype),
        "z_rho_bg": rho_concat[:, structure.d_rho_bm : structure.d_rho_bm + structure.d_rho_bg]
        if structure.d_rho_bg
        else jnp.zeros((theta_draws.shape[0], 0), dtype=theta_draws.dtype),
    }

    B0 = jax.nn.softplus(z_B0)
    mu_b = z_mu_b
    phi_b = jnp.tanh(z_phi_b)
    sigma_h = jax.nn.softplus(z_sigma_h)

    rho_norm = jnp.linalg.norm(rho_concat, axis=1)
    scale = jnp.where(rho_norm > 0.0, 1.0 / (1.0 + rho_norm), 1.0)
    rho = rho_concat * scale[:, None] if rho_dim else rho_concat
    rho_bm = rho[:, : structure.d_rho_bm] if structure.d_rho_bm else rho[:, :0]
    rho_bg = (
        rho[:, structure.d_rho_bm : structure.d_rho_bm + structure.d_rho_bg]
        if structure.d_rho_bg
        else rho[:, :0]
    )

    constrained = {
        "B0": B0,
        "mu_b": mu_b,
        "phi_b": phi_b,
        "sigma_h": sigma_h,
        "rho_bm": rho_bm,
        "rho_bg": rho_bg,
    }

    log_jac = _log_sigmoid(z_B0) + _log_sigmoid(z_sigma_h) + jnp.log1p(-phi_b**2)
    if rho_dim:
        log_jac = log_jac - (rho_dim + 1.0) * jnp.log1p(rho_norm)

    return unconstrained, constrained, log_jac


def _dict_to_bubble_params(params: Dict[str, jnp.ndarray]) -> BubbleParams:
    return BubbleParams(
        B0=float(np.asarray(params["B0"])),
        mu_b=float(np.asarray(params["mu_b"])),
        phi_b=float(np.asarray(params["phi_b"])),
        sigma_h=float(np.asarray(params["sigma_h"])),
        rho_bm=np.asarray(params["rho_bm"]),
        rho_bg=np.asarray(params["rho_bg"]),
    )


def _dict_to_unconstrained_params(params: Dict[str, jnp.ndarray]) -> BubbleParamsUnconstrained:
    return BubbleParamsUnconstrained(
        z_B0=float(np.asarray(params["z_B0"])),
        z_mu_b=float(np.asarray(params["z_mu_b"])),
        z_phi_b=float(np.asarray(params["z_phi_b"])),
        z_sigma_h=float(np.asarray(params["z_sigma_h"])),
        z_rho_bm=np.asarray(params["z_rho_bm"]),
        z_rho_bg=np.asarray(params["z_rho_bg"]),
    )


def run_pmhmc(
    data: BubbleData,
    cfg: PMHMCConfig,
    init: BubbleParamsUnconstrained,
    pre: Mapping[str, Any],
) -> PMHMCResult:
    """Run a pseudo-marginal HMC chain targeting the extended bubble posterior."""

    structure = _build_structure(init, data, cfg)
    theta0_vec = _params_to_vector(init)
    rng_np = np.random.RandomState(cfg.seed)
    u_init = pack_u(rng_np, structure.T, structure.N_is)
    u0_vec = _u_dict_to_vector(u_init)
    initial_position = jnp.concatenate((theta0_vec, u0_vec))

    inverse_mass_matrix = jnp.ones(structure.total_size, dtype=jnp.float64)

    sampler_pre = dict(pre)
    sampler_pre["structure"] = structure

    def logprob(position: jnp.ndarray) -> jnp.ndarray:
        theta_vec = position[structure.theta_slice]
        u_vec = position[structure.u_slice]
        return log_target_extended(theta_vec, u_vec, data, cfg, sampler_pre)

    state = blackjax_hmc.init(initial_position, logprob)
    kernel = blackjax_hmc.build_kernel()

    rng_key = jax.random.PRNGKey(cfg.seed)
    warmup_steps = int(getattr(cfg, "hmc_num_warmup", 0))
    num_steps = int(cfg.hmc_num_steps)
    num_samples = int(cfg.hmc_num_samples)
    target_accept = 0.75

    da_init, da_update, da_final = dual_averaging.dual_averaging()
    step_size = float(cfg.hmc_step_size)
    da_state = da_init(step_size)
    warmup_acceptance: list[jnp.ndarray] = []
    warmup_step_sizes: list[float] = []

    for _ in range(warmup_steps):
        warmup_step_sizes.append(float(step_size))
        rng_key, subkey = jax.random.split(rng_key)
        state, info = kernel(
            subkey, state, logprob, step_size, inverse_mass_matrix, num_steps
        )
        accept_prob = info.acceptance_probability
        warmup_acceptance.append(accept_prob)
        da_state = da_update(da_state, target_accept - accept_prob)
        step_size = float(jnp.exp(da_state.log_x))

    if warmup_steps:
        step_size = float(da_final(da_state))

    theta_draws_list: list[jnp.ndarray] = []
    u_draws_list: list[jnp.ndarray] = []
    accept_list: list[jnp.ndarray] = []
    loglik_list: list[jnp.ndarray] = []
    logtarget_list: list[jnp.ndarray] = []

    for _ in range(num_samples):
        rng_key, subkey = jax.random.split(rng_key)
        state, info = kernel(
            subkey, state, logprob, step_size, inverse_mass_matrix, num_steps
        )
        position = state.position
        theta_vec = position[structure.theta_slice]
        u_vec = position[structure.u_slice]
        theta_draws_list.append(theta_vec)
        u_draws_list.append(u_vec)
        accept_list.append(info.acceptance_probability)
        logtarget_list.append(state.logdensity)
        _, _, _, _, loglik_hat, _, _ = _extended_components(
            theta_vec, u_vec, data, cfg, sampler_pre
        )
        loglik_list.append(loglik_hat)

    if theta_draws_list:
        theta_draws = jnp.stack(theta_draws_list, axis=0)
    else:
        theta_draws = jnp.zeros((0, structure.theta_size), dtype=jnp.float64)

    if u_draws_list:
        u_draws = jnp.stack(u_draws_list, axis=0)
    else:
        u_draws = jnp.zeros((0, structure.u_size), dtype=jnp.float64)

    if accept_list:
        accept_arr = jnp.stack(accept_list)
    else:
        accept_arr = jnp.zeros((0,), dtype=jnp.float64)

    if loglik_list:
        loglik_arr = jnp.stack(loglik_list)
    else:
        loglik_arr = jnp.zeros((0,), dtype=jnp.float64)

    if logtarget_list:
        logtarget_arr = jnp.stack(logtarget_list)
    else:
        logtarget_arr = jnp.zeros((0,), dtype=jnp.float64)

    if warmup_acceptance:
        warmup_accept_arr = jnp.stack(warmup_acceptance)
    else:
        warmup_accept_arr = jnp.zeros((0,), dtype=jnp.float64)

    warmup_step_arr = jnp.asarray(warmup_step_sizes, dtype=jnp.float64)

    unconstrained_draws, constrained_draws, log_jac_draws = _posterior_arrays(
        theta_draws, structure
    )

    if theta_draws.size:
        ess = diagnostics.effective_sample_size(
            theta_draws[jnp.newaxis, ...], chain_axis=0, sample_axis=1
        )
        ess_values = np.asarray(ess)
    else:
        ess_values = np.zeros((structure.theta_size,))

    ess_dict = {
        "z_B0": ess_values[0] if ess_values.size else 0.0,
        "z_mu_b": ess_values[1] if ess_values.size > 1 else 0.0,
        "z_phi_b": ess_values[2] if ess_values.size > 2 else 0.0,
        "z_sigma_h": ess_values[3] if ess_values.size > 3 else 0.0,
        "z_rho_bm": ess_values[4 : 4 + structure.d_rho_bm]
        if ess_values.size >= 4 + structure.d_rho_bm
        else np.zeros((structure.d_rho_bm,)),
        "z_rho_bg": ess_values[4 + structure.d_rho_bm : 4 + structure.d_rho_bm + structure.d_rho_bg]
        if ess_values.size >= 4 + structure.d_rho_bm + structure.d_rho_bg
        else np.zeros((structure.d_rho_bg,)),
    }

    final_position = state.position
    final_theta = final_position[structure.theta_slice]
    final_u = final_position[structure.u_slice]
    final_unconstrained, final_constrained, final_u_dict, _, final_loglik, _, _ = _extended_components(
        final_theta, final_u, data, cfg, sampler_pre
    )

    result_params = _dict_to_bubble_params(final_constrained)
    result_unconstrained = _dict_to_unconstrained_params(final_unconstrained)

    accept_rate = float(jnp.mean(accept_arr)) if accept_arr.size else float("nan")
    mean_loglik = float(jnp.mean(loglik_arr)) if loglik_arr.size else float("nan")

    draws_dict: Dict[str, np.ndarray] = {
        "B0": np.asarray(constrained_draws["B0"]),
        "mu_b": np.asarray(constrained_draws["mu_b"]),
        "phi_b": np.asarray(constrained_draws["phi_b"]),
        "sigma_h": np.asarray(constrained_draws["sigma_h"]),
        "rho_bm": np.asarray(constrained_draws["rho_bm"]),
        "rho_bg": np.asarray(constrained_draws["rho_bg"]),
        "z_B0": np.asarray(unconstrained_draws["z_B0"]),
        "z_mu_b": np.asarray(unconstrained_draws["z_mu_b"]),
        "z_phi_b": np.asarray(unconstrained_draws["z_phi_b"]),
        "z_sigma_h": np.asarray(unconstrained_draws["z_sigma_h"]),
        "z_rho_bm": np.asarray(unconstrained_draws["z_rho_bm"]),
        "z_rho_bg": np.asarray(unconstrained_draws["z_rho_bg"]),
    }

    diagnostics_dict: Dict[str, Any] = {
        "warmup_acceptance": np.asarray(warmup_accept_arr),
        "warmup_step_sizes": np.asarray(warmup_step_arr),
        "acceptance": np.asarray(accept_arr),
        "loglik_hat": np.asarray(loglik_arr),
        "logtarget": np.asarray(logtarget_arr),
        "log_jacobian": np.asarray(log_jac_draws),
        "ess_unconstrained": ess_dict,
        "step_size": step_size,
        "final_unconstrained": result_unconstrained,
        "final_u": {k: np.asarray(v) for k, v in final_u_dict.items()},
        "u_draws": np.asarray(u_draws),
        "final_loglik_hat": float(np.asarray(final_loglik)),
    }

    return PMHMCResult(
        params=result_params,
        log_like_hat=mean_loglik,
        accept_rate=accept_rate,
        draws_constrained=draws_dict,
        diagnostics=diagnostics_dict,
    )


__all__ = ["pack_u", "log_target_extended", "run_pmhmc"]

