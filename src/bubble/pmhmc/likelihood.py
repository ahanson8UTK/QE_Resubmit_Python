"""Pseudo-marginal likelihood helpers for the bubble PM-HMC block.

The module provides three public functions:

``proposal_moments``
    Computes the mean and variance of the Gaussian proposal distribution for
    each time index assuming the bubble state is started at its steady-state
    level.

``log_weight_one_s``
    Evaluates the log importance weights for a single time index ``s`` given
    the Monte Carlo draws of the auxiliary variables.

``loglik_hat``
    Assembles the pseudo-marginal log-likelihood estimator by combining the
    per-period weights with Gauss–Hermite quadrature over the ``chi``
    auxiliary variable.

The implementation is written with JAX in mind: all heavy computations rely on
JAX primitives, enabling vectorisation and ``jax.jit`` compilation.  The code
is intentionally flexible so callers can inject problem-specific inputs through
``pre`` and ``u`` while keeping the functional core easily testable.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import log_sigmoid
from jax.scipy import special as jsp_special
from jax.tree_util import Partial

from .transforms import constrain_params
from .types import BubbleData, BubbleParams, BubbleParamsUnconstrained, PMHMCConfig

Array = jnp.ndarray


def _get_param(theta: BubbleParams, name: str) -> Array:
    """Return ``theta.name`` as a float64 JAX array."""

    return jnp.asarray(getattr(theta, name), dtype=jnp.float64)


def _infer_num_steps(steady_inputs: Mapping[str, Any]) -> int:
    """Infer the number of time steps encoded in ``steady_inputs``."""

    if "num_steps" in steady_inputs:
        return int(steady_inputs["num_steps"])
    if "T" in steady_inputs:
        return int(steady_inputs["T"])
    for value in steady_inputs.values():
        if isinstance(value, int):
            return int(value)
        if hasattr(value, "shape") and value.shape:
            return int(value.shape[0])
    raise ValueError("Unable to infer number of steps from steady_inputs.")


def _stable_divide(num: Array, denom: Array) -> Array:
    """Compute ``num / denom`` guarding against tiny denominators."""

    tiny = jnp.array(1e-8, dtype=num.dtype)
    safe_denom = jnp.where(jnp.abs(denom) < tiny, jnp.sign(denom) * tiny, denom)
    return num / safe_denom


def proposal_moments(
    theta: BubbleParams, steady_inputs: Mapping[str, Any]
) -> Tuple[Array, Array]:
    """Return steady-state proposal moments for the bubble state.

    Parameters
    ----------
    theta:
        Constrained bubble parameters.
    steady_inputs:
        Mapping containing auxiliary information.  Only the time dimension is
        required (via ``num_steps`` or ``T``); additional keys are ignored.

    Returns
    -------
    Tuple[Array, Array]
        Arrays ``(mu_X, sigma2_X)`` with the steady-state mean and variance for
        each time index.
    """

    num_steps = _infer_num_steps(steady_inputs)
    mu_b = _get_param(theta, "mu_b")
    phi_b = _get_param(theta, "phi_b")
    sigma_h = _get_param(theta, "sigma_h")

    one = jnp.array(1.0, dtype=jnp.float64)
    mu_ss = _stable_divide(mu_b, one - phi_b)
    var_ss = _stable_divide(sigma_h ** 2, one - phi_b ** 2)

    mu_path = jnp.full((num_steps,), mu_ss, dtype=jnp.float64)
    var_path = jnp.full((num_steps,), var_ss, dtype=jnp.float64)
    return mu_path, var_path


def _reshape_mc_draws(draw: Array, T: int, N_is: int) -> Array:
    """Reshape importance sampling draws to ``(T, N_is, ...)`` if necessary."""

    arr = jnp.asarray(draw, dtype=jnp.float64)
    if arr.shape[:2] == (T, N_is):
        return arr
    if arr.shape and arr.shape[0] == T * N_is:
        new_shape = (T, N_is) + arr.shape[1:]
        return jnp.reshape(arr, new_shape)
    if arr.shape and arr.shape[0] == T:
        arr = jnp.expand_dims(arr, axis=1)
        return jnp.broadcast_to(arr, (T, N_is) + arr.shape[2:])
    raise ValueError(
        "Draws must have leading dimension T or T*N_is; received shape "
        f"{arr.shape}."
    )


def _expand_time_array(arr: Array, T: int, event_shape: Tuple[int, ...]) -> Array:
    """Broadcast ``arr`` to shape ``(T,) + event_shape``."""

    value = jnp.asarray(arr, dtype=jnp.float64)
    target_shape = (T,) + event_shape
    if value.shape == target_shape:
        return value
    if value.ndim == 0:
        return jnp.broadcast_to(value, target_shape)
    if value.shape == (T,):
        reshape_shape = (T,) + (1,) * len(event_shape)
        return jnp.broadcast_to(jnp.reshape(value, reshape_shape), target_shape)
    if value.shape == event_shape:
        return jnp.broadcast_to(value, target_shape)
    raise ValueError(f"Cannot broadcast array of shape {value.shape} to {target_shape}.")


def _sum_event_dims(x: Array) -> Array:
    """Sum over all event dimensions (axes other than the first)."""

    if x.ndim <= 1:
        return x
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes)


def _event_size(x: Array) -> Array:
    """Return the size of the event dimension as a scalar array."""

    if x.ndim <= 1:
        return jnp.array(1.0, dtype=x.dtype)
    event_dims = jnp.array(x.shape[1:], dtype=x.dtype)
    return jnp.prod(event_dims)


def _gaussian_logpdf(x: Array, mean: Array, variance: Array) -> Array:
    """Log-density of an independent Gaussian with diagonal variance."""

    var = jnp.clip(variance, a_min=1e-12)
    resid = jnp.asarray(x - mean, dtype=x.dtype)
    quad = _sum_event_dims(resid ** 2 / var)
    log_norm = _sum_event_dims(jnp.log(2.0 * jnp.pi * var))
    return -0.5 * (log_norm + quad)


def _student_t_logpdf(resid: Array, df: Array, scale: Array) -> Array:
    """Student-``t`` log-density with optional Gaussian limit for ``df=inf``."""

    dtype = resid.dtype
    scale_arr = jnp.asarray(scale, dtype=dtype)
    scale_arr = jnp.clip(scale_arr, a_min=1e-8)
    resid_arr = jnp.asarray(resid, dtype=dtype)
    resid_scaled_sq = (resid_arr / scale_arr) ** 2
    quad = _sum_event_dims(resid_scaled_sq)
    log_scale_term = _sum_event_dims(jnp.log(scale_arr))

    df_arr = jnp.asarray(df, dtype=dtype)
    df_broadcast = jnp.broadcast_to(df_arr, quad.shape)
    use_gaussian = jnp.isinf(df_broadcast)
    df_safe = jnp.where(use_gaussian, jnp.ones_like(df_broadcast), df_broadcast)
    df_safe = jnp.clip(df_safe, a_min=1e-6)

    expand_shape = df_safe.shape + (1,) * (resid_arr.ndim - 1)
    df_event = jnp.reshape(df_safe, expand_shape)

    norm_term = (
        jsp_special.gammaln(0.5 * (df_event + 1.0))
        - jsp_special.gammaln(0.5 * df_event)
        - 0.5 * (jnp.log(df_event) + jnp.log(jnp.pi))
        - jnp.log(scale_arr)
    )
    quad_term = -0.5 * (df_event + 1.0) * jnp.log1p(resid_scaled_sq / df_event)
    log_student = _sum_event_dims(norm_term + quad_term)

    event_size = _event_size(resid_arr)
    log_gaussian = -0.5 * (
        quad + 2.0 * log_scale_term + event_size * jnp.log(2.0 * jnp.pi)
    )
    return jnp.where(use_gaussian, log_gaussian, log_student)


def _default_chi_integrand(
    _theta: BubbleParams,
    t_inputs: Mapping[str, Any],
    X1_s: Array,
    _z_b_s: Array,
    chi: Array,
) -> Array:
    """Fallback integrand linear in ``chi`` and ``X1_s``."""

    dtype = X1_s.dtype
    constant = jnp.asarray(t_inputs["chi_constant"], dtype=dtype)
    linear = jnp.asarray(t_inputs["chi_linear"], dtype=dtype)
    cross = jnp.asarray(t_inputs["chi_cross"], dtype=dtype)
    return constant + linear * chi + cross * chi * X1_s


def integrate_over_chi(
    theta: BubbleParams,
    t_inputs: Mapping[str, Any],
    X1_s: Array,
    z_b_s: Array,
    gh_nodes: int,
    use_truncation: bool,
) -> Array:
    """Integrate the ``chi``-dependent term using Gauss–Hermite quadrature."""

    dtype = X1_s.dtype
    nodes, weights = jnp.polynomial.hermite.hermgauss(gh_nodes)
    nodes = nodes.astype(dtype)
    weights = weights.astype(dtype)
    sqrt_two = jnp.sqrt(jnp.array(2.0, dtype=dtype))

    loc = jnp.asarray(t_inputs["chi_loc"], dtype=dtype)
    scale = jnp.asarray(t_inputs["chi_scale"], dtype=dtype)
    scale = jnp.clip(scale, a_min=1e-8)
    lower = jnp.asarray(t_inputs["chi_lower"], dtype=dtype)
    upper = jnp.asarray(t_inputs["chi_upper"], dtype=dtype)
    integrand = t_inputs["chi_integrand"]

    def eval_node(node: Array) -> Array:
        chi = loc + scale * sqrt_two * node
        if use_truncation:
            chi = jnp.clip(chi, lower, upper)
        return integrand(theta, t_inputs, X1_s, z_b_s, chi)

    values = jax.vmap(eval_node)(nodes)
    weighted = jnp.tensordot(weights, values, axes=(0, 0))
    sqrt_pi = jnp.sqrt(jnp.array(jnp.pi, dtype=dtype))
    integral = weighted / sqrt_pi
    return _sum_event_dims(integral)


def _cumprod_path(factors: Array) -> Array:
    """Return cumulative product path for ``factors`` via ``lax.scan``."""

    dtype = factors.dtype

    def step(carry: Array, x: Array) -> Tuple[Array, Array]:
        new_carry = carry * x
        return new_carry, new_carry

    _, out = lax.scan(step, jnp.array(1.0, dtype=dtype), factors)
    return out


def _build_growth_path(T: int, pre: Mapping[str, Any], dtype: jnp.dtype) -> Array:
    """Construct the growth path used in the observation equation."""

    if "growth_path" in pre:
        arr = jnp.asarray(pre["growth_path"], dtype=dtype)
        if arr.shape[0] != T:
            arr = jnp.broadcast_to(arr, (T,))
        return arr
    if "growth_factors" in pre:
        factors = jnp.asarray(pre["growth_factors"], dtype=dtype)
        if factors.shape[0] != T:
            factors = jnp.broadcast_to(factors, (T,))
        return _cumprod_path(factors)
    if "growth_rate" in pre:
        rate = jnp.asarray(pre["growth_rate"], dtype=dtype)
        if rate.shape[0] != T:
            rate = jnp.broadcast_to(rate, (T,))
        factors = 1.0 + rate
        return _cumprod_path(factors)
    return jnp.ones((T,), dtype=dtype)


def _build_state_paths(
    theta: BubbleParams, T: int, pre: Mapping[str, Any], dtype: jnp.dtype
) -> Tuple[Array, Array]:
    """Return the mean path ``h_b`` and scale path for the AR(1) state."""

    mu_b = _get_param(theta, "mu_b")
    phi_b = _get_param(theta, "phi_b")
    sigma_h = _get_param(theta, "sigma_h")
    steady = _stable_divide(mu_b, jnp.array(1.0, dtype=dtype) - phi_b)
    h0 = jnp.asarray(pre.get("h_b0", steady), dtype=dtype)

    def step(carry: Array, _unused: Array) -> Tuple[Array, Array]:
        next_val = mu_b + phi_b * carry
        return next_val, next_val

    _, h_path = lax.scan(step, h0, jnp.arange(T, dtype=jnp.int32))
    scale_path = jnp.full((T,), sigma_h, dtype=dtype)
    return h_path, scale_path


def _log_abs_det_jacobian(params_u: BubbleParamsUnconstrained) -> Array:
    """Log absolute determinant of the transform Jacobian."""

    z_B0 = jnp.asarray(params_u.z_B0, dtype=jnp.float64)
    z_sigma_h = jnp.asarray(params_u.z_sigma_h, dtype=jnp.float64)
    z_phi = jnp.asarray(params_u.z_phi_b, dtype=jnp.float64)
    rho_concat = jnp.concatenate(
        (
            jnp.asarray(params_u.z_rho_bm, dtype=jnp.float64),
            jnp.asarray(params_u.z_rho_bg, dtype=jnp.float64),
        )
    )

    log_det_B0 = log_sigmoid(z_B0)
    log_det_sigma_h = log_sigmoid(z_sigma_h)
    tanh_z = jnp.tanh(z_phi)
    log_det_phi = jnp.log1p(-tanh_z ** 2)

    norm = jnp.linalg.norm(rho_concat)
    dim = rho_concat.size
    log_det_rho = -(dim + 1.0) * jnp.log1p(norm)
    return log_det_B0 + log_det_sigma_h + log_det_phi + log_det_rho


def _as_theta_dict(theta: BubbleParams) -> Dict[str, Array]:
    """Convert ``theta`` to a dictionary of JAX arrays."""

    theta_dict = asdict(theta)
    return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in theta_dict.items()}


def _prepare_time_inputs(
    base_inputs: Mapping[str, Any],
    y: Array,
    mu_X: Array,
    sigma2_X: Array,
    proposal_scale: Array,
    X1: Array,
    eps_b: Array,
    growth_path: Array,
    chi_integrand: Partial,
    dtype: jnp.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the per-time inputs dictionary and matching ``vmap`` axis spec."""

    T = y.shape[0]
    time_inputs: Dict[str, Any] = {}
    axis_spec: Dict[str, Any] = {}

    def register(key: str, value: Any, axis: Any) -> None:
        time_inputs[key] = value
        axis_spec[key] = axis

    zeros_T = jnp.zeros((T,), dtype=dtype)
    ones_T = jnp.ones((T,), dtype=dtype)
    inf_T = jnp.full((T,), jnp.inf, dtype=dtype)

    register("y", y, 0)
    register("mu_X", mu_X, 0)
    register("sigma2_X", sigma2_X, 0)
    register("proposal_scale", proposal_scale, 0)
    register("proposal_mean", mu_X, 0)
    register("proposal_var", sigma2_X, 0)
    register("growth_path", growth_path, 0)
    register("eps_b", eps_b, 0)
    register("obs_base", zeros_T, 0)
    register("bubble_scale", ones_T, 0)
    register("eps_b_scale", zeros_T, 0)
    register("obs_shift", zeros_T, 0)
    register("obs_scale", ones_T, 0)
    register("obs_df", inf_T, 0)
    register("chi_loc", zeros_T, 0)
    register("chi_scale", ones_T, 0)
    register("chi_lower", jnp.full((T,), -jnp.inf, dtype=dtype), 0)
    register("chi_upper", jnp.full((T,), jnp.inf, dtype=dtype), 0)
    register("chi_linear", zeros_T, 0)
    register("chi_constant", zeros_T, 0)
    register("chi_cross", zeros_T, 0)

    chi_fn = chi_integrand
    base_dict = dict(base_inputs or {})
    if "chi_integrand" in base_dict:
        raw = base_dict.pop("chi_integrand")
        if isinstance(raw, Partial):
            chi_fn = raw
        elif callable(raw):
            chi_fn = Partial(raw)
        else:
            raise TypeError("chi_integrand must be callable when provided.")
    register("chi_integrand", chi_fn, None)

    for key, value in base_dict.items():
        if isinstance(value, Partial):
            register(key, value, None)
            continue
        if callable(value):
            register(key, Partial(value), None)
            continue
        arr = jnp.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            arr = jnp.broadcast_to(arr, (T,))
            register(key, arr, 0)
            continue
        if arr.shape[0] != T:
            raise ValueError(
                f"Per-time input '{key}' must have leading dimension {T}, received {arr.shape}."
            )
        register(key, arr, 0)

    return time_inputs, axis_spec


def log_weight_one_s(
    theta: BubbleParams,
    t_inputs: Mapping[str, Any],
    X1_s: Array,
    z_b_s: Array,
    gh_nodes: int,
    use_truncation: bool,
) -> Array:
    """Return log importance weights for a single time index ``s``."""

    dtype = X1_s.dtype
    y_s = jnp.asarray(t_inputs["y"], dtype=dtype)
    growth = jnp.asarray(t_inputs["growth_path"], dtype=dtype)
    obs_base = jnp.asarray(t_inputs["obs_base"], dtype=dtype)
    bubble_scale = jnp.asarray(t_inputs["bubble_scale"], dtype=dtype)
    eps_scale = jnp.asarray(t_inputs["eps_b_scale"], dtype=dtype)
    eps_b = jnp.asarray(t_inputs["eps_b"], dtype=dtype)
    obs_shift = jnp.asarray(t_inputs["obs_shift"], dtype=dtype)

    B0 = jnp.asarray(theta.B0, dtype=dtype)
    mean_component = growth * B0 + obs_base + bubble_scale * X1_s + eps_scale * eps_b + obs_shift
    resid = y_s - mean_component

    df = jnp.asarray(t_inputs["obs_df"], dtype=dtype)
    scale = jnp.asarray(t_inputs["obs_scale"], dtype=dtype)
    scale = jnp.broadcast_to(scale, resid.shape)
    log_p_y = _student_t_logpdf(resid, df, scale)

    chi_term = integrate_over_chi(theta, t_inputs, X1_s, z_b_s, gh_nodes, use_truncation)

    mu_X = jnp.asarray(t_inputs["mu_X"], dtype=dtype)
    sigma2_X = jnp.asarray(t_inputs["sigma2_X"], dtype=dtype)
    log_q = _gaussian_logpdf(X1_s, mu_X, sigma2_X)

    return log_p_y + chi_term - log_q


def loglik_hat(
    theta_u: BubbleParamsUnconstrained,
    u: Dict[str, Array],
    data: BubbleData,
    cfg: PMHMCConfig,
    pre: Dict[str, Any],
) -> Tuple[Array, Dict[str, Any]]:
    """Pseudo-marginal log-likelihood estimator for the bubble block."""

    dtype = jnp.float64
    theta = constrain_params(theta_u)
    log_jac = _log_abs_det_jacobian(theta_u)

    T = int(data.T)
    N_is = int(cfg.N_is)

    steady_inputs = dict(pre.get("steady_inputs", {}))
    steady_inputs.setdefault("num_steps", T)
    mu_X_path, var_X_path = proposal_moments(theta, steady_inputs)

    z_b = _reshape_mc_draws(u["z_b"], T, N_is)
    event_shape = z_b.shape[2:]
    mu_X = _expand_time_array(mu_X_path, T, event_shape)
    sigma2_X = _expand_time_array(var_X_path, T, event_shape)
    sigma2_X = jnp.clip(sigma2_X, a_min=1e-12)
    proposal_scale = jnp.sqrt(sigma2_X)
    X1 = mu_X[:, None, ...] + proposal_scale[:, None, ...] * z_b

    if "eps_b" in u:
        eps_b = _reshape_mc_draws(u["eps_b"], T, N_is)
    else:
        eps_b = jnp.zeros_like(X1)

    y = jnp.asarray(data.y_net_fund, dtype=dtype)
    if y.shape[0] != T:
        raise ValueError(f"Observed series must have length {T}, received {y.shape}.")

    growth_path = _build_growth_path(T, pre, dtype)
    h_b_path, scale_path = _build_state_paths(theta, T, pre, dtype)

    chi_integrand_raw = pre.get("chi_integrand", _default_chi_integrand)
    if isinstance(chi_integrand_raw, Partial):
        chi_integrand = chi_integrand_raw
    elif callable(chi_integrand_raw):
        chi_integrand = Partial(chi_integrand_raw)
    else:
        raise TypeError("pre['chi_integrand'] must be callable when provided.")

    base_time_inputs = pre.get("time_inputs", {})
    time_inputs, axis_spec = _prepare_time_inputs(
        base_time_inputs,
        y,
        mu_X,
        sigma2_X,
        proposal_scale,
        X1,
        eps_b,
        growth_path,
        chi_integrand,
        dtype,
    )

    def weight_fn(t_in: Mapping[str, Any], X_s: Array, z_s: Array) -> Array:
        return log_weight_one_s(theta, t_in, X_s, z_s, cfg.gh_nodes, cfg.use_truncation)

    log_weights = jax.vmap(weight_fn, in_axes=(axis_spec, 0, 0))(time_inputs, X1, z_b)
    log_avg_weights = jsp_special.logsumexp(log_weights, axis=1) - jnp.log(float(N_is))
    loglik = log_jac + jnp.sum(log_avg_weights)

    info = {
        "log_jacobian": log_jac,
        "log_weights": log_weights,
        "log_avg_weights": log_avg_weights,
        "paths": {
            "h_b": h_b_path,
            "scale": scale_path,
            "growth": growth_path,
            "proposal_mean": mu_X,
            "proposal_var": sigma2_X,
            "proposal_scale": proposal_scale,
        },
        "theta": _as_theta_dict(theta),
        "X1": X1,
        "eps_b": eps_b,
        "z_b": z_b,
    }

    return loglik, info


__all__ = ["proposal_moments", "log_weight_one_s", "loglik_hat", "integrate_over_chi"]
