"""Durbin–Koopman simulation smoother for the :math:`g`-block.

The implementation follows the linear state-space specification described in
Section 1.3 of ``docs/model_writeup`` and the Gibbs block construction outlined
in Section 3.3 of ``docs/algorithm_overview``.  Conditioning on the current
draws of :math:`h_t` and all structural parameters, we run the square-root
Kalman filter forward pass and subsequently apply the Durbin–Koopman simulation
smoother (Form I) to obtain a draw of the latent :math:`g_{1:T}` path together
with updated long-run means :math:`(\bar{\mu}_m, \bar{\mu}_g^u,\bar{\mu}_g^{\mathbb{Q},u})`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp

from kalman.ffbs import Step, dk_sample, kf_forward_sr_wrapper

from ..typing import Array, PRNGKey


@dataclass(frozen=True)
class StateSpaceSpec:
    """Container describing the augmented state-space model.

    Attributes
    ----------
    steps:
        Callable or sequence returning :class:`~kalman.ffbs.Step` instances for
        each time period.
    x0_mean, x0_cov:
        Mean and covariance of the initial state.
    slices:
        Mapping exposing slices for the :math:`g` block and the static means.
    """

    steps: Iterable[Step] | Sequence[Step] | Any
    x0_mean: Array
    x0_cov: Array
    slices: Dict[str, slice]


def _as_array(x: Any) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.float64)


def _stack_observations(m_t: Array | None, y_t: Array | None, h_t: Array | None) -> jnp.ndarray:
    """Stack observable blocks ``(m_t, y_t, h_t)`` into a single matrix."""

    pieces = []
    for block in (m_t, y_t, h_t):
        if block is None:
            continue
        arr = _as_array(block)
        if arr.ndim != 2:
            raise ValueError("Each observation block must be two-dimensional")
        pieces.append(arr)

    if not pieces:
        raise ValueError("At least one observation block is required")

    obs = jnp.concatenate(pieces, axis=1)
    return obs


def _resolve_spec(params: Any, obs: jnp.ndarray, h_t: Array | None, m_t: Array | None) -> StateSpaceSpec:
    """Return the state-space specification used by the simulation smoother."""

    mapping: Mapping[str, Any] | None = None
    if isinstance(params, Mapping):
        mapping = params
    elif hasattr(params, "__dict__"):
        mapping = params.__dict__  # type: ignore[assignment]

    spec_obj: Any = None
    if mapping is not None:
        if "state_space" in mapping:
            spec_obj = mapping["state_space"]
        elif "build_state_space" in mapping:
            builder = mapping["build_state_space"]
            spec_obj = builder(params=params, y_t=obs, h_t=h_t, m_t=m_t)
    elif hasattr(params, "build_state_space"):
        spec_obj = params.build_state_space(params=params, y_t=obs, h_t=h_t, m_t=m_t)

    if spec_obj is None:
        raise ValueError(
            "A state-space specification with keys 'steps', 'x0_mean', 'x0_cov',"
            " and 'slices' must be provided via `params`."
        )

    if isinstance(spec_obj, StateSpaceSpec):
        return spec_obj

    if isinstance(spec_obj, Mapping):
        steps = spec_obj.get("steps")
        x0_mean = spec_obj.get("x0_mean")
        x0_cov = spec_obj.get("x0_cov")
        slices = dict(spec_obj.get("slices", {}))
    elif isinstance(spec_obj, tuple) and len(spec_obj) == 4:
        steps, x0_mean, x0_cov, slices = spec_obj
        slices = dict(slices)
    else:
        raise TypeError("Unsupported state-space specification format")

    if steps is None or x0_mean is None or x0_cov is None:
        raise ValueError("State-space specification is missing required fields")

    return StateSpaceSpec(steps=steps, x0_mean=_as_array(x0_mean), x0_cov=_as_array(x0_cov), slices=slices)


def _extract_means(sample: jnp.ndarray, slices: Dict[str, slice]) -> jnp.ndarray:
    """Return the concatenated static means from the state trajectory."""

    mu_m_slice = slices.get("mu_m", slice(0, 0))
    mu_gu_slice = slices.get("mu_g_u", slice(mu_m_slice.stop, mu_m_slice.stop))
    mu_gq_slice = slices.get("mu_gQ_u", slice(mu_gu_slice.stop, mu_gu_slice.stop))

    def _slice(arr: jnp.ndarray, sl: slice) -> jnp.ndarray:
        if sl.stop <= sl.start:
            return jnp.asarray([], dtype=arr.dtype)
        return arr[..., sl]

    mu_m = _slice(sample, mu_m_slice)
    mu_gu = _slice(sample, mu_gu_slice)
    mu_gq = _slice(sample, mu_gq_slice)

    return jnp.concatenate([mu_m, mu_gu, mu_gq], axis=-1)


def dk_simulation_smoother(
    params: Any,
    h_t: Array,
    y_t: Array,
    m_t: Array,
    key: PRNGKey,
) -> Tuple[Array, jnp.ndarray]:
    """Draw :math:`g_{1:T}` and the long-run means using the DK smoother."""

    h_obs = _as_array(h_t) if h_t is not None else None
    m_obs = _as_array(m_t) if m_t is not None else None
    y_obs = _as_array(y_t) if y_t is not None else None

    stacked_obs = _stack_observations(m_obs, y_obs, h_obs)
    spec = _resolve_spec(params, stacked_obs, h_obs, m_obs)

    cache = kf_forward_sr_wrapper(
        stacked_obs,
        spec.steps,
        spec.x0_mean,
        spec.x0_cov,
        exact_tol=0.0,
    )

    g_slice = spec.slices.get("g", slice(0, 0))
    key, draw_key = jax.random.split(key)
    full_sample = dk_sample(draw_key, stacked_obs, spec.steps, spec.x0_mean, spec.x0_cov, cache)

    if g_slice.stop <= g_slice.start:
        raise ValueError("State-space specification must expose a non-empty 'g' slice")

    g_sample = full_sample[:, g_slice]
    static_slices = spec.slices.get("static_all")
    if static_slices is not None and static_slices.stop > static_slices.start:
        static_segment = full_sample[:, static_slices][-1]
    else:
        static_segment = full_sample[-1]

    means = _extract_means(static_segment, spec.slices)
    return g_sample, means


__all__ = ["dk_simulation_smoother"]
