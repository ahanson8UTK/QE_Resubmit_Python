"""Gauss--Hermite helpers for integrating over the auxiliary ``\chi`` shocks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from .types import BubbleParams

Array = jnp.ndarray


def gauss_hermite_nodes(n: int) -> Tuple[Array, Array]:
    """Return Gauss--Hermite nodes and weights for a standard normal expectation.

    The helper wraps :func:`numpy.polynomial.hermite_e.hermegauss` which delivers
    nodes and weights that are orthonormal with respect to ``exp(-x^2 / 2)``.  We
    rescale the weights so they integrate functions under ``N(0, 1)`` exactly in
    the polynomial limit.  The returned arrays live on the JAX device and are
    explicitly promoted to ``float64`` because the Kalman and pseudo-marginal
    routines operate in double precision throughout the project.
    """

    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    norm_const = np.sqrt(2.0 * np.pi)
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    weights = jnp.asarray(weights / norm_const, dtype=jnp.float64)
    return nodes, weights


def mu_star_sigma2(
    theta: BubbleParams,
    t_inputs: Dict[str, Array],
    chi: Array,
) -> Tuple[Array, Array]:
    """Conditional moments ``(\mu_t^*(\chi), \sigma_t^2(\chi))`` for the bubble shock.

    The formulas mirror the derivation used for the pseudo-marginal estimator.
    Denote the block innovations by ``\xi_m`` and ``\xi_g`` so that

    .. math::

        \xi_m &= \Sigma_m^{-1} D_{m,t}^{-1} (m_{t+1} - \mu_{m,t}),\\
        \xi_g &= \Sigma_g^{-1} \Bigl(D_{g,t}^{-1} (g_{t+1} - \mu_{g,t})
                  - \Sigma_{gm} \xi_m\Bigr).

    With correlation vectors ``\rho_{bm}`` and ``\rho_{bg}`` we express the
    bubble innovation as

    .. math::

        \varepsilon_{b,t+1}
            = \rho_{bm}^{\top} \xi_m
            + \rho_{bg}^{\top} \xi_g
            + \sqrt{1 - \Vert (\rho_{bm}, \rho_{bg}) \Vert^2}\, \chi.

    The conditional moments follow immediately from the affine state equation

    .. math::

        B_{t+1} = (1 - \phi_b) \mu_b + \phi_b B_t + \sigma_h \varepsilon_{b,t+1},

    which yields

    .. math::

        \mu_t^*(\chi) &= (1 - \phi_b) \mu_b + \phi_b B_t + r_t
            + \sigma_h \bigl(\rho_{bm}^{\top} \xi_m + \rho_{bg}^{\top} \xi_g\bigr)
            + \sigma_h \sqrt{1 - \Vert (\rho_{bm}, \rho_{bg}) \Vert^2}\, \chi,\\
        \sigma_t^2(\chi) &= \sigma_h^2
            \bigl(1 - \Vert (\rho_{bm}, \rho_{bg}) \Vert^2\bigr).

    The ``D_{m,t}`` and ``D_{g,t}`` diagonals are clipped away from zero before
    inversion and the orthogonal complement variance is floored at ``1e-12``.  We
    broadcast the variance across the supplied ``\chi`` nodes so the caller can
    combine it with other terms elementwise when building the log-likelihood.
    """

    chi = jnp.asarray(chi, dtype=jnp.float64)

    phi_b = jnp.asarray(theta.phi_b, dtype=jnp.float64)
    mu_b = jnp.asarray(theta.mu_b, dtype=jnp.float64)
    sigma_h = jnp.asarray(theta.sigma_h, dtype=jnp.float64)
    rho_bm = jnp.asarray(theta.rho_bm, dtype=jnp.float64)
    rho_bg = jnp.asarray(theta.rho_bg, dtype=jnp.float64)

    r_t = jnp.asarray(t_inputs.get("r_t", 0.0), dtype=jnp.float64)
    bubble_t = jnp.asarray(t_inputs.get("bubble_t", 0.0), dtype=jnp.float64)
    Dm_t = jnp.asarray(t_inputs["Dm_t"], dtype=jnp.float64)
    Dg_t = jnp.asarray(t_inputs["Dg_t"], dtype=jnp.float64)
    Sigma_m = jnp.asarray(t_inputs["Sigma_m"], dtype=jnp.float64)
    Sigma_g = jnp.asarray(t_inputs["Sigma_g"], dtype=jnp.float64)
    Sigma_gm = jnp.asarray(t_inputs["Sigma_gm"], dtype=jnp.float64)
    resid_m = jnp.asarray(t_inputs["resid_m"], dtype=jnp.float64)
    resid_g = jnp.asarray(t_inputs["resid_g"], dtype=jnp.float64)

    safe_Dm = jnp.clip(Dm_t, a_min=1e-12)
    safe_Dg = jnp.clip(Dg_t, a_min=1e-12)

    xi_m = jsp_linalg.solve_triangular(
        Sigma_m, resid_m / safe_Dm, lower=True, check_finite=False
    )
    xi_g = jsp_linalg.solve_triangular(
        Sigma_g,
        resid_g / safe_Dg - Sigma_gm @ xi_m,
        lower=True,
        check_finite=False,
    )

    rho_concat = jnp.concatenate((rho_bm, rho_bg))
    rho_sq = jnp.sum(rho_concat**2)
    rho_sq = jnp.clip(rho_sq, a_min=0.0, a_max=1.0 - 1e-12)
    orth_scale = jnp.sqrt(1.0 - rho_sq)

    drift = (1.0 - phi_b) * mu_b + phi_b * bubble_t + r_t
    correlated = sigma_h * (rho_bm @ xi_m + rho_bg @ xi_g)
    mu_star = drift + correlated + sigma_h * orth_scale * chi

    sigma2_base = jnp.maximum(sigma_h**2 * (1.0 - rho_sq), 1e-12)
    sigma2 = jnp.broadcast_to(sigma2_base, chi.shape)

    return mu_star, sigma2


def integrate_over_chi(
    log_num: Callable[..., Array],
    args: Tuple[Any, ...],
    gh_nodes: int,
    *,
    truncation_adjustment: bool = False,
    truncation_z: Array | None = None,
) -> Array:
    """Integrate ``exp(log_num(\chi))`` against ``N(0, 1)`` with Gauss--Hermite nodes.

    Parameters
    ----------
    log_num:
        Callable that accepts a quadrature node and the provided ``args`` tuple
        and returns the log-integrand evaluated at that node.
    args:
        Additional positional arguments forwarded to ``log_num``.
    gh_nodes:
        Number of Gauss--Hermite quadrature nodes.
    truncation_adjustment:
        When ``True`` multiply the result by ``1 - \Phi(z)`` in log-space.  The
        tail probability is evaluated via :func:`jax.scipy.stats.norm.logcdf`
        using the identity ``\log(1 - \Phi(z)) = \log \Phi(-z)`` for numerical
        stability.
    truncation_z:
        Argument passed to the Gaussian CDF when ``truncation_adjustment`` is
        active.  A ``ValueError`` is raised when the adjustment is requested
        without a corresponding ``z`` value.
    """

    nodes, weights = gauss_hermite_nodes(gh_nodes)
    log_weights = jnp.log(weights)

    def _eval(node: Array) -> Array:
        return log_num(node, *args)

    log_vals = jax.vmap(_eval)(nodes)
    reshape = (nodes.shape[0],) + (1,) * max(log_vals.ndim - 1, 0)
    log_integrand = log_vals + log_weights.reshape(reshape)
    log_result = logsumexp(log_integrand, axis=0)

    if truncation_adjustment:
        if truncation_z is None:
            raise ValueError("truncation_z must be provided when truncation_adjustment is True")
        log_result = log_result + norm.logcdf(-jnp.asarray(truncation_z, dtype=jnp.float64))

    return log_result


__all__ = ["gauss_hermite_nodes", "mu_star_sigma2", "integrate_over_chi"]
