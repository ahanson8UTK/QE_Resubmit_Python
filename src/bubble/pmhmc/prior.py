"""Prior helpers for the bubble PM-HMC module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import betaln
import numpy as np
from numpy.typing import NDArray

from .types import BubbleParams


#: Hyper-parameters for the weakly-informative prior used by :func:`log_prior`.
#: Tweak these constants to shift the prior mass while preserving differentiability
#: of the resulting log-density.
B0_LOGNORMAL_MEANLOG = float(jnp.log(1.0))
B0_LOGNORMAL_SDLOG = 1.0
MU_B_NORMAL_LOC = 0.0
MU_B_NORMAL_SCALE = 1.0
PHI_B_BETA_ALPHA = 8.0
PHI_B_BETA_BETA = 2.0
SIGMA_H_HALF_NORMAL_SCALE = 1.0
RHO_TARGET_RADIUS = 0.5

LogScalarPrior = Callable[[float], float]
LogRhoPrior = Callable[[NDArray[np.float_], NDArray[np.float_]], float]


def _zero_scalar(_: float) -> float:
    return 0.0


def _zero_rho(_: NDArray[np.float_], __: NDArray[np.float_]) -> float:
    return 0.0


def gaussian_logpdf(value: float, *, mean: float = 0.0, scale: float = 1.0) -> float:
    """Evaluate the log-density of a univariate Gaussian distribution."""

    if scale <= 0.0:
        raise ValueError("Scale must be strictly positive for a Gaussian prior.")
    variance = scale ** 2
    return -0.5 * (np.log(2.0 * np.pi * variance) + ((value - mean) ** 2) / variance)


def isotropic_gaussian_logpdf(
    rho_bm: NDArray[np.float_],
    rho_bg: NDArray[np.float_],
    *,
    scale: float = 1.0,
) -> float:
    """Isotropic Gaussian log prior for the concatenated correlation vector."""

    if scale <= 0.0:
        raise ValueError("Scale must be strictly positive for the isotropic Gaussian prior.")
    vec = np.concatenate((np.asarray(rho_bm, dtype=float), np.asarray(rho_bg, dtype=float)))
    variance = scale ** 2
    quad_form = float(np.dot(vec, vec))
    dim = vec.size
    return -0.5 * (dim * np.log(2.0 * np.pi * variance) + quad_form / variance)


def log_prior(theta: BubbleParams) -> jnp.ndarray:
    """Evaluate the log-prior on constrained bubble parameters.

    The prior combines weakly-informative marginals that respect the
    support of each constrained parameter:

    ``B0``
        Log-normal with parameters ``(meanlog=B0_LOGNORMAL_MEANLOG,``
        ``sdlog=B0_LOGNORMAL_SDLOG)`` to enforce positivity.
    ``mu_b``
        Standard normal centred at ``MU_B_NORMAL_LOC`` with scale
        ``MU_B_NORMAL_SCALE``.
    ``phi_b``
        A transformed Beta prior defined through ``phi = 2 * Beta(α, β) - 1``
        with hyper-parameters ``(PHI_B_BETA_ALPHA, PHI_B_BETA_BETA)`` placing
        most mass on persistent yet stationary dynamics.
    ``sigma_h``
        Half-normal with scale ``SIGMA_H_HALF_NORMAL_SCALE`` to maintain
        positivity of the volatility level.
    ``rho``
        The concatenated correlation vector receives a ridge-like penalty
        ``-0.5 * (‖rho‖ / RHO_TARGET_RADIUS)²`` nudging its length toward the
        interior of the unit ball while keeping the implicit uniform prior on
        directions induced by the re-normalisation transform.

    Adjusting the module-level constants documented above alters the prior in
    a transparent way and keeps the log-density differentiable for use with
    gradient-based samplers.
    """

    B0 = jnp.asarray(theta.B0, dtype=jnp.float64)
    mu_b = jnp.asarray(theta.mu_b, dtype=jnp.float64)
    phi_b = jnp.asarray(theta.phi_b, dtype=jnp.float64)
    sigma_h = jnp.asarray(theta.sigma_h, dtype=jnp.float64)
    rho_bm = jnp.asarray(theta.rho_bm, dtype=jnp.float64)
    rho_bg = jnp.asarray(theta.rho_bg, dtype=jnp.float64)

    # Log-normal prior for B0 (positive support)
    safe_B0 = jnp.clip(B0, a_min=jnp.finfo(B0.dtype).tiny, a_max=None)
    log_B0_density = (
        -jnp.log(safe_B0)
        - jnp.log(B0_LOGNORMAL_SDLOG)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5
        * ((jnp.log(safe_B0) - B0_LOGNORMAL_MEANLOG) / B0_LOGNORMAL_SDLOG) ** 2
    )
    log_B0_density = jnp.where(B0 > 0.0, log_B0_density, -jnp.inf)

    # Gaussian prior for mu_b
    log_mu_density = -0.5 * (
        jnp.log(2.0 * jnp.pi * MU_B_NORMAL_SCALE**2)
        + ((mu_b - MU_B_NORMAL_LOC) / MU_B_NORMAL_SCALE) ** 2
    )

    # Beta prior mapped to (-1, 1) for phi_b
    u = 0.5 * (phi_b + 1.0)
    eps = jnp.finfo(phi_b.dtype).eps
    safe_u = jnp.clip(u, a_min=jnp.finfo(phi_b.dtype).tiny, a_max=1.0 - eps)
    log_phi_density = (
        (PHI_B_BETA_ALPHA - 1.0) * jnp.log(safe_u)
        + (PHI_B_BETA_BETA - 1.0) * jnp.log1p(-safe_u)
        - betaln(PHI_B_BETA_ALPHA, PHI_B_BETA_BETA)
        - jnp.log(2.0)
    )
    log_phi_density = jnp.where(
        (phi_b > -1.0) & (phi_b < 1.0), log_phi_density, -jnp.inf
    )

    # Half-normal prior for sigma_h (positive scale parameter)
    safe_sigma = jnp.clip(
        sigma_h, a_min=jnp.finfo(sigma_h.dtype).tiny, a_max=None
    )
    log_sigma_density = (
        0.5 * jnp.log(2.0 / jnp.pi)
        - jnp.log(SIGMA_H_HALF_NORMAL_SCALE)
        - 0.5 * (safe_sigma / SIGMA_H_HALF_NORMAL_SCALE) ** 2
    )
    log_sigma_density = jnp.where(sigma_h > 0.0, log_sigma_density, -jnp.inf)

    # Ridge penalty encouraging rho to stay away from the boundary of the unit ball
    rho_vec = jnp.concatenate((rho_bm.ravel(), rho_bg.ravel()))
    rho_norm = jnp.linalg.norm(rho_vec)
    rho_penalty = -0.5 * (rho_norm / RHO_TARGET_RADIUS) ** 2

    return log_B0_density + log_mu_density + log_phi_density + log_sigma_density + rho_penalty


@dataclass(frozen=True)
class BubblePrior:
    """Composable log prior for :class:`~bubble.pmhmc.types.BubbleParams`."""

    log_B0: LogScalarPrior = field(default=_zero_scalar)
    log_mu_b: LogScalarPrior = field(default=_zero_scalar)
    log_phi_b: LogScalarPrior = field(default=_zero_scalar)
    log_sigma_h: LogScalarPrior = field(default=_zero_scalar)
    log_rho: LogRhoPrior = field(default=_zero_rho)

    def log_prob(self, params: BubbleParams) -> float:
        """Return the total log prior probability for ``params``."""

        return (
            self.log_B0(params.B0)
            + self.log_mu_b(params.mu_b)
            + self.log_phi_b(params.phi_b)
            + self.log_sigma_h(params.sigma_h)
            + self.log_rho(params.rho_bm, params.rho_bg)
        )

    __call__ = log_prob

    @classmethod
    def flat(cls) -> BubblePrior:
        """Create a prior with zero contribution from every parameter."""

        return cls()

    @classmethod
    def gaussian(
        cls,
        *,
        B0_scale: float = 1.0,
        mu_b_scale: float = 1.0,
        phi_b_scale: float = 1.0,
        sigma_h_scale: float = 1.0,
        rho_scale: float = 1.0,
    ) -> BubblePrior:
        """Construct a simple independent Gaussian prior for demonstration."""

        return cls(
            log_B0=lambda x: gaussian_logpdf(x, scale=B0_scale),
            log_mu_b=lambda x: gaussian_logpdf(x, scale=mu_b_scale),
            log_phi_b=lambda x: gaussian_logpdf(x, scale=phi_b_scale),
            log_sigma_h=lambda x: gaussian_logpdf(x, scale=sigma_h_scale),
            log_rho=lambda bm, bg: isotropic_gaussian_logpdf(bm, bg, scale=rho_scale),
        )


__all__ = [
    "BubblePrior",
    "gaussian_logpdf",
    "isotropic_gaussian_logpdf",
    "log_prior",
]
