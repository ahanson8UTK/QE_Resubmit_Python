"""Prior helpers for the bubble PM-HMC module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .types import BubbleParams

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
]
