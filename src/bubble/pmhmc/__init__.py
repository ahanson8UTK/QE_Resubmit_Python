"""Scaffolding for the bubble particle MCMC (PM-HMC) package."""
from __future__ import annotations

from .likelihood import (
    integrate_over_chi,
    log_weight_one_s,
    loglik_hat,
    proposal_moments,
)

from .prior import BubblePrior, gaussian_logpdf, isotropic_gaussian_logpdf
from .transforms import (
    inv_softplus,
    renorm_rho,
    softplus,
    tanh_to_interval,
    unconstrained_to_constrained,
)
from .types import (
    BubbleData,
    BubbleParams,
    BubbleParamsUnconstrained,
    PMHMCConfig,
    PMHMCResult,
)

__all__ = [
    "BubbleData",
    "BubbleParams",
    "BubbleParamsUnconstrained",
    "BubblePrior",
    "draw_bubble_block",
    "PMHMCConfig",
    "PMHMCResult",

    "integrate_over_chi",
    "log_weight_one_s",
    "loglik_hat",
    "proposal_moments",
    "artanh_unconstrain",
    "constrain_params",
    "from_unit_ball",

    "gaussian_logpdf",
    "inv_softplus",
    "isotropic_gaussian_logpdf",
    "renorm_rho",
    "softplus",
    "tanh_to_interval",
    "unconstrained_to_constrained",
]
