"""High-level Gibbs block wrapper for the bubble PM-HMC update."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .types import (
    BubbleData,
    BubbleParamsUnconstrained,
    PMHMCConfig,
    PMHMCResult,
)


def _default_unconstrained_params(data: BubbleData) -> BubbleParamsUnconstrained:
    """Construct default unconstrained parameters based on the data dimensions."""

    d_m = int(data.m_t.shape[1]) if data.m_t.ndim > 1 else 1
    d_g = int(data.g_t.shape[1]) if data.g_t.ndim > 1 else 1
    return BubbleParamsUnconstrained(
        z_B0=0.0,
        z_mu_b=0.0,
        z_phi_b=0.0,
        z_sigma_h=0.0,
        z_rho_bm=np.zeros(d_m, dtype=float),
        z_rho_bg=np.zeros(d_g, dtype=float),
    )


def _copy_unconstrained_params(
    params: BubbleParamsUnconstrained,
) -> BubbleParamsUnconstrained:
    """Return a defensive copy of unconstrained bubble parameters."""

    return BubbleParamsUnconstrained(
        z_B0=float(params.z_B0),
        z_mu_b=float(params.z_mu_b),
        z_phi_b=float(params.z_phi_b),
        z_sigma_h=float(params.z_sigma_h),
        z_rho_bm=np.array(params.z_rho_bm, dtype=float, copy=True),
        z_rho_bg=np.array(params.z_rho_bg, dtype=float, copy=True),
    )


def draw_bubble_block(
    data: BubbleData,
    cfg: PMHMCConfig,
    init: Optional[BubbleParamsUnconstrained],
    precomputed: Dict[str, object],
) -> PMHMCResult:
    """Draw the bubble block parameters using the PM-HMC sampler.

    The ``precomputed`` mapping should contain cached quantities from the
    surrounding Gibbs sweep such as the :math:`\Lambda_{g, t}` loadings with
    shape ``(T, d_g)`` in addition to any fixed scalars that would otherwise be
    reassembled inside the sampler.
    """

    theta_u = (
        _copy_unconstrained_params(init)
        if init is not None
        else _default_unconstrained_params(data)
    )

    from . import sampler

    result = sampler.run_pmhmc(data, cfg, theta_u, precomputed)
    return result


__all__ = ["draw_bubble_block"]
