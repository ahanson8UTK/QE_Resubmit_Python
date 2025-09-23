"""Core type definitions for the bubble particle MCMC module.

The classes defined here capture the observed data, algorithmic
configuration, model parameters in both constrained and unconstrained
spaces, and the structure of the returned results.  These are kept
dataclass-based so that they play nicely with static type checking and
can be easily serialized in later development stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray


@dataclass
class BubbleData:
    """Container for the observable inputs required by the bubble model.

    Attributes
    ----------
    T:
        Number of time periods in the panel.
    y_net_fund:
        Observed net-fundamental price series :math:`y_t = V_t - p^{d}_t D_t`
        with shape ``(T,)``.
    r_t:
        Short rate series :math:`r_t = g_{1, t}` with shape ``(T,)``.
    m_t:
        Factor loadings for the ``m`` block with shape ``(T, d_m)``.
    g_t:
        Factor loadings for the ``g`` block with shape ``(T, d_g)``.
    h_t:
        Latent volatility states with shape ``(T, d_h)``.
    Dm_t:
        Diagonal entries used for the ``m`` block dynamics at each
        ``t`` with shape ``(T, d_m)``.
    Dg_t:
        Diagonal entries used for the ``g`` block dynamics at each
        ``t`` with shape ``(T, d_g)``.
    Sigma_m:
        Lower-triangular matrix with unit diagonal elements describing
        factor correlations for the ``m`` block, shape ``(d_m, d_m)``.
    Sigma_g:
        Lower-triangular Cholesky factor of :math:`\Sigma_g` with
        shape ``(d_g, d_g)``.
    Sigma_gm:
        Cross-block covariance matrix between ``g`` and ``m`` with shape
        ``(d_g, d_m)``.
    """

    T: int
    y_net_fund: NDArray[np.float_]
    r_t: NDArray[np.float_]
    m_t: NDArray[np.float_]
    g_t: NDArray[np.float_]
    h_t: NDArray[np.float_]
    Dm_t: NDArray[np.float_]
    Dg_t: NDArray[np.float_]
    Sigma_m: NDArray[np.float_]
    Sigma_g: NDArray[np.float_]
    Sigma_gm: NDArray[np.float_]


@dataclass
class PMHMCConfig:
    """Configuration options controlling the PM-HMC sampler."""

    gh_nodes: int
    N_is: int
    seed: int
    hmc_num_warmup: int
    hmc_num_samples: int
    hmc_step_size: float
    hmc_num_steps: int
    max_tree_depth: int
    use_truncation: bool


@dataclass
class BubbleParams:
    """Model parameters in their constrained (interpretable) space."""

    B0: float
    mu_b: float
    phi_b: float
    sigma_h: float
    rho_bm: NDArray[np.float_]
    rho_bg: NDArray[np.float_]


@dataclass
class BubbleParamsUnconstrained:
    """Model parameters expressed on an unconstrained real-valued space."""

    z_B0: float
    z_mu_b: float
    z_phi_b: float
    z_sigma_h: float
    z_rho_bm: NDArray[np.float_]
    z_rho_bg: NDArray[np.float_]


@dataclass
class PMHMCResult:
    """Result bundle returned by future PM-HMC inference routines."""

    params: BubbleParams
    log_like_hat: float
    accept_rate: float
    draws_constrained: Dict[str, NDArray[np.float_]]
    diagnostics: Dict[str, Any]


__all__ = [
    "BubbleData",
    "BubbleParams",
    "BubbleParamsUnconstrained",
    "PMHMCConfig",
    "PMHMCResult",
]
