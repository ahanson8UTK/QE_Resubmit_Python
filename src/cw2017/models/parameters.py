"""Parameter objects and transformations for block 3a."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax.nn import softplus

from ..math.fill_q import count_qs_entries, fill_Q_jax
from ..math.ordered_bounded import ordered_bounded_descending

Array = jnp.ndarray


@dataclass
class Params3aUnconstrained:
    """Unconstrained representation used for HMC proposals."""

    log_omega: Array
    Lg_free: Array
    u_eigs_head: Array
    u_eigs_tail: Array
    qs_free: Array
    phiQ_toprow_raw: Array
    phiQ_sign_raw: Array
    gamma0_last_raw: Array
    gamma1_free_2elts: Array


@dataclass
class PhiPBlocks:
    Phi_gg: Array
    Phi_gm: Array
    Phi_gh: Array


@dataclass
class Params3a:
    """Constrained parameter collection after applying all transforms."""

    Omega_diag: Array
    L_g: Array
    Sigma_g: Array
    eigs_full: Array
    Q_full: Array
    PhiP_blocks: PhiPBlocks
    phiQ_toprow: Array
    Gamma0: Array
    Gamma1: Array


def unit_lower_from_free(free_entries: Array) -> Array:
    """Construct a unit lower-triangular matrix from its free entries."""

    free_entries = jnp.asarray(free_entries, dtype=jnp.float64)
    L = jnp.eye(3, dtype=jnp.float64)
    L = L.at[1, 0].set(free_entries[0])
    L = L.at[2, 0].set(free_entries[1])
    L = L.at[2, 1].set(free_entries[2])
    return L


def _partition_phi(PhiP: Array, info: Dict[str, int]) -> PhiPBlocks:
    Nm = int(info["Nm"])
    Ng = int(info["Ng"])
    Phi_gm = PhiP[Nm : Nm + Ng, :Nm]
    Phi_gg = PhiP[Nm : Nm + Ng, Nm : Nm + Ng]
    Phi_gh = PhiP[Nm : Nm + Ng, Nm + Ng :]
    return PhiPBlocks(Phi_gg=Phi_gg, Phi_gm=Phi_gm, Phi_gh=Phi_gh)


def constrain_params3a(
    u: Params3aUnconstrained,
    cfg: Dict[str, float],
    info: Dict[str, int],
) -> Tuple[Params3a, float]:
    """Map unconstrained parameters to the constrained space required by block 3a."""

    rho_max = float(cfg.get("rho_max", 0.995))
    Nm = int(info["Nm"])
    Ng = int(info["Ng"])
    Nh = int(info["Nh"])
    Nstates = int(info["Nstates"])

    Omega_diag = softplus(jnp.asarray(u.log_omega, dtype=jnp.float64)) + 1e-12

    L_g = unit_lower_from_free(u.Lg_free)
    Sigma_g = L_g @ L_g.T

    eigs_head = ordered_bounded_descending(u.u_eigs_head, -rho_max, rho_max)
    eigs_tail = ordered_bounded_descending(u.u_eigs_tail, -rho_max, rho_max)
    eigs_full = jnp.concatenate([eigs_head, eigs_tail], axis=0)

    Q_full = fill_Q_jax(eigs_full, u.qs_free, info)
    Lambda = jnp.diag(eigs_full)
    PhiP = Q_full @ Lambda @ jsp.solve(Q_full, jnp.eye(Nstates, dtype=jnp.float64))
    phi_blocks = _partition_phi(PhiP, info)

    phi_raw = jnp.asarray(u.phiQ_toprow_raw, dtype=jnp.float64)
    phi11, phi12, phi13_mag = phi_raw
    sign13 = jnp.tanh(jnp.asarray(u.phiQ_sign_raw, dtype=jnp.float64))
    phi13 = sign13 * softplus(phi13_mag)
    phiQ_toprow = jnp.stack([phi11, phi12, phi13])[None, :]

    Gamma0 = jnp.zeros((Nm + Ng,), dtype=jnp.float64)
    Gamma0 = Gamma0.at[-1].set(jnp.asarray(u.gamma0_last_raw, dtype=jnp.float64))

    rows = Nm + Ng
    cols = Nh
    Gamma1 = jnp.zeros((rows, cols), dtype=jnp.float64)
    Gamma1 = Gamma1.at[: Nm + 1, : Nm + 1].set(1200.0 * jnp.eye(Nm + 1, dtype=jnp.float64))
    row_idx = Nm + 1
    Gamma1 = Gamma1.at[row_idx, cols - 2 : cols].set(jnp.asarray(u.gamma1_free_2elts, dtype=jnp.float64))
    Gamma1 = Gamma1.at[rows - 1, cols - 1].set(1200.0)

    params = Params3a(
        Omega_diag=Omega_diag,
        L_g=L_g,
        Sigma_g=Sigma_g,
        eigs_full=eigs_full,
        Q_full=Q_full,
        PhiP_blocks=phi_blocks,
        phiQ_toprow=phiQ_toprow,
        Gamma0=Gamma0,
        Gamma1=Gamma1,
    )
    return params, 0.0


__all__ = [
    "Params3aUnconstrained",
    "Params3a",
    "PhiPBlocks",
    "unit_lower_from_free",
    "constrain_params3a",
    "count_qs_entries",
]
