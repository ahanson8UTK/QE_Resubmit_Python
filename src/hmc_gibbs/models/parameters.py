"""Parameter containers and reparameterisation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from ..typing import Array
from ..math import transforms


@dataclass
class MeasurementErrorDiag:
    """Diagonal measurement error covariance matrix Ω."""

    omega_diag: Array

    def to_unconstrained(self) -> Tuple[Array, jnp.float64]:
        unconstrained = jnp.log(self.omega_diag)
        logdet = jnp.sum(unconstrained)
        return unconstrained, logdet

    @classmethod
    def from_unconstrained(cls, raw: Array) -> Tuple["MeasurementErrorDiag", jnp.float64]:
        omega = jnp.exp(raw)
        return cls(omega_diag=omega), jnp.sum(raw)


@dataclass
class SigmaGLowerTri:
    """Lower-triangular factor of Σ_g with unit diagonal."""

    subdiag: Array
    dimension: int

    def to_matrix(self) -> Array:
        lt = jnp.eye(self.dimension)
        tril_indices = jnp.tril_indices(self.dimension, k=-1)
        lt = lt.at[tril_indices].set(self.subdiag)
        return lt

    def to_unconstrained(self) -> Tuple[Array, jnp.float64]:
        return self.subdiag, jnp.array(0.0)

    @classmethod
    def from_unconstrained(cls, raw: Array, dimension: int) -> Tuple["SigmaGLowerTri", jnp.float64]:
        return cls(subdiag=raw, dimension=dimension), jnp.array(0.0)


@dataclass
class EigenstructurePMeasure:
    """P-measure eigenvalues λ^f and eigenvectors Q."""

    lambda_f: Array
    q_free: Array

    def ordered_lambda(self) -> Array:
        return transforms.ordered_eigenvalues_transform(self.lambda_f)

    def to_unconstrained(self) -> Tuple[Array, jnp.float64]:
        return jnp.concatenate([self.lambda_f, self.q_free]), jnp.array(0.0)


@dataclass
class QMeasureRestrictions:
    """Top-row parameters of the Q-measure dynamics."""

    top_row: Array
    eigenvalues: Array

    def to_unconstrained(self) -> Tuple[Array, jnp.float64]:
        return jnp.concatenate([self.top_row, self.eigenvalues]), jnp.array(0.0)


@dataclass
class VolatilityParameters:
    """Volatility dynamics parameters μ̄_h, Γ_0, Γ_1, and mapping matrices."""

    mu_bar_h: Array
    gamma0: Array
    gamma1: Array
    phi_mh: Array
    phi_hg: Array

    def to_unconstrained(self) -> Tuple[Array, jnp.float64]:
        stacked = jnp.concatenate(
            [
                self.mu_bar_h.ravel(),
                self.gamma0.ravel(),
                self.gamma1.ravel(),
                self.phi_mh.ravel(),
                self.phi_hg.ravel(),
            ]
        )
        return stacked, jnp.array(0.0)


@dataclass
class CrossCovariances:
    """Cross-covariance matrices with structural restrictions."""

    sigma_h: Array
    sigma_hm: Array
    sigma_hg: Array
    sigma_m: Array

    def enforce_restrictions(self) -> Array:
        sigma_m = self.sigma_m
        last_row = sigma_m[-1]
        sigma_m = sigma_m.at[-1].set(jnp.zeros_like(last_row))
        return sigma_m

    def to_unconstrained(self) -> Tuple[Array, jnp.float64]:
        return (
            jnp.concatenate(
                [
                    self.sigma_h.ravel(),
                    self.sigma_hm.ravel(),
                    self.sigma_hg.ravel(),
                    self.sigma_m.ravel(),
                ]
            ),
            jnp.array(0.0),
        )


__all__ = [
    "MeasurementErrorDiag",
    "SigmaGLowerTri",
    "EigenstructurePMeasure",
    "QMeasureRestrictions",
    "VolatilityParameters",
    "CrossCovariances",
]
