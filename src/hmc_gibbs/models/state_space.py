"""State-space model containers for the Creal & Wu (2017) specification."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..typing import Array


@dataclass
class StateSpaceModel:
    """Basic container capturing dimensions of the linear state-space model."""

    transition_matrix: Array
    measurement_matrix: Array
    transition_noise: Array
    measurement_noise: Array

    def validate(self) -> None:
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")


__all__ = ["StateSpaceModel"]
