"""hmc_gibbs
=================

A modular, typed JAX implementation scaffold for the Creal & Wu (2017) HMC-in-Gibbs sampler.
The package is intentionally incomplete; math-heavy routines raise ``NotImplementedError``
with detailed TODO lists describing the next development steps.
"""

from . import config as _config  # Re-export for convenience.

__all__ = ["_config"]
