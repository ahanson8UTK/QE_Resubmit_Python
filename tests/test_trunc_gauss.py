"""Tests for truncated Gaussian sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from samplers.trunc_gauss import sample_halfspace_trunc_normal

jax.config.update("jax_enable_x64", True)


def test_halfspace_truncation_respects_constraint():
    key = jax.random.PRNGKey(0)
    mean = jnp.array([0.5, -0.25])
    cov = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    a = jnp.array([1.0, 0.0])
    b = 0.0

    keys = jax.random.split(key, 10_000)
    samples = jax.vmap(lambda k: sample_halfspace_trunc_normal(k, mean, cov, a, b))(keys)

    constraint_values = samples @ a + b
    assert float(jnp.max(constraint_values)) <= 1e-6

    proj_mean = jnp.mean(samples[:, 0])
    assert proj_mean < -0.1


def test_nearly_nonbinding_constraint_matches_unconstrained():
    key = jax.random.PRNGKey(1)
    mean = jnp.array([0.3, -0.2])
    cov = jnp.array([[0.7, 0.1], [0.1, 0.5]])
    a = jnp.array([1e-16, 0.0])
    b = -1.0

    num_samples = 2_000
    keys = jax.random.split(key, num_samples)
    samples = jax.vmap(lambda k: sample_halfspace_trunc_normal(k, mean, cov, a, b))(keys)

    empirical_mean = jnp.mean(samples, axis=0)
    assert jnp.allclose(empirical_mean, mean, atol=5e-2)

    constraint_values = samples @ a + b
    assert float(jnp.max(constraint_values)) <= 1e-6
