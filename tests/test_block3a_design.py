import jax.numpy as jnp

from cw2017.kalman.design import build_block3a_design
from cw2017.models.conditionals import unpack_params3a_unconstrained
from cw2017.models.parameters import constrain_params3a
from cw2017.equity.atsm_measurement import build_measurement_terms
from test_block3a_smoke import block3a_setup


def test_block3a_design_shapes(block3a_setup) -> None:
    _, fixed, data, cfg, _warmup, info, maturities, init_vec = block3a_setup

    params_u = unpack_params3a_unconstrained(init_vec, info, maturities)
    params_c, _ = constrain_params3a(params_u, cfg, info)
    measurement_terms = build_measurement_terms(params_c, fixed, maturities)

    design = build_block3a_design(
        params_c,
        fixed,
        measurement_terms,
        data["y_t"],
        fixed["m_t"],
        fixed["h_t"],
        maturities,
    )

    y_t = jnp.asarray(data["y_t"], dtype=jnp.float64)
    m_t = jnp.asarray(fixed["m_t"], dtype=jnp.float64)
    h_t = jnp.asarray(fixed["h_t"], dtype=jnp.float64)
    Ng = int(info["Ng"])
    dy = y_t.shape[1]
    dm = m_t.shape[1]
    dh = h_t.shape[1]
    mu_gu_dim = fixed["M1"].shape[1]
    mu_gq_dim = measurement_terms[4].shape[1]
    state_dim = dm + Ng + dh + dm + mu_gu_dim + mu_gq_dim
    obs_dim = dm + dy + dh
    T = y_t.shape[0]

    assert design.observations.shape == (T, obs_dim)
    assert design.measurement_matrix.shape == (obs_dim, state_dim)
    assert design.measurement_noise_chol.shape == (obs_dim, obs_dim)
    assert design.process_noise_chol.shape == (T, state_dim, state_dim)
    assert design.state_offsets.shape == (T, state_dim)
    assert design.initial_mean.shape == (state_dim,)
    assert design.initial_sqrt_cov.shape == (state_dim, state_dim)

    # Noise rows corresponding to yields should be positive.
    diag_entries = jnp.diag(design.measurement_noise_chol)
    det_diag = jnp.concatenate(
        [
            diag_entries[:dm],
            diag_entries[-dh:] if dh > 0 else jnp.array([], dtype=jnp.float64),
        ]
    )
    assert jnp.allclose(det_diag, jnp.sqrt(1e-6), atol=1e-10)
    assert jnp.all(diag_entries[dm : dm + dy] > 0.0)

    # Initial covariance should match the first process noise.
    assert jnp.allclose(
        design.initial_sqrt_cov,
        design.process_noise_chol[0],
    )
