"""Full Gibbs sweep for the bond sub-model."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import jax

from sampling.hmc_block import HMCConfig, HMCCache, NUTSBlock
from sampling.diagnostics import write_jsonl


def _require(mapping: Dict[str, Any], name: str) -> Any:
    if name not in mapping or mapping[name] is None:
        raise KeyError(f"Missing required entry '{name}' in state/context")
    return mapping[name]


def _logpost_block3a(theta_c: Any, ctx: Dict[str, Any]) -> Tuple[jax.Array, Dict[str, Any]]:
    reconstruct = _require(ctx, "reconstruct_theta")
    theta_full = reconstruct(theta_c, ctx["theta_rest"])
    loglike_fn = _require(ctx, "loglikelihood")
    logprior_fn = _require(ctx, "logprior")
    loglik = loglike_fn(theta_full, ctx)
    logprior = logprior_fn(theta_c, ctx.get("priors"))
    return loglik + logprior, {"loglik": loglik, "logprior": logprior}


def _logpost_block3b(theta_c: Any, ctx: Dict[str, Any]) -> Tuple[jax.Array, Dict[str, Any]]:
    reconstruct = _require(ctx, "reconstruct_theta")
    theta_full = reconstruct(theta_c, ctx["theta_rest"])
    loglike_fn = _require(ctx, "loglikelihood")
    logprior_fn = _require(ctx, "logprior")
    loglik = loglike_fn(theta_full, ctx)
    logprior = logprior_fn(theta_c, ctx.get("priors"))
    return loglik + logprior, {"loglik": loglik, "logprior": logprior}


def run_bond_sweep(
    rng_key: jax.Array,
    state: Dict[str, Any],
    caches: Dict[str, HMCCache],
    cfg_blocks: Dict[str, HMCConfig],
    save_diag_path: str = "results/diagnostics/bond_sweep.jsonl",
) -> Tuple[Dict[str, Any], Dict[str, HMCCache]]:
    """Execute blocks 3a–3e for a single Gibbs sweep."""

    t0 = time.perf_counter()
    key = rng_key

    # ----- Block 3a -----
    key, subkey = jax.random.split(key)
    ctx3a = {
        "theta_rest": state.get("theta"),
        "loglikelihood": state.get("loglikelihood_block3a"),
        "logprior": state.get("logprior_block3a"),
        "priors": state.get("priors"),
        "reconstruct_theta": state.get("reconstruct_theta_3a"),
    }
    if ctx3a["loglikelihood"] is None:
        ctx3a["loglikelihood"] = state.get("loglikelihood")
    nuts3a = NUTSBlock(
        _logpost_block3a,
        _require(state, "transform_3a"),
        _require(state, "inv_transform_3a"),
        cfg_blocks["3a"],
    )
    theta3a_new, diag3a, cache3a = nuts3a.warmup_and_draw(
        _require(state, "theta_block3a"),
        caches.get("3a"),
        ctx3a,
        subkey,
    )
    state["theta_block3a"] = theta3a_new
    state["theta"] = _require(ctx3a, "reconstruct_theta")(theta3a_new, state["theta"])

    # ----- Block 3b -----
    key, subkey = jax.random.split(key)
    ctx3b = {
        "theta_rest": state.get("theta"),
        "loglikelihood": state.get("loglikelihood_block3b"),
        "logprior": state.get("logprior_block3b"),
        "priors": state.get("priors"),
        "reconstruct_theta": state.get("reconstruct_theta_3b"),
    }
    if ctx3b["loglikelihood"] is None:
        ctx3b["loglikelihood"] = state.get("loglikelihood")
    nuts3b = NUTSBlock(
        _logpost_block3b,
        _require(state, "transform_3b"),
        _require(state, "inv_transform_3b"),
        cfg_blocks["3b"],
    )
    theta3b_new, diag3b, cache3b = nuts3b.warmup_and_draw(
        _require(state, "theta_block3b"),
        caches.get("3b"),
        ctx3b,
        subkey,
    )
    state["theta_block3b"] = theta3b_new
    state["theta"] = _require(ctx3b, "reconstruct_theta")(theta3b_new, state["theta"])

    # ----- Block 3c -----
    key, subkey = jax.random.split(key)
    draw_g_fn = _require(state, "draw_block3c")
    g_new, extra_new, dk_diag = draw_g_fn(subkey, state)
    state.update(extra_new)
    state["g"] = g_new

    # ----- Block 3d -----
    key, subkey = jax.random.split(key)
    draw_h_fn = _require(state, "draw_block3d")
    h_new, pg_diag = draw_h_fn(subkey, state)
    state["h"] = h_new

    # ----- Block 3e -----
    cov_update_fn = _require(state, "draw_cov_blocks")
    state = cov_update_fn(state)

    caches["3a"] = cache3a
    caches["3b"] = cache3b

    diag_record = {
        "phase": "bond_sweep",
        "time_sec": time.perf_counter() - t0,
        "diag_3a": diag3a,
        "diag_3b": diag3b,
        "dk": dk_diag,
        "pgas": pg_diag,
    }
    write_jsonl(save_diag_path, diag_record)
    return state, caches


__all__ = ["run_bond_sweep"]
