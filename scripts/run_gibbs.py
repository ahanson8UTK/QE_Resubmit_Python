"""Command-line interface to launch the Gibbs sampler scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hmc_gibbs.utils import jax_setup  # noqa: F401

import jax
import jax.numpy as jnp

from hmc_gibbs import config as app_config
from hmc_gibbs.data import datasets
from hmc_gibbs.reporting import save_results, summarize
from hmc_gibbs.samplers import gibbs
from hmc_gibbs.utils import rng as rng_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="run the block-3a smoke sweep")
    return parser.parse_args()


def _make_smoke_data(rng_key: jax.Array, T: int, d_state: int, d_y: int, d_m: int) -> Dict[str, jnp.ndarray]:
    key_y, key_m, key_h = jax.random.split(rng_key, 3)
    y_t = jax.random.normal(key_y, (T, d_y), dtype=jnp.float64)
    m_t = jax.random.normal(key_m, (T, d_m), dtype=jnp.float64)
    h_t = jax.random.normal(key_h, (T, d_state), dtype=jnp.float64)
    params = {
        "a0": jnp.zeros(d_state, dtype=jnp.float64),
        "S0": jnp.eye(d_state, dtype=jnp.float64) * 0.3,
        "theta": jnp.zeros(d_state, dtype=jnp.float64),
    }
    return {"y_t": y_t, "m_t": m_t, "fixed": {"h_t": h_t}, "params": params}


def run_smoke(cfg: app_config.AppConfig, rng_key: jax.Array) -> None:
    chains = cfg.run.chains
    theta_dim = 4
    warmup_cfg = {
        "num_warmup_steps": 50,
        "target_accept": 0.651,
        "jitter_amount": 0.9,
        "decay_rate": 0.5,
        "adam_lr": 0.02,
        "initial_step_size": 0.25,
    }

    data = _make_smoke_data(rng_key, T=50, d_state=4, d_y=3, d_m=2)
    theta0_key, rng_key = jax.random.split(rng_key)
    theta0 = 0.1 * jax.random.normal(theta0_key, (chains, theta_dim), dtype=jnp.float64)
    state = gibbs.SmokeState(theta_block3a=theta0)

    cfg_dict = {"warmup": warmup_cfg}
    rng_key, state, metrics = gibbs.run_smoke_sweeps(rng_key, state, cfg_dict, data, num_sweeps=5)

    run_id = save_results.build_run_id(cfg.run.run_id_prefix)
    summary_dir = save_results.prepare_run_directory(Path(cfg.run.results_dir), run_id)
    summarize.write_summary(summary_dir, metrics)
    save_results.write_metadata(
        summary_dir,
        {
            "run_id": run_id,
            "chains": cfg.run.chains,
            "seed": cfg.run.seed,
        },
    )

    tuned = state.diagnostics_3a or {}
    epsilon = tuned.get("final_step_size", float("nan"))
    traj = tuned.get("final_trajectory_length", float("nan"))
    print(f"[smoke] tuned step size: {float(epsilon):.4f}; trajectory length: {float(traj):.4f}")


def main() -> None:
    args = parse_args()
    config = app_config.load_app_config(args.config)
    if args.chains is not None:
        config.run.chains = args.chains
    if args.seed is not None:
        config.run.seed = args.seed

    rng_manager = rng_utils.RNGManager(config.run.seed)
    rng_key = rng_manager.key

    if args.smoke:
        run_smoke(config, rng_key)
        return

    if config.data.synthetic:
        datasets.load_synthetic(seed=config.run.seed)
    else:
        path_cfg = app_config.load_yaml(config.data.paths_config)
        data_paths = {key: Path(value) for key, value in path_cfg.get("csv", {}).items()}
        data_paths.update({key: Path(value) for key, value in path_cfg.get("matrices", {}).items()})
        datasets.load_from_config(data_paths, seed=config.run.seed)

    results_root = config.run.results_dir
    gibbs.run_gibbs(rng_key, config, Path(results_root))


if __name__ == "__main__":
    main()
