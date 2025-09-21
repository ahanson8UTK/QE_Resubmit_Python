"""Command-line interface to launch the Gibbs sampler scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cw2017 import config as app_config
from cw2017.data import datasets
from cw2017.samplers import gibbs
from cw2017.utils import rng as rng_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/defaults.yaml"))
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = app_config.load_app_config(args.config)
    if args.chains is not None:
        config.run.chains = args.chains
    if args.seed is not None:
        config.run.seed = args.seed

    rng_manager = rng_utils.RNGManager(config.run.seed)
    rng_key = rng_manager.key

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
