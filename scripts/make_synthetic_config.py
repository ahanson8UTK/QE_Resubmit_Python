"""Utility script to dump a synthetic-data configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "run": {"seed": 0, "chains": 2, "results_dir": "results", "run_id_prefix": "synthetic"},
    "data": {"synthetic": True, "paths_config": "configs/data_paths.yaml", "priors_config": "configs/priors_example.yaml"},
}


def main() -> None:
    path = Path("configs/synthetic_defaults.yaml")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_CONFIG, handle)
    print(f"Wrote synthetic config to {path}")


if __name__ == "__main__":
    main()
