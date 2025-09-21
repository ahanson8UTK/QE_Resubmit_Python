"""Main Gibbs sampler driver (placeholder implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp

from ..config import AppConfig
from ..reporting import save_results, summarize
from ..utils.logging import setup_logging

BLOCK_SEQUENCE = ["3a", "3b", "3c", "3d", "3e", "4", "5", "6"]


def run_gibbs(rng_key: jax.Array, config: AppConfig, results_root: Path) -> Dict[str, object]:
    """Execute a single Gibbs sweep and write a dummy report."""

    setup_logging(level=config.logging.level, rich_tracebacks=config.logging.rich_tracebacks)
    run_id = save_results.build_run_id(config.run.run_id_prefix)
    summary_dir = save_results.prepare_run_directory(results_root, run_id)

    block_metrics: List[Dict[str, float]] = []
    for block in BLOCK_SEQUENCE:
        block_metrics.append(
            {
                "block": block,
                "acceptance": 1.0,
                "ess_per_second": 0.0,
                "work_units": 0.0,
            }
        )

    summary_df = summarize.summarise_blocks(block_metrics)
    summarize.write_summary(summary_df, summary_dir)
    save_results.write_metadata(
        summary_dir,
        {
            "run_id": run_id,
            "chains": config.run.chains,
            "seed": config.run.seed,
        },
    )
    return {"run_id": run_id, "summary": summary_df}


__all__ = ["run_gibbs", "BLOCK_SEQUENCE"]
