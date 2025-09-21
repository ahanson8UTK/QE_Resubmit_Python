"""Summarise Gibbs sampling outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def summarise_blocks(block_metrics: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Convert block-level metrics to a tidy :class:`~pandas.DataFrame`."""

    df = pd.DataFrame(list(block_metrics))
    return df


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Write tabular summary to HTML and Parquet files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_html(output_dir / "report.html", index=False)
    df.to_parquet(output_dir / "metrics.parquet")


__all__ = ["summarise_blocks", "write_summary"]
