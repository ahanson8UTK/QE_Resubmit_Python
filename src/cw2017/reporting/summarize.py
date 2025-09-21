"""Summarise Gibbs sampling outputs."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import pandas as pd


def summarise_blocks(block_metrics: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Convert block-level metrics to a tidy :class:`~pandas.DataFrame`."""

    return pd.DataFrame(list(block_metrics))


def write_summary(output_dir: Path, records: List[Dict[str, float]]) -> None:
    """Write a CSV metrics table and a short textual report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = summarise_blocks(records)
    df.to_csv(output_dir / "metrics.csv", index=False)
    acceptances = [float(rec.get("acceptance", float("nan"))) for rec in records]
    acceptances = [value for value in acceptances if value == value]
    mean_accept = mean(acceptances) if acceptances else float("nan")
    with (output_dir / "report.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Mean acceptance probability across sweeps: "
            f"{mean_accept:.3f}\n"
        )


__all__ = ["summarise_blocks", "write_summary"]
