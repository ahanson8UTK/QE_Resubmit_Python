"""CLI entry point for the bubble PM-HMC stage using stored equity artifacts."""

from __future__ import annotations

import argparse

import pandas as pd

from artifacts.registry import ArtifactRegistry
from bubble.data_io import get_net_fundamental


def _format_series_block(series: pd.Series, label: str) -> str:
    formatted = series.to_string(max_rows=3)
    return f"{label}:\n{formatted}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=str, default=None, help="Equity artifact run identifier (defaults to latest).")
    args = parser.parse_args()

    registry = ArtifactRegistry.from_default()
    resolved_run_id = args.run_id or registry.latest_run_id("equity")
    if resolved_run_id is None:
        raise FileNotFoundError("No equity artifacts available; run the pricing step first.")

    series = get_net_fundamental(resolved_run_id)

    print(f"[bubble] using equity run: {resolved_run_id}")
    print(_format_series_block(series.head(3), "first 3"))
    print(_format_series_block(series.tail(3), "last 3"))
    # Placeholder for the actual HMC/PM-HMC execution. Once integrated, invoke the
    # sampling routine here using ``series`` as the observed net-fundamental prices.


if __name__ == "__main__":
    main()
