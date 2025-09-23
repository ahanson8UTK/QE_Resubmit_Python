from __future__ import annotations

from typing import Optional

import pandas as pd

from artifacts.registry import ArtifactRegistry
from equity.pricing_io import load_net_fundamental


def get_net_fundamental(run_id: Optional[str] = None) -> pd.Series:
    """Load the net-fundamental price series for the bubble module."""

    registry = ArtifactRegistry.from_default()
    resolved_run_id = run_id or registry.latest_run_id("equity")
    if resolved_run_id is None:
        raise FileNotFoundError("No equity runs found. Generate an equity pricing artifact first.")
    return load_net_fundamental(resolved_run_id, registry=registry)


__all__ = ["get_net_fundamental"]
