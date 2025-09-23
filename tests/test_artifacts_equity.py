from __future__ import annotations

import json

import numpy as np
import pandas as pd

from artifacts.registry import ArtifactRegistry, new_run_id
from equity.pricing_io import load_net_fundamental, save_net_fundamental


def test_net_fundamental_round_trip(tmp_path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    run_id = new_run_id("equity")

    index = pd.date_range("2000-01-31", periods=5, freq="ME", name="time")
    values = pd.Series(np.linspace(0.0, 1.0, len(index)), index=index, name="net_fundamental")

    parquet_path = save_net_fundamental(run_id, values, registry=registry)
    assert parquet_path.exists()

    manifest_payload = {"run_id": run_id, "note": "roundtrip"}
    manifest_path = registry.write_manifest(run_id, manifest_payload)
    assert manifest_path.exists()

    loaded = load_net_fundamental(run_id, registry=registry)
    pd.testing.assert_series_equal(loaded, values)

    latest = registry.latest_run_id("equity")
    assert latest == run_id

    manifest_loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_loaded["run_id"] == run_id

    latest_marker = registry.root / "equity" / "_latest"
    assert latest_marker.read_text(encoding="utf-8").strip() == run_id
