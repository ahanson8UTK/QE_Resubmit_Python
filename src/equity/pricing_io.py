from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from artifacts.registry import ArtifactRegistry

SERIES_NAME = "net_fundamental"
TIME_COLUMN = "time"


def _ensure_series(ts: pd.Series) -> pd.Series:
    if isinstance(ts, pd.Series):
        return ts.copy()
    return pd.Series(ts)


def _validate_index(index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        return
    if pd.api.types.is_integer_dtype(index):
        return
    raise TypeError("net-fundamental series must use a DatetimeIndex or integer index")


def save_net_fundamental(
    run_id: str,
    ts: pd.Series,
    *,
    registry: ArtifactRegistry,
) -> Path:
    """Persist the net-fundamental series to Parquet.

    Parameters
    ----------
    run_id:
        Identifier returned by :func:`artifacts.registry.new_run_id`.
    ts:
        Net-fundamental price series. Values should be expressed in monthly
        percentage units divided by ``1200`` in accordance with the pipeline
        conventions.
    registry:
        Artifact registry controlling output locations.
    """

    series = _ensure_series(ts)
    _validate_index(series.index)
    series = series.astype(float)
    series.name = SERIES_NAME

    series.index = pd.Index(series.index, name=TIME_COLUMN)
    df = series.to_frame().reset_index()

    path = registry.net_fundamental_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=str(path.parent), delete=False) as tmp:
        temp_path = Path(tmp.name)
    df.to_parquet(temp_path, engine="pyarrow", index=False)
    temp_path.replace(path)
    return path


def load_net_fundamental(
    run_id: Optional[str] = None,
    *,
    registry: Optional[ArtifactRegistry] = None,
) -> pd.Series:
    """Load a previously exported net-fundamental series.

    Parameters
    ----------
    run_id:
        Optional identifier. When omitted the registry ``_latest`` pointer is
        used.
    registry:
        Registry instance. Defaults to :meth:`ArtifactRegistry.from_default`.

    Returns
    -------
    pandas.Series
        Series indexed by ``DatetimeIndex`` or ``Index`` of integers with values
        in monthly percentage units divided by ``1200``.
    """

    reg = registry or ArtifactRegistry.from_default()
    resolved_run_id = run_id or reg.latest_run_id("equity")
    if resolved_run_id is None:
        raise FileNotFoundError("No equity runs found in the artifact registry.")

    path = reg.net_fundamental_path(resolved_run_id)
    if not path.exists():
        raise FileNotFoundError(f"net_fundamental.parquet not found for run {resolved_run_id}")

    df = pd.read_parquet(path, engine="pyarrow")
    if TIME_COLUMN not in df.columns or SERIES_NAME not in df.columns:
        raise ValueError("Parquet file missing required columns 'time'/'net_fundamental'.")

    time_col = df[TIME_COLUMN]
    if pd.api.types.is_datetime64_any_dtype(time_col):
        dt_index = pd.DatetimeIndex(pd.to_datetime(time_col), name=TIME_COLUMN)
        inferred = pd.infer_freq(dt_index)
        if inferred:
            dt_index = pd.DatetimeIndex(dt_index, name=TIME_COLUMN, freq=inferred)
        index = dt_index
    elif pd.api.types.is_integer_dtype(time_col):
        index = pd.Index(time_col.astype(int), name=TIME_COLUMN)
    else:
        raise TypeError("Stored time column must be datetime64 or integer typed.")

    series = pd.Series(df[SERIES_NAME].astype(float).to_numpy(), index=index, name=SERIES_NAME)
    return series


__all__ = ["load_net_fundamental", "save_net_fundamental"]
