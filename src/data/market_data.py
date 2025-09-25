"""Utilities for loading bond and equity market data files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


def _require_file(path: str | Path) -> Path:
    """Return the path if it exists, otherwise raise ``FileNotFoundError``."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    return resolved


def _first(iterable: Iterable[str]) -> str:
    try:
        return next(iter(iterable))
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise KeyError("HDF5 file does not contain any datasets") from exc


@dataclass(frozen=True)
class MarketData:
    """Container for parsed market data arrays."""

    bond_yields: np.ndarray
    sp500_price: np.ndarray
    sp500_div: np.ndarray
    log_div_grow: np.ndarray


def load_market_data(
    bond_yields_csv: str | Path,
    sp500_csv: str | Path,
) -> MarketData:
    """Load the market data CSV files into numpy arrays.

    Parameters
    ----------
    bond_yields_csv:
        Path to ``bond_yields.csv`` containing a header row and dates in the
        first column. Rows are returned from the first data row onwards while
        omitting the date column.
    sp500_csv:
        Path to ``sp500_data.csv`` containing a header row with price and
        dividend information. The function skips the first two data rows and
        returns the remaining values for the price, dividend, and log dividend
        growth columns.
    """

    bond_df = pd.read_csv(_require_file(bond_yields_csv))
    bond_yields = (
        bond_df.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=False)
    ) / 1200.0

    sp500_df = pd.read_csv(_require_file(sp500_csv))
    # Skip the first two data rows as requested (rows 3-end in 1-based indexing).
    data_slice = sp500_df.iloc[2:]
    numeric = data_slice.apply(pd.to_numeric, errors="coerce")
    sp500_price = numeric.iloc[:, 0].to_numpy(dtype=float, copy=False)
    sp500_div = numeric.iloc[:, 1].to_numpy(dtype=float, copy=False)
    log_div_grow = numeric.iloc[:, 3].to_numpy(dtype=float, copy=False) / 1200.0

    return MarketData(
        bond_yields=bond_yields,
        sp500_price=sp500_price,
        sp500_div=sp500_div,
        log_div_grow=log_div_grow,
    )


def get_fac_draw_slice(
    mat_file: str | Path,
    nn: int,
    *,
    dataset: str | None = None,
    log_div_grow: Iterable[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the ``nn``-th slice of the factor draws tensor using ``h5py``.

    The ``fac_draws_K4.mat`` file stores a tensor with shape ``(4, 547, N)``.
    Rather than loading the entire tensor, this helper reads only the
    two-dimensional slice requested by ``nn``. The slice is scaled by ``1/1200``
    and the provided ``log_div_grow`` series is appended as the final row,
    yielding an array with shape ``(5, 547)``.
    """

    if nn < 0:
        raise ValueError("'nn' must be a non-negative integer")

    mat_path = _require_file(mat_file)
    with h5py.File(mat_path, "r") as handle:
        dataset_name = dataset or _first(handle.keys())
        if dataset_name not in handle:
            available = ", ".join(handle.keys()) or "<none>"
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {mat_path}. "
                f"Available: {available}"
            )

        data = handle[dataset_name]
        if data.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor in dataset '{dataset_name}', got shape {data.shape}"
            )
        if nn >= data.shape[2]:
            raise IndexError(
                f"Index {nn} is out of bounds for axis 2 with size {data.shape[2]}"
            )

        # h5py reads only the requested slice into memory.
        slice_ = data[:, :, nn]

    slice_array = np.asarray(slice_, dtype=float) / 1200.0

    if log_div_grow is None:
        raise ValueError("'log_div_grow' must be provided to append to the factor slice")

    log_div_array = np.asarray(log_div_grow, dtype=float)
    if log_div_array.ndim != 1:
        raise ValueError("'log_div_grow' must be a one-dimensional sequence")

    if log_div_array.shape[0] != slice_array.shape[1]:
        raise ValueError(
            "Length of 'log_div_grow' must match the second dimension of the factor slice"
        )

    return np.vstack([slice_array, log_div_array])

