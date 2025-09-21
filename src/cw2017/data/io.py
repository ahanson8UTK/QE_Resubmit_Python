"""Data loading utilities for the Creal & Wu (2017) sampler scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import io as scipy_io

DIVIDE_BY_1200_COLUMNS = ("dividend_growth", "yield")


def _load_csv_series(path: Path, value_column: str | None = None) -> pd.Series:
    df = pd.read_csv(path)
    if value_column is None:
        if df.shape[1] != 2:
            raise ValueError("CSV must contain exactly two columns when value_column is None")
        value_column = df.columns[1]
    series = pd.Series(df[value_column].to_numpy(), index=pd.to_datetime(df.iloc[:, 0]))
    return series


def load_m_mat_random(mat_path: str, key: np.random.Generator) -> np.ndarray:
    """Load candidate ``m`` matrices from ``mat_path`` and sample one at random."""

    mat = scipy_io.loadmat(mat_path)
    candidates: Iterable[np.ndarray] = []
    for candidate_key in ("m_candidates", "m_list", "m_store"):
        if candidate_key in mat:
            raw = mat[candidate_key]
            if raw.ndim == 3:
                candidates = raw
            elif isinstance(raw, np.ndarray) and raw.dtype == object:
                candidates = [np.asarray(item) for item in raw.ravel()]
            break
    else:
        raise KeyError("Expected one of m_candidates, m_list, or m_store in .mat file")

    if isinstance(candidates, np.ndarray) and candidates.ndim == 3:
        idx = key.integers(0, candidates.shape[0])
        matrix = np.asarray(candidates[idx])
    else:
        candidate_list = list(candidates)
        idx = key.integers(0, len(candidate_list))
        matrix = np.asarray(candidate_list[idx])
    if matrix.ndim != 2:
        raise ValueError("Selected m candidate must be 2D")
    return matrix


def load_div_growth_csv(csv_path: str) -> pd.Series:
    """Load dividend growth series and convert from annualised percentage to monthly."""

    series = _load_csv_series(Path(csv_path))
    series = series / 1200.0
    series.name = "dividend_growth"
    return series


def load_yields_csv(csv_path: str, maturities: Sequence[int] | None = None) -> pd.DataFrame:
    """Load Treasury yields and divide by 1200 to convert basis points to monthly rates."""

    maturities = tuple(maturities or (3, 6, 12, 36, 60, 84, 120))
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Yield CSV must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    value_columns = [f"m_{maturity}" for maturity in maturities]
    missing = [col for col in value_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Yield CSV missing columns: {missing}")
    result = df.set_index("date")[value_columns].astype(float) / 1200.0
    result.columns = pd.Index([f"yield_{maturity}" for maturity in maturities])
    return result


def load_sp500_price_div(csv_path_price: str, csv_path_div: str) -> pd.DataFrame:
    """Load S&P500 price and dividend series, keeping raw and scaled variants."""

    price = _load_csv_series(Path(csv_path_price))
    div = _load_csv_series(Path(csv_path_div))
    df = pd.DataFrame({"price_raw": price, "dividend_raw": div})
    df["dividend_growth"] = df["dividend_raw"].pct_change().fillna(0.0)
    df["dividend_growth_scaled"] = df["dividend_growth"] / 1200.0
    return df


def merge_macro_and_divgrowth(m: np.ndarray, divgrowth: pd.Series) -> pd.DataFrame:
    """Merge macro series ``m`` with dividend growth ensuring aligned horizons."""

    if m.ndim != 2:
        raise ValueError("m must be a 2D array")
    if m.shape[1] != divgrowth.shape[0]:
        raise ValueError("Time dimension mismatch between m and dividend growth")
    columns = [f"macro_{i}" for i in range(m.shape[0])]
    df = pd.DataFrame(m.T, columns=columns, index=divgrowth.index)
    df[divgrowth.name or "dividend_growth"] = divgrowth.to_numpy()
    return df


__all__ = [
    "load_m_mat_random",
    "load_div_growth_csv",
    "load_yields_csv",
    "load_sp500_price_div",
    "merge_macro_and_divgrowth",
]
