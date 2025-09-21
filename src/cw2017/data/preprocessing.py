"""Pre-processing hooks for macro and financial time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    """Simple standardisation transform for arrays."""

    mean_: float | np.ndarray
    scale_: float | np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.scale_


def make_standard_scaler(data: np.ndarray) -> StandardScaler:
    return StandardScaler(mean_=float(np.mean(data)), scale_=float(np.std(data) + 1e-8))


def align_to_month_end(series: pd.Series) -> pd.Series:
    return series.resample("M").last()


def apply_pipeline(df: pd.DataFrame, *transforms: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    for transform in transforms:
        df = transform(df)
    return df


__all__ = ["StandardScaler", "make_standard_scaler", "align_to_month_end", "apply_pipeline"]
