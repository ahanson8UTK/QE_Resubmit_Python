"""Dataset factory functions for synthetic smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import io


@dataclass
class Dataset:
    """Container bundling macro variables, dividend growth, and yields."""

    macro: pd.DataFrame
    yields: pd.DataFrame
    sp500: pd.DataFrame


def load_synthetic(seed: int = 0, periods: int = 24) -> Dataset:
    rng = np.random.default_rng(seed)
    macro = pd.DataFrame(
        rng.normal(size=(periods, 3)),
        index=pd.date_range("2000-01-31", periods=periods, freq="M"),
        columns=["macro_0", "macro_1", "macro_2"],
    )
    yields = pd.DataFrame(
        rng.normal(loc=0.02, scale=0.001, size=(periods, 3)),
        index=macro.index,
        columns=["yield_3", "yield_12", "yield_60"],
    )
    sp500 = pd.DataFrame(
        {
            "price_raw": 1000 * np.exp(0.01 * np.arange(periods)),
            "dividend_raw": np.linspace(10, 12, periods),
        },
        index=macro.index,
    )
    sp500["dividend_growth"] = sp500["dividend_raw"].pct_change().fillna(0.0)
    sp500["dividend_growth_scaled"] = sp500["dividend_growth"] / 1200.0
    return Dataset(macro=macro, yields=yields, sp500=sp500)


def load_from_config(paths: Dict[str, Path], seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    m = io.load_m_mat_random(str(paths["m_candidates"]), rng)
    div = io.load_div_growth_csv(str(paths["dividend_growth"]))
    macro = io.merge_macro_and_divgrowth(m, div)
    yields = io.load_yields_csv(str(paths["yields"]))
    sp500 = io.load_sp500_price_div(str(paths["sp500_price"]), str(paths["sp500_dividend"]))
    return Dataset(macro=macro, yields=yields, sp500=sp500)


__all__ = ["Dataset", "load_synthetic", "load_from_config"]
