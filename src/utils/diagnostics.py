from __future__ import annotations

import time
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd


class TraceAccumulator:
    """
    Minimal accumulator to store per-iteration parameter draws and compute ESS/R-hat.
    Assumes single-chain, can be extended to multi-chain later.
    """

    def __init__(self) -> None:
        self.buffer: Dict[str, List[np.ndarray]] = {}
        self.walltimes: List[float] = []
        self._t0 = time.perf_counter()

    def add_draws(self, draw_dict: Dict[str, np.ndarray]) -> None:
        for k, v in draw_dict.items():
            v = np.atleast_1d(np.asarray(v))
            self.buffer.setdefault(k, []).append(v)
        self.walltimes.append(time.perf_counter() - self._t0)

    def to_inferencedata(self, burn_in: int = 0, thin: int = 1) -> az.InferenceData:
        post: Dict[str, np.ndarray] = {}
        for k, seq in self.buffer.items():
            if not seq:
                continue
            arr = np.stack(seq, axis=0)
            if burn_in > 0:
                arr = arr[burn_in:, ...]
            if thin > 1:
                arr = arr[::thin, ...]
            if arr.size == 0:
                continue
            arr = arr[np.newaxis, ...]  # (chain=1, draw, ...)
            post[k] = arr
        return az.from_dict(posterior=post) if post else az.InferenceData()

    def summarize(self, idata: az.InferenceData) -> pd.DataFrame:
        if not hasattr(idata, "posterior"):
            return pd.DataFrame()
        try:
            var_names = list(idata.posterior.data_vars)
        except AttributeError:
            return pd.DataFrame()
        if not var_names:
            return pd.DataFrame()
        summ = az.summary(
            idata,
            var_names=var_names,
            kind="stats",
            stat_focus="median",
            round_to=None,
        )
        return summ


def compute_ess_per_sec(idata: az.InferenceData, walltimes: List[float]) -> Dict[str, float]:
    """Rough ESS/sec using last timestamp as total time."""

    if not hasattr(idata, "posterior") or not walltimes:
        return {}
    try:
        data_vars = list(idata.posterior.data_vars)
    except AttributeError:
        return {}
    if not data_vars:
        return {}
    total_sec = walltimes[-1]
    ess = az.ess(idata, method="bulk")
    out: Dict[str, float] = {}
    for v in ess.data_vars:  # type: ignore[union-attr]
        val = float(np.asarray(ess[v]).reshape(-1).mean())
        out[str(v)] = val / max(total_sec, 1e-9)
    return out
