from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _require_file(p: str | Path) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p


def load_observed_inputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load yields, dividends, raw price/dividend as pandas objects."""

    dpaths = cfg["data"]
    divg = pd.read_csv(_require_file(dpaths["dividends_csv"]))
    yields = pd.read_csv(_require_file(dpaths["yields_csv"]))
    price = pd.read_csv(_require_file(dpaths["price_csv"]))
    raw_div = pd.read_csv(_require_file(dpaths["raw_div_csv"]))
    return {"div_growth": divg, "yields": yields, "price": price, "raw_div": raw_div}


def sample_m(cfg: Dict[str, Any], rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Draw a random matrix m from a pre-generated MAT pool.
    Expects a 3D array (Nm x T x Npool) or list-of-matrices in `m_pool`.
    Returns (m, index).
    """

    f = _require_file(cfg["data"]["m_pool_mat"])
    var = cfg["data"]["m_pool_var"]
    mat = loadmat(f)
    if var not in mat:
        raise KeyError(f"Variable '{var}' not in {f}")
    pool = mat[var]
    # Accept: (Nm, T, K) or object array of (Nm,T)
    if pool.ndim == 3:
        k = pool.shape[2]
        idx = rng.integers(0, k)
        m = np.asarray(pool[:, :, idx], dtype=float)
    elif pool.dtype == object:
        k = pool.size
        idx = rng.integers(0, k)
        m = np.asarray(pool[0, idx], dtype=float)
    else:
        raise ValueError(f"Unsupported shape for m_pool: {pool.shape} d={pool.ndim}")
    return m, int(idx)
