from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from artifacts.registry import ArtifactRegistry, new_run_id
from equity.pricing_io import save_net_fundamental
from tsm.equity_pricer import EquityPricingInputs, price_equity_ratio


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _git_dirty() -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Series):
        return _normalize(obj.to_dict())
    if isinstance(obj, pd.Index):
        return _normalize(list(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _normalize(vars(obj))
    return obj


def _config_hash(config: Any) -> str:
    normalized = _normalize(config)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _theta_summary(theta: EquityPricingInputs) -> MutableMapping[str, Any]:
    gamma_dd2 = np.asarray(theta.gamma_dd2, dtype=float)
    return {
        "d_m": int(theta.d_m),
        "d_g": int(theta.d_g),
        "d_h": int(theta.d_h),
        "gamma_dd0": float(theta.gamma_dd0),
        "gamma_dd2_mean": float(gamma_dd2.mean()),
        "gamma_dd2_std": float(gamma_dd2.std()),
    }


def _ensure_series(obj: Any, name: str) -> pd.Series:
    if isinstance(obj, pd.Series):
        series = obj.copy()
    else:
        series = pd.Series(obj, name=name)
    if series.name is None:
        series.name = name
    return series


def price_equity_and_export(
    theta: EquityPricingInputs,
    states: Mapping[str, Any],
    dividends: Any,
    prices: Any,
    config: Mapping[str, Any],
) -> str:
    """Compute and persist the net-fundamental equity series.

    Parameters
    ----------
    theta:
        Equity pricing inputs controlling the ATSM recursion.
    states:
        Mapping containing ``m_t``, ``g_t``, and ``h_t`` state arrays with shapes
        ``(T, d_m)``, ``(T, d_g)``, and ``(T, d_h)`` respectively.
    dividends, prices:
        Observed monthly dividend and price series already scaled by dividing by
        ``1200`` per the pipeline convention. Accepts array-like inputs which
        are converted to :class:`pandas.Series`.
    config:
        Configuration dictionary whose hash is recorded in the manifest for
        reproducibility.

    Returns
    -------
    str
        Run identifier registered in the artifact registry.
    """

    m_t = np.asarray(states["m_t"], dtype=float)
    g_t = np.asarray(states["g_t"], dtype=float)
    h_t = np.asarray(states["h_t"], dtype=float)

    ratio = price_equity_ratio(theta, m_t=m_t, g_t=g_t, h_t=h_t)

    dividend_series = _ensure_series(dividends, "dividend_raw")
    price_series = _ensure_series(prices, "price_raw")

    if len(dividend_series) != ratio.shape[0] or len(price_series) != ratio.shape[0]:
        raise ValueError("State and observable series must share the same length.")
    if not dividend_series.index.equals(price_series.index):
        raise ValueError("Dividend and price series must be aligned on the same index.")

    ratio_series = pd.Series(ratio, index=dividend_series.index, name="ratio")
    fundamental = ratio_series * dividend_series
    net_fundamental = price_series - fundamental
    net_fundamental.name = "net_fundamental"

    registry = ArtifactRegistry.from_default()
    run_id = new_run_id("equity")
    save_net_fundamental(run_id, net_fundamental, registry=registry)

    git_commit = _git_commit() or "unknown"
    git_dirty = _git_dirty()
    manifest = {
        "run_id": run_id,
        "T": int(len(net_fundamental)),
        "start": str(net_fundamental.index[0]),
        "end": str(net_fundamental.index[-1]),
        "generated_by": "price_equity_and_export",
        "git": {"commit": git_commit, "dirty": bool(git_dirty) if git_dirty is not None else False},
        "config_hash": _config_hash(config),
        "theta_summary": dict(_theta_summary(theta)),
    }
    registry.write_manifest(run_id, manifest)
    return run_id


__all__ = ["price_equity_and_export"]
