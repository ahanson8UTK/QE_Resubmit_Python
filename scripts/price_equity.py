"""Command-line interface to compute and export equity net-fundamental series."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import yaml

from equity.price_equity import price_equity_and_export
from tsm.equity_pricer import EquityPricingInputs


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    return data


def _equity_inputs_from_npz(path: Path) -> EquityPricingInputs:
    data = np.load(path, allow_pickle=True)
    d_m = int(data["d_m"])
    d_g = int(data["d_g"])
    d_h = int(data["d_h"])
    return EquityPricingInputs(
        d_m=d_m,
        d_g=d_g,
        d_h=d_h,
        Phi_m=data["Phi_m"],
        Phi_mg=data["Phi_mg"],
        Phi_mh=data["Phi_mh"],
        Phi_h=data["Phi_h"],
        Phi_g=data["Phi_g"],
        Phi_g_Q=data["Phi_g_Q"],
        Phi_gm=data["Phi_gm"],
        Phi_gh=data["Phi_gh"],
        Sigma_m=data["Sigma_m"],
        Sigma_g=data["Sigma_g"],
        Sigma_g_Q=data["Sigma_g_Q"],
        Sigma_hm=data["Sigma_hm"],
        Sigma_hg=data["Sigma_hg"],
        Sigma_h=data["Sigma_h"],
        mu_m=data["mu_m"],
        mu_g=data["mu_g"],
        mu_g_Q=data["mu_g_Q"],
        mu_h=data["mu_h"],
        Gamma0=data["Gamma0"],
        Gamma1=data["Gamma1"],
        gamma_dd0=float(data["gamma_dd0"]),
        gamma_dd2=data["gamma_dd2"],
        e_div_ix=int(data.get("e_div_ix", d_m - 1)),
        e1_g_ix=int(data.get("e1_g_ix", 0)),
        dividend_uses_lagged_h=bool(data.get("dividend_uses_lagged_h", True)),
    )


def _build_equity_inputs(cfg: Mapping[str, Any]) -> EquityPricingInputs:
    if "npz" in cfg:
        return _equity_inputs_from_npz(Path(cfg["npz"]))
    return EquityPricingInputs(**cfg)


def _load_states_and_observables(cfg: Mapping[str, Any]) -> tuple[Dict[str, np.ndarray], pd.Series, pd.Series]:
    if "npz" in cfg:
        path = Path(cfg["npz"])
        data = np.load(path, allow_pickle=True)
        index = pd.to_datetime(data["dates"]) if "dates" in data else pd.RangeIndex(data["m_t"].shape[0], name="time")
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.Index(index, name="time")
        dividends = pd.Series(data["dividend_raw"], index=index, name="dividend_raw")
        prices = pd.Series(data["spx_price_obs"], index=index, name="price_raw")
        states = {key: data[key] for key in ("m_t", "g_t", "h_t")}
        return states, dividends, prices
    states_path = cfg.get("states_npz")
    if not states_path:
        raise ValueError("data config must provide 'npz' or 'states_npz'")
    state_data = np.load(Path(states_path), allow_pickle=True)
    index = pd.to_datetime(state_data["dates"]) if "dates" in state_data else pd.RangeIndex(state_data["m_t"].shape[0], name="time")
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.Index(index, name="time")
    states = {key: state_data[key] for key in ("m_t", "g_t", "h_t")}

    if "sp500_price" in cfg and "sp500_dividend" in cfg:
        from hmc_gibbs.data.io import load_sp500_price_div

        sp500 = load_sp500_price_div(cfg["sp500_price"], cfg["sp500_dividend"])
        dividends = sp500["dividend_raw"].reindex(index)
        prices = sp500["price_raw"].reindex(index)
    else:
        def _load_series(path_key: str, column: str) -> pd.Series:
            csv_path = Path(cfg[path_key])
            df = pd.read_csv(csv_path)
            if "date" in df.columns:
                idx = pd.to_datetime(df["date"])
            else:
                idx = index
            value_col = column if column in df.columns else df.columns[-1]
            return pd.Series(df[value_col].to_numpy(), index=idx, name=column)

        dividends = _load_series("dividends_csv", "dividend_raw")
        prices = _load_series("prices_csv", "price_raw")

    dividends = dividends.astype(float)
    prices = prices.astype(float)
    dividends = dividends.reindex(index)
    prices = prices.reindex(index)
    return states, dividends, prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Model configuration YAML file.")
    parser.add_argument("--data", type=Path, required=True, help="Data configuration YAML file.")
    args = parser.parse_args()

    model_cfg = _load_yaml(args.config)
    data_cfg = _load_yaml(args.data)

    theta_cfg = model_cfg.get("equity", model_cfg)
    data_block = data_cfg.get("equity", data_cfg)

    theta = _build_equity_inputs(theta_cfg)
    states, dividends, prices = _load_states_and_observables(data_block)

    combined_config: Dict[str, Any] = {"model": model_cfg, "data": data_cfg}
    run_id = price_equity_and_export(theta, states, dividends, prices, combined_config)
    print(run_id)


if __name__ == "__main__":
    main()
