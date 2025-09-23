from __future__ import annotations

import argparse
import numpy as np

from tsm.equity_pricer import EquityPricingInputs, price_equity_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        required=True,
        help="Input .npz with posterior draw and state series (m_t, g_t, h_t, dividends, prices).",
    )
    parser.add_argument("--out", required=True, help="Output .npz path for computed series.")
    parser.add_argument("--n_max", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--min_terms", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    d_m = int(data["d_m"])
    d_g = int(data["d_g"])
    d_h = int(data["d_h"])

    inputs = EquityPricingInputs(
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

    series = price_equity_series(
        inputs,
        m_t=data["m_t"],
        g_t=data["g_t"],
        h_t=data["h_t"],
        dividend_raw=data["dividend_raw"],
        spx_price_obs=data["spx_price_obs"],
        n_max=args.n_max,
        rtol=args.rtol,
        atol=args.atol,
        min_terms=args.min_terms,
    )

    np.savez(args.out, **series)
    print(f"[equity] wrote: {args.out}")


if __name__ == "__main__":
    main()
