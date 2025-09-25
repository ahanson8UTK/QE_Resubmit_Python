from __future__ import annotations

import numpy as np
import pandas as pd

import h5py

from data.market_data import get_fac_draw_slice, load_market_data


def test_load_market_data(tmp_path):
    bond_csv = tmp_path / "bond_yields.csv"
    bond_df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-02-01"],
            "TB3MS": [1.0, 3.0],
            "TB6MS": [2.0, 4.0],
        }
    )
    bond_df.to_csv(bond_csv, index=False)

    sp500_csv = tmp_path / "sp500_data.csv"
    sp500_df = pd.DataFrame(
        {
            "Price": [118.4, 114.2, 112.4, 110.3],
            "Dividend": [3.16, 3.16, 3.17, 3.19],
            "DATE": ["", "1973-02-01", "1973-03-01", "1973-04-01"],
            "log_div_grow": [np.nan, 0.002, 0.003, 0.004],
        }
    )
    sp500_df.to_csv(sp500_csv, index=False)

    data = load_market_data(bond_csv, sp500_csv)

    np.testing.assert_allclose(data.bond_yields, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(data.sp500_price, np.array([112.4, 110.3]))
    np.testing.assert_allclose(data.sp500_div, np.array([3.17, 3.19]))
    np.testing.assert_allclose(data.log_div_grow, np.array([0.003, 0.004]))


def test_get_fac_draw_slice(tmp_path):
    mat_path = tmp_path / "fac_draws_K4.mat"
    data = np.arange(4 * 547 * 3, dtype=float).reshape(4, 547, 3)

    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset("draws", data=data)

    slice_ = get_fac_draw_slice(mat_path, 1, dataset="draws")
    np.testing.assert_allclose(slice_, data[:, :, 1])

    # Default dataset selection should also work.
    slice_default = get_fac_draw_slice(mat_path, 2)
    np.testing.assert_allclose(slice_default, data[:, :, 2])
