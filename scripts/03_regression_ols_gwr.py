"""
03_regression_ols_gwr.py
===========================
Objective 3: comparing a global model (OLS) against a locally-varying
model (GWR) to test whether the deprivation-access relationship is
uniform across London or varies by area.

Reproduces Section 5.3 of the Results chapter: the global OLS result
(Table 4) and the GWR diagnostics (Table 5).

As explained in Section 1.4 and Section 5.3, the primary model here
uses IMD 2019 decile as its ONLY predictor. The housing-constraint
variable (borough_pct_terraced_flats_PROXY) is deliberately excluded
from this model -- see 04_supplementary_analysis.py for the
two-predictor version, which is reported separately as a robustness
check precisely because that variable is still a borough-level proxy
rather than true LSOA-level data.

Input : data/London_EV_LSOA_Master_Analysis_File.csv
Output: outputs/table4_ols_baseline.csv
        outputs/table5_gwr_diagnostics.csv
        outputs/gwr_local_results.csv (per-LSOA local R^2 / coefficients,
                                        for mapping Figure 5)

Requires: statsmodels, mgwr  (pip install statsmodels mgwr)
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

INPUT_FILE = "data/London_EV_LSOA_Master_Analysis_File.csv"


def load_data(path=INPUT_FILE):
    df = pd.read_csv(path)
    return df.dropna(subset=["IMD_decile", "EV_charger_count_2025"]).reset_index(drop=True)


def run_ols(df):
    X = sm.add_constant(df["IMD_decile"])
    y = df["EV_charger_count_2025"]
    model = sm.OLS(y, X).fit()
    table = pd.DataFrame(
        [
            {
                "predictor": "IMD 2019 decile",
                "coefficient": round(model.params["IMD_decile"], 3),
                "p_value": model.pvalues["IMD_decile"],
                "r_squared": round(model.rsquared, 3),
            }
        ]
    )
    return model, table


def run_gwr(df):
    """
    Fits GWR with an adaptive bandwidth selected automatically via
    AICc minimisation (Sel_BW), exactly as described in Methods 3.2.
    This is the slowest step in the pipeline (bandwidth search over
    4,835 LSOAs) -- expect this to take several minutes.
    """
    X = df[["IMD_decile"]].values
    y = df["EV_charger_count_2025"].values.reshape(-1, 1)
    coords = list(zip(df["centroid_lon"], df["centroid_lat"]))

    selector = Sel_BW(coords, y, X)
    bw = selector.search(criterion="AICc", bw_min=50)

    gwr_model = GWR(coords, y, X, bw)
    results = gwr_model.fit()

    diagnostics = pd.DataFrame(
        [
            {
                "bandwidth_nearest_neighbours": round(bw),
                "overall_r_squared": round(results.R2, 3),
                "adjusted_r_squared": round(results.adj_R2, 3),
                "aicc": round(results.aicc, 1),
            }
        ]
    )

    local_out = df[["LSOA_code_2011", "Borough"]].copy()
    local_out["gwr_local_r2"] = results.localR2.flatten()
    local_out["gwr_local_intercept"] = results.params[:, 0]
    local_out["gwr_local_coef_imd"] = results.params[:, 1]
    local_out["gwr_local_t_imd"] = results.tvalues[:, 1]

    return results, diagnostics, local_out


def main():
    df = load_data()

    ols_model, ols_table = run_ols(df)
    ols_table.to_csv("outputs/table4_ols_baseline.csv", index=False)
    print("Table 4 - Global OLS:")
    print(ols_table.to_string(index=False))

    print("\nRunning GWR (this can take a few minutes for 4,835 LSOAs)...")
    gwr_results, gwr_table, local_out = run_gwr(df)
    gwr_table.to_csv("outputs/table5_gwr_diagnostics.csv", index=False)
    local_out.to_csv("outputs/gwr_local_results.csv", index=False)

    print("\nTable 5 - GWR diagnostics:")
    print(gwr_table.to_string(index=False))
    print(
        f"\nLocal R^2 range: {local_out['gwr_local_r2'].min():.3f} to "
        f"{local_out['gwr_local_r2'].max():.3f}"
    )
    print("Saved per-LSOA local GWR results to outputs/gwr_local_results.csv")
    print("(used by 05_choropleth_maps.py to draw Figure 5)")


if __name__ == "__main__":
    main()
