"""
04_supplementary_analysis.py
==============================
Three supplementary checks, reproducing Section 5.4 of the Results
chapter: a Pearson correlation matrix (Table 6), a two-predictor OLS
robustness check with a VIF collinearity diagnostic (Table 7), and a
distribution/zero-inflation diagnostic on the charger count variable.

Input : data/London_EV_LSOA_Master_Analysis_File.csv
Output: outputs/table6_correlations.csv
        outputs/table7_two_predictor_ols.csv
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

INPUT_FILE = "data/London_EV_LSOA_Master_Analysis_File.csv"


def load_data(path=INPUT_FILE):
    return pd.read_csv(path)


def correlation_table(df):
    pairs = [
        ("EV_charger_count_2025", "IMD_decile"),
        ("EV_charger_count_2025", "borough_pct_terraced_flats_PROXY"),
        ("chargers_per_sqkm_2025", "IMD_decile"),
        ("chargers_per_sqkm_2025", "borough_pct_terraced_flats_PROXY"),
        ("IMD_decile", "borough_pct_terraced_flats_PROXY"),
    ]
    rows = []
    for a, b in pairs:
        sub = df[[a, b]].dropna()
        r, p = stats.pearsonr(sub[a], sub[b])
        rows.append({"variable_pair": f"{a} vs {b}", "r": round(r, 3), "p_value": p})
    return pd.DataFrame(rows)


def two_predictor_ols(df):
    sub = df.dropna(subset=["IMD_decile", "borough_pct_terraced_flats_PROXY", "EV_charger_count_2025"])
    X = sm.add_constant(sub[["IMD_decile", "borough_pct_terraced_flats_PROXY"]])
    y = sub["EV_charger_count_2025"]
    model = sm.OLS(y, X).fit()

    vif = pd.DataFrame()
    vif["variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    table = pd.DataFrame(
        {
            "predictor": ["const", "IMD_decile", "borough_pct_terraced_flats_PROXY"],
            "coefficient": model.params.values.round(3),
            "p_value": model.pvalues.values,
        }
    )
    print(f"R-squared: {model.rsquared:.3f}")
    print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(model.resid):.3f}")
    return table, vif, model


def distribution_diagnostics(df):
    y = df["EV_charger_count_2025"]
    return {
        "skewness": round(stats.skew(y), 2),
        "kurtosis": round(stats.kurtosis(y), 2),
        "n_zero": int((y == 0).sum()),
        "pct_zero": round((y == 0).mean() * 100, 1),
        "n_total": len(y),
    }


def main():
    df = load_data()

    corr = correlation_table(df)
    corr.to_csv("outputs/table6_correlations.csv", index=False)
    print("Table 6 - Correlations:")
    print(corr.to_string(index=False))

    print()
    table, vif, model = two_predictor_ols(df)
    table.to_csv("outputs/table7_two_predictor_ols.csv", index=False)
    print("\nTable 7 - Two-predictor OLS:")
    print(table.to_string(index=False))
    print("\nVariance Inflation Factors:")
    print(vif.to_string(index=False))

    diag = distribution_diagnostics(df)
    print("\nDistribution diagnostics (charger count):")
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
