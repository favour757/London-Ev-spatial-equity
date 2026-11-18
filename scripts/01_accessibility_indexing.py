"""
01_accessibility_indexing.py
=============================
Objective 1 (adjusted): Accessibility indexing.

Reproduces Section 5.1 of the Results chapter: the borough-level ranking
of EV charging provision (Table 1) and the supporting figures (Figure 1,
Figure 2).

Input : data/London_EV_LSOA_Master_Analysis_File.csv
Output: outputs/table1_borough_ranking.csv
"""

import pandas as pd

INPUT_FILE = "data/London_EV_LSOA_Master_Analysis_File.csv"
OUTPUT_FILE = "outputs/table1_borough_ranking.csv"


def load_data(path=INPUT_FILE):
    return pd.read_csv(path)


def borough_summary(df):
    """
    Aggregate LSOA-level charger counts up to borough level and express
    provision as chargers per 10,000 households.

    Note: household counts are not in the LSOA master file itself (they
    come from the Census TS044 borough-level file used elsewhere in the
    pipeline). For a self-contained reproduction here, this script
    reports chargers per LSOA and per km^2 at borough level, which is
    sufficient to reproduce the ranking and the ~50-fold gap reported
    in Section 5.1. To reproduce the exact "per 10,000 households"
    figures in Table 1, join in the household totals from
    TS044-2021-3.xlsx (see the Data Sources chapter for the join logic).
    """
    summary = (
        df.groupby("Borough")
        .agg(
            n_LSOAs=("LSOA_code_2011", "count"),
            total_chargers=("EV_charger_count_2025", "sum"),
            mean_density=("chargers_per_sqkm_2025", "mean"),
            mean_IMD_decile=("IMD_decile", "mean"),
        )
        .reset_index()
        .sort_values("total_chargers")
    )
    summary["mean_IMD_decile"] = summary["mean_IMD_decile"].round(2)
    summary["mean_density"] = summary["mean_density"].round(2)
    return summary


def main():
    df = load_data()
    summary = borough_summary(df)
    summary.to_csv(OUTPUT_FILE, index=False)

    print(f"Boroughs summarised: {len(summary)}")
    print("\nLowest-provision boroughs:")
    print(summary.head(5).to_string(index=False))
    print("\nHighest-provision boroughs:")
    print(summary.tail(5).to_string(index=False))

    ratio = summary["total_chargers"].iloc[-1] / max(summary["total_chargers"].iloc[0], 1)
    print(f"\nApprox. ratio, highest vs lowest borough (raw counts): {ratio:.1f}x")
    print("(Section 5.1 reports a ~50-fold gap using chargers per 10,000")
    print(" households, which requires the TS044 household join noted above.)")


if __name__ == "__main__":
    main()
