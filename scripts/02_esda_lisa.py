"""
02_esda_lisa.py
=================
Objective 2: Exploratory Spatial Data Analysis (Global Moran's I and
Local Indicators of Spatial Association).

Reproduces Section 5.2 of the Results chapter: Global Moran's I
(Table 2) and the LISA hotspot/coldspot classification (Table 3).

Spatial weights use k-nearest-neighbours (k=8) rather than shared-edge
polygon contiguity, because the LSOA boundary source used elsewhere in
this project (see Methods 3.2) is a generalised, simplified geometry
whose polygons do not always share exact edges -- this breaks
contiguity-based weighting, so KNN (based on centroid proximity) is
used instead. This is documented as a deliberate, necessary choice in
Methods 3.2, not an oversight.

Input : data/London_EV_LSOA_Master_Analysis_File.csv
Output: outputs/table2_global_morans_i.csv
        outputs/table3_lisa_classification.csv
        outputs/lisa_labels.csv (per-LSOA cluster label, for mapping)

Requires: libpysal, esda  (pip install libpysal esda)
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

INPUT_FILE = "data/London_EV_LSOA_Master_Analysis_File.csv"
K_NEIGHBOURS = 8
N_PERMUTATIONS = 999
RANDOM_SEED = 42


def load_data(path=INPUT_FILE):
    return pd.read_csv(path)


def build_weights(df):
    coords = df[["centroid_lon", "centroid_lat"]].values
    w = KNN.from_array(coords, k=K_NEIGHBOURS)
    w.transform = "r"
    return w


def global_morans_i(df, w):
    results = []
    for col, label in [
        ("EV_charger_count_2025", "Charger count per LSOA"),
        ("chargers_per_sqkm_2025", "Charger density (per km2)"),
    ]:
        y = df[col].replace([np.inf], np.nan).fillna(0).values
        m = Moran(y, w, permutations=N_PERMUTATIONS)
        results.append(
            {
                "variable": label,
                "morans_i": round(m.I, 3),
                "p_value": m.p_sim,
                "z_score": round(m.z_sim, 2),
            }
        )
    return pd.DataFrame(results)


def local_lisa(df, w, value_col="chargers_per_sqkm_2025"):
    """
    Run Local Moran's I (LISA) and classify each LSOA into a cluster
    type. Returns the dataframe with two new columns: 'lisa_cluster'
    (5-category: High-High / Low-Low / High-Low / Low-High / Not
    significant) and 'lisa_cluster_3cat' (the 3-category HotSpot /
    Intermediate / ColdSpot display used in the report's figures,
    where the two rare outlier categories are folded into
    Intermediate -- see the Figure 3 caption for this note).
    """
    y = df[value_col].replace([np.inf], np.nan).fillna(0).values
    lisa = Moran_Local(y, w, permutations=N_PERMUTATIONS, seed=RANDOM_SEED)

    sig = lisa.p_sim < 0.05
    q = lisa.q  # 1=High-High, 2=Low-High, 3=Low-Low, 4=High-Low

    cluster5 = np.array(["Not significant"] * len(y), dtype=object)
    cluster5[(q == 1) & sig] = "High-High (hotspot)"
    cluster5[(q == 2) & sig] = "Low-High (outlier)"
    cluster5[(q == 3) & sig] = "Low-Low (coldspot)"
    cluster5[(q == 4) & sig] = "High-Low (outlier)"

    cluster3 = np.array(["Intermediate"] * len(y), dtype=object)
    cluster3[(q == 1) & sig] = "HotSpot"
    cluster3[(q == 3) & sig] = "ColdSpot"

    df = df.copy()
    df["lisa_cluster"] = cluster5
    df["lisa_cluster_3cat"] = cluster3
    df["lisa_p_sim"] = lisa.p_sim
    return df


def cluster_table(df):
    counts = df["lisa_cluster"].value_counts()
    total = len(df)
    table = counts.reset_index()
    table.columns = ["lisa_cluster", "n_LSOAs"]
    table["pct_of_london"] = (table["n_LSOAs"] / total * 100).round(1)
    return table


def main():
    df = load_data()
    w = build_weights(df)

    gmi = global_morans_i(df, w)
    gmi.to_csv("outputs/table2_global_morans_i.csv", index=False)
    print("Table 2 - Global Moran's I:")
    print(gmi.to_string(index=False))

    df_lisa = local_lisa(df, w)
    ctable = cluster_table(df_lisa)
    ctable.to_csv("outputs/table3_lisa_classification.csv", index=False)
    print("\nTable 3 - LISA classification:")
    print(ctable.to_string(index=False))

    df_lisa[["LSOA_code_2011", "Borough", "lisa_cluster", "lisa_cluster_3cat", "lisa_p_sim"]].to_csv(
        "outputs/lisa_labels.csv", index=False
    )
    print("\nSaved per-LSOA cluster labels to outputs/lisa_labels.csv")
    print("(used by 05_choropleth_maps.py to draw Figure 3)")


if __name__ == "__main__":
    main()
