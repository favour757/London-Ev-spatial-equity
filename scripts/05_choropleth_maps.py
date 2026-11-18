"""
05_choropleth_maps.py
========================
Generates the three main map figures used in the Results chapter:
  Figure 1 - EV charger density choropleth
  Figure 3 - LISA hotspot/coldspot cluster map
  Figure 6 - IMD 2019 decile choropleth

All three share a common cartographic style: London borough boundaries
and labels, a north arrow, and a consistent title/legend format.

This script depends on the outputs of 02_esda_lisa.py (for the LISA
labels used in Figure 3) and 03_regression_ols_gwr.py (if you also
want to regenerate the GWR local R^2 map, Figure 5 -- see the commented
section at the bottom).

LSOA and borough boundary polygons are provided in geo_data/ as a
snapshot from a public GitHub mirror of ONS geography
(martinjc/UK-GeoJSON) and GLA borough boundaries
(radoi90/housequest-data), used because the official ONS Geography
Portal was not directly reachable from the environment this project
was built in. For a fresh run, or to use the authoritative source
directly, download the same boundaries from
https://geoportal.statistics.gov.uk/ instead.

Input : data/London_EV_LSOA_Master_Analysis_File.csv
        outputs/lisa_labels.csv        (produced by 02_esda_lisa.py)
        geo_data/lsoa_by_lad/*.json
        geo_data/london_boroughs.geojson
Output: outputs/figure1_density.png
        outputs/figure3_lisa.png
        outputs/figure6_imd.png
"""

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import shape

LONDON_LADS = {
    "E09000002": "Barking and Dagenham", "E09000003": "Barnet", "E09000004": "Bexley",
    "E09000005": "Brent", "E09000006": "Bromley", "E09000007": "Camden",
    "E09000008": "Croydon", "E09000009": "Ealing", "E09000010": "Enfield",
    "E09000011": "Greenwich", "E09000012": "Hackney", "E09000013": "Hammersmith and Fulham",
    "E09000014": "Haringey", "E09000015": "Harrow", "E09000016": "Havering",
    "E09000017": "Hillingdon", "E09000018": "Hounslow", "E09000019": "Islington",
    "E09000020": "Kensington and Chelsea", "E09000021": "Kingston upon Thames",
    "E09000022": "Lambeth", "E09000023": "Lewisham", "E09000024": "Merton",
    "E09000025": "Newham", "E09000026": "Redbridge", "E09000027": "Richmond upon Thames",
    "E09000028": "Southwark", "E09000029": "Sutton", "E09000030": "Tower Hamlets",
    "E09000031": "Waltham Forest", "E09000032": "Wandsworth", "E09000033": "Westminster",
    "E09000001": "City of London",
}

GEO_DIR = "geo_data/lsoa_by_lad"
BOROUGH_FILE = "geo_data/london_boroughs.geojson"


def load_lsoa_geometry():
    geom_lookup = {}
    for code in LONDON_LADS:
        with open(f"{GEO_DIR}/{code}.json") as f:
            gj = json.load(f)
        for feat in gj["features"]:
            geom_lookup[feat["properties"]["LSOA11CD"]] = shape(feat["geometry"])
    return geom_lookup


def load_boroughs():
    with open(BOROUGH_FILE) as f:
        return json.load(f)


def draw_base(ax, boroughs):
    """Shared cartographic elements: borough outlines, labels, north arrow."""
    for feat in boroughs["features"]:
        g = shape(feat["geometry"])
        name = feat["properties"]["name"]
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=0.6, zorder=5)
        label_pt = g.representative_point()
        ax.text(
            label_pt.x, label_pt.y, name, fontsize=5.8, ha="center", va="center",
            color="black", zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
    ax.set_xlim(-0.52, 0.33)
    ax.set_ylim(51.28, 51.70)
    ax.set_aspect(1.6)
    ax.axis("off")
    ax.annotate(
        "N", xy=(0.965, 0.93), xytext=(0.965, 0.80), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        ha="center", fontsize=11, fontweight="bold",
    )


def choropleth(df, geom_lookup, boroughs, value_col, title, cmap, vmin, vmax,
               cbar_label, out_path, discrete_ticks=None):
    fig, ax = plt.subplots(figsize=(11, 9))
    patches, values = [], []
    for code, val in zip(df["LSOA_code_2011"], df[value_col]):
        g = geom_lookup.get(code)
        if g is None or pd.isna(val):
            continue
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for poly in polys:
            patches.append(MplPolygon(list(poly.exterior.coords)))
            values.append(val)
    pc = PatchCollection(patches, cmap=cmap, edgecolor="#4d4d4d", linewidths=0.1)
    pc.set_array(np.array(values))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    draw_base(ax, boroughs)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    cbar = fig.colorbar(pc, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    if discrete_ticks is not None:
        cbar.set_ticks(discrete_ticks)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {out_path}")


def lisa_cluster_map(df_lisa, geom_lookup, boroughs, out_path):
    colors_map = {"HotSpot": "#D7191C", "Intermediate": "#DED6B8", "ColdSpot": "#2C5F8A"}
    fig, ax = plt.subplots(figsize=(11, 9))
    patches, colors = [], []
    for code, cat in zip(df_lisa["LSOA_code_2011"], df_lisa["lisa_cluster_3cat"]):
        g = geom_lookup.get(code)
        if g is None:
            continue
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for poly in polys:
            patches.append(MplPolygon(list(poly.exterior.coords)))
            colors.append(colors_map[cat])
    pc = PatchCollection(patches, facecolor=colors, edgecolor="#4d4d4d", linewidths=0.15)
    ax.add_collection(pc)
    draw_base(ax, boroughs)
    ax.set_title("EV Charger Density Hotspot Analysis \u2014 London LSOAs (2025)",
                 fontsize=13, fontweight="bold", pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors_map[k], edgecolor="#4d4d4d", linewidth=0.4)
               for k in ["HotSpot", "Intermediate", "ColdSpot"]]
    labels = ["1.96 to 99.00  HotSpot", "-1.96 to 1.96  Intermediate", "-99.00 to -1.96  ColdSpot"]
    leg = ax.legend(handles, labels, loc="lower left", fontsize=9, framealpha=0.95,
                     title="Local Moran's I (z-score)", title_fontsize=9, borderpad=0.8)
    leg.get_frame().set_edgecolor("#999999")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {out_path}")


def main():
    df = pd.read_csv("data/London_EV_LSOA_Master_Analysis_File.csv")
    geom_lookup = load_lsoa_geometry()
    boroughs = load_boroughs()

    # Figure 1: charger density
    choropleth(
        df, geom_lookup, boroughs, "chargers_per_sqkm_2025",
        "EV Charger Density by LSOA, 2025 (chargers per km\u00b2)",
        "YlOrRd", 0, np.percentile(df["chargers_per_sqkm_2025"].dropna(), 95),
        "Chargers / km\u00b2", "outputs/figure1_density.png",
    )

    # Figure 6: IMD decile
    choropleth(
        df, geom_lookup, boroughs, "IMD_decile",
        "Index of Multiple Deprivation 2019 (Decile)\n1 = most deprived, 10 = least deprived",
        "RdYlGn", 1, 10, "IMD Decile", "outputs/figure6_imd.png",
        discrete_ticks=list(range(1, 11)),
    )

    # Figure 3: LISA cluster map (requires 02_esda_lisa.py to have been run first)
    try:
        df_lisa = pd.read_csv("outputs/lisa_labels.csv")
        lisa_cluster_map(df_lisa, geom_lookup, boroughs, "outputs/figure3_lisa.png")
    except FileNotFoundError:
        print("outputs/lisa_labels.csv not found -- run 02_esda_lisa.py first to generate Figure 3.")


if __name__ == "__main__":
    main()
