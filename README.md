# Analysis Code — London EV Charging Spatial Equity Dissertation

This folder contains the code used to produce every table and figure in
Chapter 5 (Results) of the dissertation *A Spatial Analysis of
Socioeconomic Inequity in London's Public EV Charging Network*. Each
script was run and checked to reproduce the exact numbers reported in
the Results chapter — the "Verification" section below shows what each
one prints and how it lines up with the report.

This is the analysis code specifically. The earlier data-collection and
merging pipeline (combining the four raw data sources into the single
LSOA-level file used here) is documented separately in
`data_merging_pipeline.py`, already provided alongside the Data Sources
chapter — that script is not repeated here.

## What's in this folder

```
code_package/
├── README.md                      <- this file
├── requirements.txt                <- exact Python packages needed
├── data/
│   └── London_EV_LSOA_Master_Analysis_File.csv   <- the single input file
├── geo_data/
│   ├── lsoa_by_lad/*.json          <- LSOA boundary polygons (33 files, one per borough)
│   └── london_boroughs.geojson     <- borough outline polygons, for labels
├── scripts/
│   ├── 01_accessibility_indexing.py
│   ├── 02_esda_lisa.py
│   ├── 03_regression_ols_gwr.py
│   ├── 04_supplementary_analysis.py
│   └── 05_choropleth_maps.py
└── outputs/                        <- created automatically when you run the scripts
```

## Which script produces which part of the report

| Script | Report section | Produces |
|---|---|---|
| `01_accessibility_indexing.py` | 5.1 (Objective 1) | Borough ranking table |
| `02_esda_lisa.py` | 5.2 (Objective 2) | Table 2 (Global Moran's I), Table 3 (LISA classification) |
| `03_regression_ols_gwr.py` | 5.3 (Objective 3) | Table 4 (OLS), Table 5 (GWR diagnostics) |
| `04_supplementary_analysis.py` | 5.4 | Table 6 (correlations), Table 7 (two-predictor OLS + VIF), distribution diagnostics |
| `05_choropleth_maps.py` | Figures 1, 3, 6 | The three main map images |

Figures 2 and 4 (the borough bar chart and the scatter/histogram panel)
and the GWR local-R² map (Figure 5) are not included as separate
scripts here since they are direct plots of the tables/columns the
scripts above already produce (e.g. Figure 5 is `gwr_local_r2` from
`03_regression_ols_gwr.py`'s output, mapped using the same
`choropleth()` function pattern as `05_choropleth_maps.py`). Ask if
you'd like these built out as their own scripts too.

## How this was actually implemented (plain-English walkthrough)

1. **Starting point.** Everything here starts from one file:
   `London_EV_LSOA_Master_Analysis_File.csv` — one row per London LSOA
   (4,835 rows), already containing charger counts, IMD deprivation
   decile, and the borough-level housing proxy. This file is the
   output of the separate data-merging pipeline (see
   `data_merging_pipeline.py`), so none of the scripts here need to
   touch the four original raw data sources directly.

2. **Objective 1 — accessibility indexing** (`01_...py`) is the
   simplest step: group the LSOA rows up to borough level and rank
   boroughs by charger provision. No spatial statistics yet, just
   aggregation.

3. **Objective 2 — ESDA/LISA** (`02_...py`) is where the spatial
   statistics start. Two things happen here:
   - A **spatial weights matrix** is built from each LSOA's centroid
     coordinates, connecting every LSOA to its 8 nearest neighbours
     (`KNN` from the `libpysal` package). Nearest-neighbour weights
     were used instead of "shared border" weights because the LSOA
     boundary file is a simplified geometry that doesn't always share
     exact edges between neighbouring areas — with shared-border
     weights, almost every LSOA came back with zero neighbours, which
     breaks the statistic. This is explained in the dissertation's
     Methods chapter (Section 3.2), not something specific to the code.
   - **Global Moran's I** (one number for the whole of London) and
     **Local Moran's I / LISA** (one number per LSOA) are then computed
     using the `esda` package, which is the actual statistical engine
     — the script itself is mostly just organising inputs and outputs
     around those two function calls.

4. **Objective 3 — OLS and GWR** (`03_...py`) fits two models on the
   same data: a standard regression (`statsmodels`) that assumes one
   fixed relationship for all of London, and a geographically weighted
   regression (`mgwr`) that fits a separate mini-regression around
   every LSOA. The GWR bandwidth (how many neighbouring LSOAs each
   local regression uses) is not chosen by hand — the `Sel_BW` function
   searches for the bandwidth that minimises AICc, a statistic that
   balances how well the model fits against how complex it is. This
   step is the slowest in the whole pipeline (a few minutes), because
   it has to test many candidate bandwidths across 4,835 locations
   before settling on the best one.

5. **Supplementary checks** (`04_...py`) are simpler statistics
   (correlations, a second OLS model, a variance inflation factor
   check, and some basic distribution statistics) run directly on the
   same input file, added after the fact to stress-test the main
   findings rather than being part of the original three objectives.

6. **Maps** (`05_...py`) take the LSOA-level values (density, IMD
   decile, LISA cluster label) and draw them as coloured polygons using
   `matplotlib` and `shapely`, then overlay borough outlines and labels
   from a separate borough boundary file so the map is readable at a
   glance. This is plotting code, not statistical code — all the actual
   analysis has already happened by this point.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The GWR step (`03_regression_ols_gwr.py`) is
the slowest — expect a few minutes on a normal laptop.

## Running the scripts

Run them in order from inside this folder, since later scripts read
files that earlier ones produce:

```bash
mkdir -p outputs
python3 scripts/01_accessibility_indexing.py
python3 scripts/02_esda_lisa.py
python3 scripts/03_regression_ols_gwr.py
python3 scripts/04_supplementary_analysis.py
python3 scripts/05_choropleth_maps.py
```

`02_esda_lisa.py` must run before `05_choropleth_maps.py`, since the
map script reads the LISA cluster labels it produces
(`outputs/lisa_labels.csv`). The other scripts are independent of each
other.

## Verification

Each script was run against the actual data and checked to reproduce
the report's numbers exactly:

| Statistic | Reported in report | Reproduced by running this code |
|---|---|---|
| Global Moran's I (count) | 0.579 | 0.579 |
| Global Moran's I (density) | 0.663 | 0.663 |
| LISA hotspots | 545 LSOAs | 545 |
| LISA coldspots | 1,786 LSOAs | 1,786 |
| OLS R² | 0.004 | 0.004 |
| OLS coefficient | +0.098 | 0.098 |
| GWR bandwidth | 52 | 52 |
| GWR R² | 0.647 | 0.647 |
| Correlation, count vs IMD | r = 0.060 | 0.060 |
| Two-predictor OLS R² | 0.190 | 0.190 |
| Durbin-Watson | 0.933 | 0.933 |
| % LSOAs with zero chargers | 41.4% | 41.4% |

## A note on the boundary files

The LSOA and borough polygon files in `geo_data/` are official ONS/GLA
geography, obtained via public GitHub mirrors
(`martinjc/UK-GeoJSON` and `radoi90/housequest-data`) rather than
downloaded directly from the ONS Geography Portal, because the portal
was not reachable from the environment this project was originally
built in. If you're rerunning this with fresh access to the official
source, downloading the LSOA (2011) and borough boundaries directly
from https://geoportal.statistics.gov.uk/ and
https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london
will give the same result — these are copies of the same official
data, not an independent source.

## Known limitations, carried over from the report

- The housing-constraint variable (`borough_pct_terraced_flats_PROXY`)
  is still a borough-level figure applied to every LSOA in that
  borough, not true LSOA-level data — this is why it's excluded from
  the primary GWR model in `03_regression_ols_gwr.py` and only appears
  in the supplementary two-predictor model in `04_supplementary_analysis.py`.
- The charger dataset reflects an estimated 40–46% of London's total
  public charging network (see Data Sources chapter), so relative
  patterns across LSOAs are more reliable than absolute counts.
