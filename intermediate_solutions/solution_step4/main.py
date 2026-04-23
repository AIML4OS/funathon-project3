# %%
# ============================================
# Statistics on land-cover predictions
# ============================================

# %%
# ============================================
# Imports
# ============================================
import json

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from folium.plugins import HeatMap
from great_tables import GT

# %%
# ============================================
# Setup — API and class definitions
# ============================================

api_url = "https://funathon-2026-project3.lab.sspcloud.fr"

class_names = {
    1:  "Sealed",
    2:  "Woody – needle leaved",
    3:  "Woody – broadleaved deciduous",
    4:  "Woody – broadleaved evergreen",
    5:  "Low-growing woody plants",
    6:  "Permanent herbaceous",
    7:  "Periodically herbaceous",
    8:  "Lichens and mosses",
    9:  "Non- and sparsely-vegetated",
    10: "Water",
}

label_to_color = {
    1: "#FF0100", 2: "#238B23", 3: "#80FF00", 4: "#00FF00", 5: "#804000",
    6: "#CCF24E", 7: "#FEFF80", 8: "#FF81FF", 9: "#BFBFBF", 10: "#0080FF",
}


# %%
# ============================================
# Produce tile predictions (2024)
# ============================================

image_filepath = (
    "projet-formation/diffusion/funathon/"
    "2026/project3/data/images/LU000/2024/"
    "4042000_2951690_0_637.tif"
)

response_pred = requests.get(
    f"{api_url}/predict_image",
    params={"image": image_filepath, "polygons": True},
)

gdf_tile = gpd.GeoDataFrame.from_features(
    json.loads(response_pred.json())["features"],
    crs="EPSG:3035",
)

gdf_tile["area_m2"]    = gdf_tile.geometry.area
gdf_tile["area_km2"]   = gdf_tile["area_m2"] / 1e6
gdf_tile["class_name"] = gdf_tile["label"].map(class_names)


# %%
# ============================================================
# EXERCISE 1 — Compute land-cover area statistics for a single tile
# ============================================================
#
# Goal: Group the predicted polygons by land-cover class and compute the
# total area, number of polygons, mean and max polygon size for each class.
# Then calculate each class's share of the total tile area.
# Display the result as a GT table with formatted columns and colour-coded
# shares.
#
# Steps:
#   1. Group gdf_tile by label and class_name using .groupby().agg()
#   2. Compute n_polygons, total_area_km2, mean_polygon_area_m2,
#      max_polygon_area_m2
#   3. Compute total_km2 as the sum of all class areas
#   4. Add a share_pct column as the percentage of total area
#   5. Display with GT — format numbers and colour share_pct
# ============================================================

stats_tile = (
    gdf_tile.groupby(["label", "class_name"])
    .agg(
        n_polygons           = ("geometry", "__"),  # TODO: aggregation function (str)
        total_area_km2       = ("area_km2", "__"),  # TODO: aggregation function (str)
        mean_polygon_area_m2 = ("area_m2",  "__"),  # TODO: aggregation function (str)
        max_polygon_area_m2  = ("area_m2",  "__"),  # TODO: aggregation function (str)
    )
    .reset_index()
    .sort_values("total_area_km2", ascending=False)
)

total_km2 = __  # TODO: sum of all total_area_km2
stats_tile["share_pct"] = (__ / total_km2 * 100).round(2)  # TODO: column to normalise

(
    GT(stats_tile, rowname_col="class_name")
    .tab_header(title="Land-cover statistics — single tile (LU000, 2024)")
    .cols_label(
        label                = "Label",
        n_polygons           = "N polygons",
        total_area_km2       = "Total area (km2)",
        mean_polygon_area_m2 = "Mean area (m2)",
        max_polygon_area_m2  = "Max area (m2)",
        share_pct            = "Share (%)",
    )
    .fmt_number(columns=["total_area_km2"], decimals=2)
    .fmt_number(columns=["mean_polygon_area_m2", "max_polygon_area_m2"], decimals=0)
    .fmt_number(columns=["share_pct"], decimals=1)
    .data_color(columns=["share_pct"], palette=["white", "steelblue"])
)

# ------------------------------------------------------------
# HINT — Exercise 1
# ------------------------------------------------------------
# - Use "count" for n_polygons, "sum" for totals, "mean" and "max"
#   for polygon size statistics.
# - total_km2 = stats_tile["total_area_km2"].sum()
# - share_pct = stats_tile["total_area_km2"] / total_km2 * 100
# - Pass rowname_col="class_name" to GT() to use class names as row labels.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 1
# ------------------------------------------------------------
# stats_tile = (
#     gdf_tile.groupby(["label", "class_name"])
#     .agg(
#         n_polygons           = ("geometry", "count"),
#         total_area_km2       = ("area_km2", "sum"),
#         mean_polygon_area_m2 = ("area_m2",  "mean"),
#         max_polygon_area_m2  = ("area_m2",  "max"),
#     )
#     .reset_index()
#     .sort_values("total_area_km2", ascending=False)
# )
#
# total_km2 = stats_tile["total_area_km2"].sum()
# stats_tile["share_pct"] = (stats_tile["total_area_km2"] / total_km2 * 100).round(2)
#
# (
#     GT(stats_tile, rowname_col="class_name")
#     .tab_header(title="Land-cover statistics — single tile (LU000, 2024)")
#     .cols_label(
#         label                = "Label",
#         n_polygons           = "N polygons",
#         total_area_km2       = "Total area (km2)",
#         mean_polygon_area_m2 = "Mean area (m2)",
#         max_polygon_area_m2  = "Max area (m2)",
#         share_pct            = "Share (%)",
#     )
#     .fmt_number(columns=["total_area_km2"], decimals=2)
#     .fmt_number(columns=["mean_polygon_area_m2", "max_polygon_area_m2"], decimals=0)
#     .fmt_number(columns=["share_pct"], decimals=1)
#     .data_color(columns=["share_pct"], palette=["white", "steelblue"])
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 — Summarise key land-cover groups
# ============================================================
#
# Goal: Aggregate the per-class statistics into four meaningful land-cover
# groups and display a concise summary table with area and share,
# colour-coded by sealed surface intensity.
#
# Groups:
#   - Sealed (built-up) : label 1
#   - Forest            : labels 2, 3, 4
#   - Agricultural      : labels 6, 7
#   - Water             : label 10
#
# Steps:
#   1. Filter stats_tile by label for each group using .loc[] and .isin()
#   2. Sum total_area_km2 for each group
#   3. Build a summary pd.DataFrame with area_km2 and share_pct
#   4. Display with GT and colour share_pct
# ============================================================

sealed_km2 = stats_tile.loc[stats_tile["label"] == __,              "total_area_km2"].sum()  # TODO: label
forest_km2 = stats_tile.loc[stats_tile["label"].isin([__, __, __]), "total_area_km2"].sum()  # TODO: labels
agri_km2   = stats_tile.loc[stats_tile["label"].isin([__, __]),     "total_area_km2"].sum()  # TODO: labels
water_km2  = stats_tile.loc[stats_tile["label"] == __,              "total_area_km2"].sum()  # TODO: label

summary_df = pd.DataFrame({
    "Group":    ["Sealed (built-up)", "Forest", "Agricultural", "Water"],
    "area_km2": [sealed_km2, forest_km2, agri_km2, water_km2],
    "share_pct": [
        sealed_km2 / total_km2 * 100,
        forest_km2 / total_km2 * 100,
        agri_km2   / total_km2 * 100,
        water_km2  / total_km2 * 100,
    ],
})

(
    GT(summary_df, rowname_col="Group")
    .tab_header(title="Key land-cover groups — single tile (LU000, 2024)")
    .cols_label(area_km2="Area (km2)", share_pct="Share (%)")
    .fmt_number(columns=["area_km2"], decimals=2)
    .fmt_number(columns=["share_pct"], decimals=1)
    .data_color(columns=["share_pct"], palette=["white", "#FF0100"])
)

# ------------------------------------------------------------
# HINT — Exercise 2
# ------------------------------------------------------------
# - Sealed = label 1, Forest = labels 2, 3, 4,
#   Agricultural = labels 6, 7, Water = label 10.
# - Use .loc[condition, "total_area_km2"].sum() to filter and aggregate.
# - Build a small pd.DataFrame before passing it to GT().
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2
# ------------------------------------------------------------
# sealed_km2 = stats_tile.loc[stats_tile["label"] == 1,             "total_area_km2"].sum()
# forest_km2 = stats_tile.loc[stats_tile["label"].isin([2, 3, 4]),  "total_area_km2"].sum()
# agri_km2   = stats_tile.loc[stats_tile["label"].isin([6, 7]),     "total_area_km2"].sum()
# water_km2  = stats_tile.loc[stats_tile["label"] == 10,            "total_area_km2"].sum()
#
# summary_df = pd.DataFrame({
#     "Group":    ["Sealed (built-up)", "Forest", "Agricultural", "Water"],
#     "area_km2": [sealed_km2, forest_km2, agri_km2, water_km2],
#     "share_pct": [
#         sealed_km2 / total_km2 * 100,
#         forest_km2 / total_km2 * 100,
#         agri_km2   / total_km2 * 100,
#         water_km2  / total_km2 * 100,
#     ],
# })
#
# (
#     GT(summary_df, rowname_col="Group")
#     .tab_header(title="Key land-cover groups — single tile (LU000, 2024)")
#     .cols_label(area_km2="Area (km2)", share_pct="Share (%)")
#     .fmt_number(columns=["area_km2"], decimals=2)
#     .fmt_number(columns=["share_pct"], decimals=1)
#     .data_color(columns=["share_pct"], palette=["white", "#FF0100"])
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 3 — Plot the land-cover distribution for the tile
# ============================================================
#
# Goal: Create a horizontal bar chart showing the total area (km2) per
# land-cover class for the single tile.
#
# Steps:
#   1. Set class_name as the index of stats_tile
#   2. Select the total_area_km2 column and sort values ascending
#   3. Plot as a horizontal bar chart with kind="barh"
#   4. Add axis label and title
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))

stats_tile.set_index("__")["__"].sort_values().plot(  # TODO: index column, value column
    kind="barh", ax=ax, color="steelblue"
)

ax.set_xlabel("__")  # TODO: axis label (str)
ax.set_title("__")   # TODO: chart title (str)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 3
# ------------------------------------------------------------
# - Index on "class_name", plot "total_area_km2".
# - ax.set_xlabel("Area (km2)") and give a descriptive title.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 3
# ------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 4))
#
# stats_tile.set_index("class_name")["total_area_km2"].sort_values().plot(
#     kind="barh", ax=ax, color="steelblue"
# )
#
# ax.set_xlabel("Area (km2)")
# ax.set_title("Land-cover distribution — single tile (LU000, 2024)")
# plt.tight_layout()
# plt.show()
# ------------------------------------------------------------


# %%
# ============================================
# Produce NUTS3 predictions (2024)
# ============================================

response_nuts = requests.get(
    f"{api_url}/predict_nuts",
    params={"nuts_id": "LU000", "year": 2024},
)

gdf_nuts = gpd.GeoDataFrame.from_features(
    json.loads(response_nuts.json()["predictions"])["features"],
    crs="EPSG:3035",
)

gdf_nuts["area_m2"]    = gdf_nuts.geometry.area
gdf_nuts["area_km2"]   = gdf_nuts["area_m2"] / 1e6
gdf_nuts["class_name"] = gdf_nuts["label"].map(class_names)


# %%
# ============================================================
# EXERCISE 4 — Compute land-cover statistics for the full NUTS3 region
# ============================================================
#
# Goal: Apply the same aggregation as Exercise 1, but on gdf_nuts.
# Then compare class shares between the single tile and the full region
# in a GT table.
#
# Steps:
#   1. Repeat the .groupby().agg() pattern from Exercise 1 on gdf_nuts
#   2. Compute total_nuts_km2 and add share_pct
#   3. Display the stats table with GT
#   4. Build a side-by-side comparison of tile vs. NUTS3 shares with GT
# ============================================================

stats_nuts = (
    gdf_nuts.groupby(["label", "class_name"])
    .agg(
        n_polygons           = ("geometry", "count"),
        total_area_km2       = ("area_km2", "sum"),
        mean_polygon_area_m2 = ("area_m2",  "mean"),
        max_polygon_area_m2  = ("area_m2",  "max"),
    )
    .reset_index()
    .sort_values("total_area_km2", ascending=False)
)

total_nuts_km2 = __  # TODO: sum of total_area_km2
stats_nuts["share_pct"] = (__ / total_nuts_km2 * 100).round(2)  # TODO: column

(
    GT(stats_nuts, rowname_col="class_name")
    .tab_header(title="Land-cover statistics — LU000 NUTS3 (2024)")
    .cols_label(
        label                = "Label",
        n_polygons           = "N polygons",
        total_area_km2       = "Total area (km2)",
        mean_polygon_area_m2 = "Mean area (m2)",
        max_polygon_area_m2  = "Max area (m2)",
        share_pct            = "Share (%)",
    )
    .fmt_number(columns=["total_area_km2"], decimals=2)
    .fmt_number(columns=["mean_polygon_area_m2", "max_polygon_area_m2"], decimals=0)
    .fmt_number(columns=["share_pct"], decimals=1)
    .data_color(columns=["share_pct"], palette=["white", "steelblue"])
)

tile_shares = stats_tile.set_index("class_name")["share_pct"].rename("__")  # TODO: column name (str)
nuts_shares = stats_nuts.set_index("class_name")["share_pct"].rename("__")  # TODO: column name (str)

comparison = pd.concat([__, __], axis=1).fillna(0).reset_index()  # TODO: tile_shares, nuts_shares

(
    GT(comparison, rowname_col="class_name")
    .tab_header(title="Land-cover share — tile vs. NUTS3 (2024)")
    .cols_label(**{"tile_share_pct": "Tile (%)", "nuts3_share_pct": "NUTS3 (%)"})
    .fmt_number(decimals=1)
    .data_color(palette=["white", "steelblue"])
)

# ------------------------------------------------------------
# HINT — Exercise 4
# ------------------------------------------------------------
# - Same pattern as Exercise 1 — just replace gdf_tile with gdf_nuts.
# - Rename the share columns "tile_share_pct" and "nuts3_share_pct".
# - Call .reset_index() on the comparison DataFrame before passing to GT().
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 4
# ------------------------------------------------------------
# total_nuts_km2 = stats_nuts["total_area_km2"].sum()
# stats_nuts["share_pct"] = (stats_nuts["total_area_km2"] / total_nuts_km2 * 100).round(2)
#
# tile_shares = stats_tile.set_index("class_name")["share_pct"].rename("tile_share_pct")
# nuts_shares = stats_nuts.set_index("class_name")["share_pct"].rename("nuts3_share_pct")
# comparison  = pd.concat([tile_shares, nuts_shares], axis=1).fillna(0).reset_index()
#
# (
#     GT(stats_nuts, rowname_col="class_name")
#     .tab_header(title="Land-cover statistics — LU000 NUTS3 (2024)")
#     .cols_label(
#         label                = "Label",
#         n_polygons           = "N polygons",
#         total_area_km2       = "Total area (km2)",
#         mean_polygon_area_m2 = "Mean area (m2)",
#         max_polygon_area_m2  = "Max area (m2)",
#         share_pct            = "Share (%)",
#     )
#     .fmt_number(columns=["total_area_km2"], decimals=2)
#     .fmt_number(columns=["mean_polygon_area_m2", "max_polygon_area_m2"], decimals=0)
#     .fmt_number(columns=["share_pct"], decimals=1)
#     .data_color(columns=["share_pct"], palette=["white", "steelblue"])
# )
#
# (
#     GT(comparison, rowname_col="class_name")
#     .tab_header(title="Land-cover share — tile vs. NUTS3 (2024)")
#     .cols_label(**{"tile_share_pct": "Tile (%)", "nuts3_share_pct": "NUTS3 (%)"})
#     .fmt_number(decimals=1)
#     .data_color(palette=["white", "steelblue"])
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 5 — Plot area and share for the NUTS3 region
# ============================================================
#
# Goal: Create a two-panel figure showing (left) the total area in km2 and
# (right) the share in % for each land-cover class across the full NUTS3
# region.
#
# Steps:
#   1. Create a figure with 2 side-by-side subplots
#   2. Plot total_area_km2 as a horizontal bar chart on the left panel
#   3. Plot share_pct as a horizontal bar chart on the right panel
#   4. Add labels and titles to both panels
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stats_nuts.set_index("class_name")["__"].sort_values().plot(  # TODO: column for left panel
    kind="barh", ax=axes[0], color="steelblue"
)
axes[0].set_xlabel("__")  # TODO: axis label
axes[0].set_title("__")   # TODO: title

stats_nuts.set_index("class_name")["__"].sort_values().plot(  # TODO: column for right panel
    kind="barh", ax=axes[1], color="darkorange"
)
axes[1].set_xlabel("__")  # TODO: axis label
axes[1].set_title("__")   # TODO: title

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 5
# ------------------------------------------------------------
# - Left panel: "total_area_km2", label "Area (km2)".
# - Right panel: "share_pct", label "Share (%)".
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 5
# ------------------------------------------------------------
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# stats_nuts.set_index("class_name")["total_area_km2"].sort_values().plot(
#     kind="barh", ax=axes[0], color="steelblue"
# )
# axes[0].set_xlabel("Area (km2)")
# axes[0].set_title("Land-cover area — LU000 (2024)")
#
# stats_nuts.set_index("class_name")["share_pct"].sort_values().plot(
#     kind="barh", ax=axes[1], color="darkorange"
# )
# axes[1].set_xlabel("Share (%)")
# axes[1].set_title("Land-cover share — LU000 (2024)")
#
# plt.tight_layout()
# plt.show()
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 6 — Build a heatmap of sealed surface density
# ============================================================
#
# Goal: Display the spatial density of sealed surfaces (label 1) across
# the NUTS3 region using a folium.plugins.HeatMap. Each point represents
# the centroid of a sealed polygon, weighted by its area.
#
# Steps:
#   1. Reproject gdf_nuts to EPSG:4326 and compute the map centre
#   2. Filter sealed polygons (label 1) from the reprojected GeoDataFrame
#   3. Compute the centroid of each sealed polygon
#   4. Build heat_data as a list of [lat, lon, weight] where weight is area_km2
#   5. Create a folium.Map and add a HeatMap layer with a red gradient
# ============================================================

gdf_nuts_wgs84 = gdf_nuts.to_crs("__")  # TODO: target CRS for Folium (str)
nuts_center    = gdf_nuts_wgs84.geometry.centroid.union_all().centroid

gdf_sealed = gdf_nuts_wgs84[gdf_nuts_wgs84["label"] == __].copy()  # TODO: sealed label (int)
gdf_sealed["centroid"] = gdf_sealed.geometry.centroid

heat_data = [
    [row.centroid.__, row.centroid.__, row.area_km2]  # TODO: lat/lon attributes
    for _, row in gdf_sealed.iterrows()
]

m = folium.Map(location=[nuts_center.y, nuts_center.x], zoom_start=__)  # TODO: zoom level

HeatMap(
    __,         # TODO: heat_data
    radius=__,  # TODO: radius in pixels (int), e.g. 15
    blur=__,    # TODO: blur in pixels (int), e.g. 20
    max_zoom=13,
    gradient={0.4: "blue", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
).add_to(m)

m

# ------------------------------------------------------------
# HINT — Exercise 6
# ------------------------------------------------------------
# - Reproject with .to_crs("EPSG:4326").
# - Filter with gdf_nuts_wgs84[gdf_nuts_wgs84["label"] == 1].
# - geometry.centroid gives a Point; access its coordinates with .y
#   (latitude) and .x (longitude).
# - heat_data must be a list of [lat, lon, weight] — use area_km2 as
#   the weight so larger sealed polygons contribute more to the heatmap.
# - radius=15 and blur=20 are good starting values.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 6
# ------------------------------------------------------------
# gdf_nuts_wgs84 = gdf_nuts.to_crs("EPSG:4326")
# nuts_center    = gdf_nuts_wgs84.geometry.centroid.union_all().centroid
#
# gdf_sealed = gdf_nuts_wgs84[gdf_nuts_wgs84["label"] == 1].copy()
# gdf_sealed["centroid"] = gdf_sealed.geometry.centroid
#
# heat_data = [
#     [row.centroid.y, row.centroid.x, row.area_km2]
#     for _, row in gdf_sealed.iterrows()
# ]
#
# m = folium.Map(location=[nuts_center.y, nuts_center.x], zoom_start=10)
#
# HeatMap(
#     heat_data,
#     radius=15,
#     blur=20,
#     max_zoom=13,
#     gradient={0.4: "blue", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
# ).add_to(m)
#
# m
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 7 — Compare land-cover on a single tile: 2021 vs 2024
# ============================================================
#
# Goal: Fetch the 2021 prediction for the same tile, compute per-class
# statistics, compare shares with 2024, and display the result as a GT table
# with colour-coded change (pp) and a grouped bar chart.
#
# Steps:
#   1. Call /predict_image with polygons=True for the 2021 tile path
#   2. Build gdf_tile_2021 with area and class name columns
#   3. Define compute_stats() and apply it to both years
#   4. Build tile_comparison with columns "2021 (%)", "2024 (%)",
#      "change (pp)"
#   5. Display with GT — colour change (pp) red-white-green
#   6. Plot a grouped horizontal bar chart
# ============================================================

image_filepath_2021 = (
    "projet-formation/diffusion/funathon/"
    "2026/project3/data/images/LU000/2021/"
    "__"  # TODO: same tile filename as 2024
)

response_2021 = requests.get(
    f"{api_url}/__",  # TODO: endpoint name
    params={"image": __, "polygons": __},  # TODO: filepath, True
)

gdf_tile_2021 = gpd.GeoDataFrame.from_features(
    json.loads(response_2021.json())["features"],
    crs="EPSG:3035",
)

gdf_tile_2021["area_m2"]    = gdf_tile_2021.geometry.area
gdf_tile_2021["area_km2"]   = gdf_tile_2021["area_m2"] / 1e6
gdf_tile_2021["class_name"] = gdf_tile_2021["label"].map(class_names)


def compute_stats(gdf):
    stats = (
        gdf.groupby(["label", "class_name"])
        .agg(
            n_polygons           = ("geometry", "count"),
            total_area_km2       = ("area_km2", "sum"),
            mean_polygon_area_m2 = ("area_m2",  "mean"),
            max_polygon_area_m2  = ("area_m2",  "max"),
        )
        .reset_index()
        .sort_values("total_area_km2", ascending=False)
    )
    total = stats["total_area_km2"].sum()
    stats["share_pct"] = (stats["total_area_km2"] / total * 100).round(2)
    return stats, total


stats_tile_2021, _ = compute_stats(__)  # TODO: 2021 tile GeoDataFrame
stats_tile_2024, _ = compute_stats(__)  # TODO: 2024 tile GeoDataFrame (gdf_tile)

tile_2021 = stats_tile_2021.set_index("class_name")["share_pct"].rename("2021 (%)")
tile_2024 = stats_tile_2024.set_index("class_name")["share_pct"].rename("2024 (%)")

tile_comparison = pd.concat([tile_2021, tile_2024], axis=1).fillna(0)
tile_comparison["change (pp)"] = (tile_comparison["__"] - tile_comparison["__"]).round(2)  # TODO: 2024 - 2021

(
    GT(tile_comparison.reset_index(), rowname_col="class_name")
    .tab_header(title="Land-cover share — single tile, 2021 to 2024")
    .fmt_number(decimals=1)
    .data_color(
        columns=["change (pp)"],
        palette=["red", "white", "green"],
        domain=[-5, 5],
    )
    .data_color(columns=["2021 (%)", "2024 (%)"], palette=["white", "steelblue"])
)

fig, ax = plt.subplots(figsize=(9, 5))
x     = range(len(tile_comparison))
width = 0.35
ax.barh([i + width / 2 for i in x], tile_comparison["2021 (%)"], width, label="2021", color="steelblue")
ax.barh([i - width / 2 for i in x], tile_comparison["2024 (%)"], width, label="2024", color="darkorange")
ax.set_yticks(list(x))
ax.set_yticklabels(tile_comparison.index.tolist())
ax.set_xlabel("Share (%)")
ax.set_title("Land-cover share — single tile, 2021 vs 2024")
ax.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 7
# ------------------------------------------------------------
# - The 2021 tile filename is identical to 2024 — only the year in the path
#   changes: replace 2024 with 2021.
# - compute_stats(gdf) returns (stats_df, total_km2) — use _ to discard
#   the total if not needed.
# - change (pp) = "2024 (%)" - "2021 (%)" — positive means the class
#   gained area share between 2021 and 2024.
# - data_color(palette=["red", "white", "green"], domain=[-5, 5]) colours
#   losses red and gains green automatically.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 7
# ------------------------------------------------------------
# image_filepath_2021 = (
#     "projet-formation/diffusion/funathon/"
#     "2026/project3/data/images/LU000/2021/"
#     "4042000_2951690_0_637.tif"
# )
#
# response_2021 = requests.get(
#     f"{api_url}/predict_image",
#     params={"image": image_filepath_2021, "polygons": True},
# )
#
# gdf_tile_2021 = gpd.GeoDataFrame.from_features(
#     json.loads(response_2021.json())["features"],
#     crs="EPSG:3035",
# )
#
# gdf_tile_2021["area_m2"]    = gdf_tile_2021.geometry.area
# gdf_tile_2021["area_km2"]   = gdf_tile_2021["area_m2"] / 1e6
# gdf_tile_2021["class_name"] = gdf_tile_2021["label"].map(class_names)
#
#
# def compute_stats(gdf):
#     stats = (
#         gdf.groupby(["label", "class_name"])
#         .agg(
#             n_polygons           = ("geometry", "count"),
#             total_area_km2       = ("area_km2", "sum"),
#             mean_polygon_area_m2 = ("area_m2",  "mean"),
#             max_polygon_area_m2  = ("area_m2",  "max"),
#         )
#         .reset_index()
#         .sort_values("total_area_km2", ascending=False)
#     )
#     total = stats["total_area_km2"].sum()
#     stats["share_pct"] = (stats["total_area_km2"] / total * 100).round(2)
#     return stats, total
#
#
# stats_tile_2021, _ = compute_stats(gdf_tile_2021)
# stats_tile_2024, _ = compute_stats(gdf_tile)
#
# tile_2021 = stats_tile_2021.set_index("class_name")["share_pct"].rename("2021 (%)")
# tile_2024 = stats_tile_2024.set_index("class_name")["share_pct"].rename("2024 (%)")
#
# tile_comparison = pd.concat([tile_2021, tile_2024], axis=1).fillna(0)
# tile_comparison["change (pp)"] = (tile_comparison["2024 (%)"] - tile_comparison["2021 (%)"]).round(2)
#
# (
#     GT(tile_comparison.reset_index(), rowname_col="class_name")
#     .tab_header(title="Land-cover share — single tile, 2021 to 2024")
#     .fmt_number(decimals=1)
#     .data_color(
#         columns=["change (pp)"],
#         palette=["red", "white", "green"],
#         domain=[-5, 5],
#     )
#     .data_color(columns=["2021 (%)", "2024 (%)"], palette=["white", "steelblue"])
# )
#
# fig, ax = plt.subplots(figsize=(9, 5))
# x     = range(len(tile_comparison))
# width = 0.35
# ax.barh([i + width / 2 for i in x], tile_comparison["2021 (%)"], width, label="2021", color="steelblue")
# ax.barh([i - width / 2 for i in x], tile_comparison["2024 (%)"], width, label="2024", color="darkorange")
# ax.set_yticks(list(x))
# ax.set_yticklabels(tile_comparison.index.tolist())
# ax.set_xlabel("Share (%)")
# ax.set_title("Land-cover share — single tile, 2021 vs 2024")
# ax.legend()
# plt.tight_layout()
# plt.show()
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 8 — Compare NUTS3 land-cover between 2021 and 2024
# ============================================================
#
# Goal: Scale up the temporal comparison to the full NUTS3 region. Fetch
# 2021 predictions from /predict_nuts, compute statistics, display a GT
# comparison table with colour-coded changes, an absolute area summary
# table, and a two-panel figure.
#
# Steps:
#   1. Call /predict_nuts with nuts_id="LU000" and year=2021
#   2. Build gdf_nuts_2021 with area and class name columns
#   3. Apply compute_stats() to both years
#   4. Build nuts_comparison with "2021 (%)", "2024 (%)", "change (pp)"
#   5. Display with GT — colour change (pp) red-white-green
#   6. Build and display an absolute area summary table for the four groups
#   7. Plot the two-panel figure (grouped bars + change bars green/red)
# ============================================================

response_nuts_2021 = requests.get(
    f"{api_url}/__",  # TODO: endpoint name (str)
    params={"nuts_id": "__", "year": __},  # TODO: "LU000", 2021
)

gdf_nuts_2021 = gpd.GeoDataFrame.from_features(
    json.loads(response_nuts_2021.json()["__"])["features"],  # TODO: response key (str)
    crs="EPSG:3035",
)

gdf_nuts_2021["area_m2"]    = gdf_nuts_2021.geometry.area
gdf_nuts_2021["area_km2"]   = gdf_nuts_2021["area_m2"] / 1e6
gdf_nuts_2021["class_name"] = gdf_nuts_2021["label"].map(class_names)

stats_nuts_2021, _ = compute_stats(__)  # TODO: 2021 NUTS3 GeoDataFrame
stats_nuts_2024, _ = compute_stats(__)  # TODO: 2024 NUTS3 GeoDataFrame (gdf_nuts)

nuts_2021 = stats_nuts_2021.set_index("class_name")["share_pct"].rename("2021 (%)")
nuts_2024 = stats_nuts_2024.set_index("class_name")["share_pct"].rename("2024 (%)")

nuts_comparison = pd.concat([nuts_2021, nuts_2024], axis=1).fillna(0)
nuts_comparison["change (pp)"] = (nuts_comparison["__"] - nuts_comparison["__"]).round(2)  # TODO: 2024 - 2021

(
    GT(nuts_comparison.reset_index(), rowname_col="class_name")
    .tab_header(title="Land-cover share — LU000 NUTS3, 2021 to 2024")
    .fmt_number(decimals=1)
    .data_color(
        columns=["change (pp)"],
        palette=["red", "white", "green"],
        domain=[-5, 5],
    )
    .data_color(columns=["2021 (%)", "2024 (%)"], palette=["white", "steelblue"])
)

rows = []
for label, category in [
    (1,         "Sealed"),
    ([2, 3, 4], "Forest"),
    ([6, 7],    "Agricultural"),
    (10,        "Water"),
]:
    labels_list = [label] if isinstance(label, int) else label
    km2_2021 = stats_nuts_2021.loc[stats_nuts_2021["label"].isin(labels_list), "total_area_km2"].sum()
    km2_2024 = stats_nuts_2024.loc[stats_nuts_2024["label"].isin(labels_list), "total_area_km2"].sum()
    rows.append({"Group": category, "2021 (km2)": round(km2_2021, 2),
                 "2024 (km2)": round(km2_2024, 2), "Delta (km2)": round(km2_2024 - km2_2021, 2)})

area_summary = pd.DataFrame(rows)

(
    GT(area_summary, rowname_col="Group")
    .tab_header(title="Absolute area change — LU000, 2021 to 2024")
    .fmt_number(decimals=2)
    .data_color(
        columns=["Delta (km2)"],
        palette=["red", "white", "green"],
        domain=[-10, 10],
    )
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x      = range(len(nuts_comparison))
width  = 0.35
labels = nuts_comparison.index.tolist()
axes[0].barh([i + width / 2 for i in x], nuts_comparison["2021 (%)"], width, label="2021", color="steelblue")
axes[0].barh([i - width / 2 for i in x], nuts_comparison["2024 (%)"], width, label="2024", color="darkorange")
axes[0].set_yticks(list(x))
axes[0].set_yticklabels(labels)
axes[0].set_xlabel("Share (%)")
axes[0].set_title("Land-cover share — LU000, 2021 vs 2024")
axes[0].legend()
colors = ["green" if v >= 0 else "red" for v in nuts_comparison["change (pp)"]]
axes[1].barh(labels, nuts_comparison["change (pp)"], color=colors)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Change (percentage points)")
axes[1].set_title("Land-cover change 2021 to 2024 — LU000")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 8
# ------------------------------------------------------------
# - The response key for /predict_nuts is "predictions".
# - compute_stats() is already defined in Exercise 7 — reuse it directly.
# - change (pp) = "2024 (%)" - "2021 (%)".
# - Build the absolute area summary as a pd.DataFrame with a loop over
#   the four groups, then pass it to GT().
# - For the change bar chart:
#   colors = ["green" if v >= 0 else "red" for v in nuts_comparison["change (pp)"]]
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 8
# ------------------------------------------------------------
# response_nuts_2021 = requests.get(
#     f"{api_url}/predict_nuts",
#     params={"nuts_id": "LU000", "year": 2021},
# )
#
# gdf_nuts_2021 = gpd.GeoDataFrame.from_features(
#     json.loads(response_nuts_2021.json()["predictions"])["features"],
#     crs="EPSG:3035",
# )
#
# gdf_nuts_2021["area_m2"]    = gdf_nuts_2021.geometry.area
# gdf_nuts_2021["area_km2"]   = gdf_nuts_2021["area_m2"] / 1e6
# gdf_nuts_2021["class_name"] = gdf_nuts_2021["label"].map(class_names)
#
# stats_nuts_2021, _ = compute_stats(gdf_nuts_2021)
# stats_nuts_2024, _ = compute_stats(gdf_nuts)
#
# nuts_2021 = stats_nuts_2021.set_index("class_name")["share_pct"].rename("2021 (%)")
# nuts_2024 = stats_nuts_2024.set_index("class_name")["share_pct"].rename("2024 (%)")
#
# nuts_comparison = pd.concat([nuts_2021, nuts_2024], axis=1).fillna(0)
# nuts_comparison["change (pp)"] = (nuts_comparison["2024 (%)"] - nuts_comparison["2021 (%)"]).round(2)
#
# (
#     GT(nuts_comparison.reset_index(), rowname_col="class_name")
#     .tab_header(title="Land-cover share — LU000 NUTS3, 2021 to 2024")
#     .fmt_number(decimals=1)
#     .data_color(
#         columns=["change (pp)"],
#         palette=["red", "white", "green"],
#         domain=[-5, 5],
#     )
#     .data_color(columns=["2021 (%)", "2024 (%)"], palette=["white", "steelblue"])
# )
#
# rows = []
# for label, category in [
#     (1,         "Sealed"),
#     ([2, 3, 4], "Forest"),
#     ([6, 7],    "Agricultural"),
#     (10,        "Water"),
# ]:
#     labels_list = [label] if isinstance(label, int) else label
#     km2_2021 = stats_nuts_2021.loc[stats_nuts_2021["label"].isin(labels_list), "total_area_km2"].sum()
#     km2_2024 = stats_nuts_2024.loc[stats_nuts_2024["label"].isin(labels_list), "total_area_km2"].sum()
#     rows.append({"Group": category, "2021 (km2)": round(km2_2021, 2),
#                  "2024 (km2)": round(km2_2024, 2), "Delta (km2)": round(km2_2024 - km2_2021, 2)})
#
# area_summary = pd.DataFrame(rows)
#
# (
#     GT(area_summary, rowname_col="Group")
#     .tab_header(title="Absolute area change — LU000, 2021 to 2024")
#     .fmt_number(decimals=2)
#     .data_color(
#         columns=["Delta (km2)"],
#         palette=["red", "white", "green"],
#         domain=[-10, 10],
#     )
# )
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# x      = range(len(nuts_comparison))
# width  = 0.35
# labels = nuts_comparison.index.tolist()
# axes[0].barh([i + width / 2 for i in x], nuts_comparison["2021 (%)"], width, label="2021", color="steelblue")
# axes[0].barh([i - width / 2 for i in x], nuts_comparison["2024 (%)"], width, label="2024", color="darkorange")
# axes[0].set_yticks(list(x))
# axes[0].set_yticklabels(labels)
# axes[0].set_xlabel("Share (%)")
# axes[0].set_title("Land-cover share — LU000, 2021 vs 2024")
# axes[0].legend()
# colors = ["green" if v >= 0 else "red" for v in nuts_comparison["change (pp)"]]
# axes[1].barh(labels, nuts_comparison["change (pp)"], color=colors)
# axes[1].axvline(0, color="black", linewidth=0.8)
# axes[1].set_xlabel("Change (percentage points)")
# axes[1].set_title("Land-cover change 2021 to 2024 — LU000")
# plt.tight_layout()
# plt.show()
# ------------------------------------------------------------
