# %%
# ============================================
# STEP 4 — Statistics
# ============================================
# %%
# ============================================
# Statistics on a single Sentinel-2 image
# ============================================

import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
import pandas as pd

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

# %%
# =========================
# Load the predictions
# =========================

gdf_tile = gpd.read_parquet("predictions_LU000_3034500_2011690_2024.parquet")
gdf_tile["area_m2"] = gdf_tile.geometry.area
gdf_tile["area_km2"] = gdf_tile["area_m2"] / 1e6
gdf_tile["class_name"] = gdf_tile["label"].map(class_names)

# %%
# =========================
# Compute area statistics per class
# =========================

stats_tile = (
    gdf_tile.groupby(["label", "class_name"])
    .agg(
        n_polygons=("geometry", "count"),
        total_area_km2=("area_km2", "sum"),
        mean_polygon_area_m2=("area_m2",  "mean"),
        max_polygon_area_m2=("area_m2",  "max"),
    )
    .reset_index()
    .sort_values("total_area_km2", ascending=False)
)

total_km2 = stats_tile["total_area_km2"].sum()
stats_tile["share_pct"] = (stats_tile["total_area_km2"] / total_km2 * 100).round(2)

print(stats_tile.to_string(index=False))

# %%
# =========================
# Highlight key land-cover categories
# =========================

sealed_km2 = stats_tile.loc[stats_tile["label"] == 1,           "total_area_km2"].sum()
forest_km2 = stats_tile.loc[stats_tile["label"].isin([2, 3, 4]),"total_area_km2"].sum()
agri_km2 = stats_tile.loc[stats_tile["label"].isin([6, 7]),   "total_area_km2"].sum()
water_km2 = stats_tile.loc[stats_tile["label"] == 10,          "total_area_km2"].sum()

print(f"Sealed (built-up)  : {sealed_km2:.2f} km²  ({sealed_km2 / total_km2 * 100:.1f} %)")
print(f"Forest             : {forest_km2:.2f} km²  ({forest_km2 / total_km2 * 100:.1f} %)")
print(f"Agricultural       : {agri_km2:.2f}  km²  ({agri_km2  / total_km2 * 100:.1f} %)")
print(f"Water              : {water_km2:.2f} km²  ({water_km2 / total_km2 * 100:.1f} %)")

# %%
# =========================
# Visualise the distribution
# =========================

fig, ax = plt.subplots(figsize=(8, 4))

stats_tile.set_index("class_name")["total_area_km2"].sort_values().plot(
    kind="barh", ax=ax, color="steelblue"
)

ax.set_xlabel("Area (km²)")
ax.set_title("Land-cover distribution — single tile (LU000, 2024)")
plt.tight_layout()
plt.show()


# %%
# ============================================
# Statistics on a NUTS3 / year pair
# ============================================

# %%
# =========================
# Load the predictions
# =========================

gdf_nuts = gpd.read_parquet("predictions_LU000_2024.parquet")
gdf_nuts["area_m2"] = gdf_nuts.geometry.area
gdf_nuts["area_km2"] = gdf_nuts["area_m2"] / 1e6
gdf_nuts["class_name"] = gdf_nuts["label"].map(class_names)

# %%
# =========================
# Compute area statistics per class
# =========================

stats_nuts = (
    gdf_nuts.groupby(["label", "class_name"])
    .agg(
        n_polygons=("geometry", "count"),
        total_area_km2=("area_km2", "sum"),
        mean_polygon_area_m2=("area_m2",  "mean"),
        max_polygon_area_m2=("area_m2",  "max"),
    )
    .reset_index()
    .sort_values("total_area_km2", ascending=False)
)

total_nuts_km2 = stats_nuts["total_area_km2"].sum()
stats_nuts["share_pct"] = (stats_nuts["total_area_km2"] / total_nuts_km2 * 100).round(2)

print(stats_nuts.to_string(index=False))

# %%
# =========================
# Highlight key land-cover categories
# =========================

sealed_km2 = stats_nuts.loc[stats_nuts["label"] == 1,           "total_area_km2"].sum()
forest_km2 = stats_nuts.loc[stats_nuts["label"].isin([2, 3, 4]),"total_area_km2"].sum()
agri_km2 = stats_nuts.loc[stats_nuts["label"].isin([6, 7]),   "total_area_km2"].sum()
water_km2 = stats_nuts.loc[stats_nuts["label"] == 10,          "total_area_km2"].sum()

print(f"Sealed (built-up)  : {sealed_km2:.2f} km²  ({sealed_km2 / total_nuts_km2 * 100:.1f} %)")
print(f"Forest             : {forest_km2:.2f} km²  ({forest_km2 / total_nuts_km2 * 100:.1f} %)")
print(f"Agricultural       : {agri_km2:.2f}  km²  ({agri_km2   / total_nuts_km2 * 100:.1f} %)")
print(f"Water              : {water_km2:.2f} km²  ({water_km2  / total_nuts_km2 * 100:.1f} %)")

# %%
# =========================
# Compare tile vs. NUTS3
# =========================

tile_shares = stats_tile.set_index("class_name")["share_pct"].rename("tile_share_pct")
nuts_shares = stats_nuts.set_index("class_name")["share_pct"].rename("nuts3_share_pct")

comparison = pd.concat([tile_shares, nuts_shares], axis=1).fillna(0)
print(comparison.sort_values("nuts3_share_pct", ascending=False).to_string())

# %%
# =========================
# Visualise the NUTS3 distribution
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stats_nuts.set_index("class_name")["total_area_km2"].sort_values().plot(
    kind="barh", ax=axes[0], color="steelblue"
)
axes[0].set_xlabel("Area (km²)")
axes[0].set_title("Land-cover area — LU000 (2024)")

stats_nuts.set_index("class_name")["share_pct"].sort_values().plot(
    kind="barh", ax=axes[1], color="darkorange"
)
axes[1].set_xlabel("Share (%)")
axes[1].set_title("Land-cover share — LU000 (2024)")

plt.tight_layout()
plt.show()

# %%
# =========================
# Interactive map — sealed area per polygon
# =========================

# Reproject to WGS84 for folium
gdf_nuts_wgs84 = gdf_nuts[gdf_nuts["label"] == 1].to_crs("EPSG:4326")

# Centroid of each polygon
gdf_nuts_wgs84["centroid"] = gdf_nuts_wgs84.geometry.centroid
gdf_nuts_wgs84["lat"] = gdf_nuts_wgs84["centroid"].y
gdf_nuts_wgs84["lon"] = gdf_nuts_wgs84["centroid"].x

# Colormap scaled to polygon area
colormap = cm.LinearColormap(
    colors=["#ffffcc", "#fd8d3c", "#800026"],
    vmin=gdf_nuts_wgs84["area_m2"].min(),
    vmax=gdf_nuts_wgs84["area_m2"].max(),
    caption="Sealed area (m²)",
)

# Map centred on the NUTS3 region
center = [gdf_nuts_wgs84["lat"].mean(), gdf_nuts_wgs84["lon"].mean()]
m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

for _, row in gdf_nuts_wgs84.iterrows():
    # Radius proportional to area (m²), capped at 30
    radius = min(3 + row["area_m2"] / 500, 30)
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=colormap(row["area_m2"]),
        fill=True,
        fill_color=colormap(row["area_m2"]),
        fill_opacity=0.7,
        tooltip=f"Area: {row['area_m2']:.0f} m²",
    ).add_to(m)

colormap.add_to(m)
m.save("map_sealed_LU000_2024.html")
