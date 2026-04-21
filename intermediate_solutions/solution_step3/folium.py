# %%
# ============================================================
# EXERCISE 5 — Display predictions on an interactive Folium map
# ============================================================
#
# Goal: Display the RGB image overlay and the predicted
# polygons on an interactive Folium map.
#
# Steps:
#   1. Reproject the image bounds to EPSG:4326 with
#      transform_bounds()
#   2. Create a folium.Map centred on the tile
#   3. Add an ImageOverlay with the RGB array
#   4. Reproject gdf_pred to EPSG:4326 and add a GeoJson layer
#      coloured by label using label_to_color
# ============================================================

west, south, east, north = transform_bounds(
    satellite_img.crs,
    "EPSG:4326",
    *satellite_img.bounds,
)
center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

ImageOverlay(
    image="__",                              # TODO: composite RGB normalisé
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(m)

gdf_pred_wgs84 = gdf_pred.to_crs("__")  # TODO: code EPSG cible pour Folium (str)

folium.GeoJson(
    gdf_pred_wgs84,
    style_function=lambda feature: {
        "fillColor": "__",  # TODO: utiliser label_to_color pour colorier selon le label
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(m)

m

# ------------------------------------------------------------
# HINT — Exercise 5
# ------------------------------------------------------------
# - transform_bounds(src_crs, "EPSG:4326", *bounds) returns
#   (west, south, east, north) in degrees
# - Folium requires coordinates in EPSG:4326 (WGS84)
# - Pass rgb (the normalized float32 array) to ImageOverlay
# - feature["properties"]["label"] gives the class ID;
#   use label_to_color.get(..., "#808080") for a safe fallback
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 5
# ------------------------------------------------------------
# ImageOverlay(image=rgb, ...)
#
# gdf_pred_wgs84 = gdf_pred.to_crs("EPSG:4326")
#
# style_function=lambda feature: {
#     "fillColor": label_to_color.get(
#         feature["properties"]["label"], "#808080"
#     ),
#     ...
# }
# ------------------------------------------------------------