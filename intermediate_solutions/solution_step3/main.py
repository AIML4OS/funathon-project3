# %%
# ============================================
# STEP 3 — Inference
# ============================================

# %%
# ============================================
# Imports
# ============================================
import json
import os

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import requests
from dotenv import load_dotenv
from folium.raster_layers import ImageOverlay
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import transform_bounds

from utils import (
    create_geojson_from_mask,
    get_satellite_image,
    predict,
)

load_dotenv()


# %%
# ============================================================
# EXERCISE 1 — Load a model from MLflow
# ============================================================
#
# Goal: Load a trained segmentation model from the MLflow
# model registry and retrieve its metadata (n_bands,
# tiles_size, augment_size, normalization parameters).
#
# Steps:
#   1. Set model_name and model_version
#   2. Load the model from the registry
#   3. Read the run parameters from mlflow.get_run()
#   4. Print the metadata to verify
# ============================================================

model_name = "__"       # TODO: name of the model registered in MLflow (str)
model_version = "__"    # TODO: version of the model to load (str)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(mlflow_tracking_uri)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

run = mlflow.get_run(model.metadata.run_id)

n_bands      = "__"  # TODO: read "n_bands" from run.data.params and cast to int
tiles_size   = "__"  # TODO: read "tiles_size" from run.data.params and cast to int
augment_size = "__"  # TODO: read "augment_size" from run.data.params and cast to int
module_name  = "__"  # TODO: read "module_name" from run.data.params (str)

normalization_mean = json.loads(
    "__"  # TODO: read "normalization_mean" from run.data.params
)[:n_bands]

normalization_std = [
    float(v) for v in eval(
        "__"  # TODO: read "normalization_std" from run.data.params
    )
][:n_bands]

print(f"n_bands={n_bands}, tiles_size={tiles_size}, augment_size={augment_size}")
print(f"mean={normalization_mean}")
print(f"std={normalization_std}")

# ------------------------------------------------------------
# HINT — Exercise 1
# ------------------------------------------------------------
# - model_name and model_version are strings — check the MLflow
#   UI to find the right values.
# - All parameters are stored in run.data.params as strings;
#   cast them to int where needed (e.g. int(run.data.params["n_bands"])).
# - MLFLOW_TRACKING_URI is loaded from the .env file via
#   load_dotenv() + os.getenv().
# - Print run.data.params to explore all available keys.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 1
# ------------------------------------------------------------
# model_name    = "segmentation-sentinel2-model"
# model_version = "2"
# mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
#
# mlflow.set_tracking_uri(mlflow_tracking_uri)
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
#
# run = mlflow.get_run(model.metadata.run_id)
#
# n_bands      = int(run.data.params["n_bands"])
# tiles_size   = int(run.data.params["tiles_size"])
# augment_size = int(run.data.params["augment_size"])
# module_name  = run.data.params["module_name"]
#
# normalization_mean = json.loads(
#     run.data.params["normalization_mean"]
# )[:n_bands]
#
# normalization_std = [
#     float(v) for v in eval(
#         run.data.params["normalization_std"]
#     )
# ][:n_bands]
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 — Run inference on a single Sentinel-2 image
# ============================================================
#
# Goal: Load a Sentinel-2 image from MinIO and run the
# segmentation model on it to produce a labelled mask.
#
# The image is publicly available at:
#   https://minio.lab.sspcloud.fr/projet-formation/
#   diffusion/funathon/2026/project3/data/images/
#   {NUTS}/{year}/{filename}.tif
#
# Steps:
#   1. Build the full image URL from image_target
#   2. Call predict() to run the model on the image
#   3. Print the mask shape and the set of predicted classes
# ============================================================

image_target = "LU000/2024/4022000_2979190_0_354.tif"

image_path = (
    "https://minio.lab.sspcloud.fr/projet-formation/"
    "diffusion/funathon/2026/project3/data/images/"
    + "__"  # TODO: relative path to the image (str), use image_target
)

satellite_img, predictions = predict(
    images="__",             # TODO: full image URL
    model="__",              # TODO: model loaded in Exercise 1
    tiles_size="__",         # TODO: tile size from model metadata
    augment_size="__",       # TODO: augmentation size from model metadata
    n_bands="__",            # TODO: number of bands
    normalization_mean="__", # TODO: normalisation mean
    normalization_std="__",  # TODO: normalisation std
    module_name="__",        # TODO: module name
)

print(f"Mask shape    : {predictions.shape}")
print(f"Classes found : {set(predictions.flatten().tolist())}")

# ------------------------------------------------------------
# HINT — Exercise 2
# ------------------------------------------------------------
# - Concatenate the base URL with image_target to get image_path.
# - predict() returns a tuple (satellite_img dict, predictions array).
# - All metadata variables (tiles_size, augment_size, etc.)
#   were retrieved in Exercise 1.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2
# ------------------------------------------------------------
# image_target = "LU000/2024/4022000_2979190_0_354.tif"
#
# image_path = (
#     "https://minio.lab.sspcloud.fr/projet-formation/"
#     "diffusion/funathon/2026/project3/data/images/"
#     + image_target
# )
#
# satellite_img, predictions = predict(
#     images=image_path,
#     model=model,
#     tiles_size=tiles_size,
#     augment_size=augment_size,
#     n_bands=n_bands,
#     normalization_mean=normalization_mean,
#     normalization_std=normalization_std,
#     module_name=module_name,
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 3 — Display the prediction
# ============================================================
#
# Goal: Build an RGB composite from the satellite image bands
# and display it side by side with the predicted land cover
# mask, using a shared legend for the 10 CLC+ classes.
#
# Steps:
#   1. Extract bands 4, 3, 2 (indices 3, 2, 1) and transpose
#      to (H, W, 3), then normalise with the 98th percentile
#   2. Create a figure with 2 subplots (RGB / predicted mask)
#   3. Display rgb on axes[0] and the label on axes[1] with cmap
#   4. Add a shared legend
# ============================================================

classes = [
    ("Sealed (1)",                        "#FF0100"),
    ("Woody – needle leaved trees (2)",   "#238B23"),
    ("Woody – broadleaved deciduous (3)", "#80FF00"),
    ("Woody – broadleaved evergreen (4)", "#00FF00"),
    ("Low-growing woody plants (5)",      "#804000"),
    ("Permanent herbaceous (6)",          "#CCF24E"),
    ("Periodically herbaceous (7)",       "#FEFF80"),
    ("Lichens and mosses (8)",            "#FF81FF"),
    ("Non- and sparsely-vegetated (9)",   "#BFBFBF"),
    ("Water (10)",                        "#0080FF"),
]

cmap           = ListedColormap([color for _, color in classes])
label_to_color = {i + 1: color for i, (_, color) in enumerate(classes)}
legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]

# RGB composite — bands 4, 3, 2 → indices 3, 2, 1
rgb = np.transpose(
    satellite_img["array"][[__, __, __]],  # TODO: band indices for R, G, B
    (1, 2, 0)
).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(__)                           # TODO: display the RGB composite
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(__, cmap=__, vmin=1, vmax=10) # TODO: predictions array, colormap
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=True,
)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 3
# ------------------------------------------------------------
# - satellite_img["array"] has shape (n_bands, H, W); bands are
#   0-indexed so B4=3, B3=2, B2=1.
# - np.transpose(..., (1, 2, 0)) reshapes from (3, H, W) to (H, W, 3).
# - Normalise: divide by np.percentile(rgb, 98) then np.clip(..., 0, 1).
# - Use vmin=1, vmax=10 on imshow so the colormap aligns with
#   the 10 CLC+ classes.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 3
# ------------------------------------------------------------
# rgb = np.transpose(
#     satellite_img["array"][[3, 2, 1]], (1, 2, 0)
# ).astype(np.float32)
# p98 = np.percentile(rgb, 98)
# rgb = np.clip(rgb / p98, 0, 1)
#
# axes[0].imshow(rgb)
# axes[1].imshow(predictions, cmap=cmap, vmin=1, vmax=10)
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 4 — Vectorise the mask and display the polygons
# ============================================================
#
# Goal: Convert the predicted mask into a GeoDataFrame of
# polygons (one polygon per connected region of the same class),
# display all three outputs side by side, and save the result.
#
# Steps:
#   1. Call create_geojson_from_mask() with satellite_img and predictions
#   2. Create a 3-subplot figure: RGB / predicted mask / polygons
#   3. Use gdf_pred.plot() with column="label" and the CLC+ colormap
#   4. Fix the axis limits of the polygon subplot with total_bounds
# ============================================================

gdf_pred = create_geojson_from_mask(__, __)  # TODO: satellite_img and predictions

print(f"{len(gdf_pred)} polygons extracted")
print(gdf_pred.head())

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(predictions, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

gdf_pred.plot(
    column=__,  # TODO: column to use for colouring (str)
    cmap=__,    # TODO: colormap
    vmin=1,
    vmax=10,
    ax=axes[2],
    legend=False,
)
axes[2].set_title("Predicted polygons")
axes[2].set_aspect("equal")
xmin, ymin, xmax, ymax = gdf_pred.total_bounds
axes[2].set_xlim(xmin, xmax)
axes[2].set_ylim(ymin, ymax)
axes[2].axis("off")

fig.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 4
# ------------------------------------------------------------
# - create_geojson_from_mask(satellite_img, predictions) returns
#   a GeoDataFrame with columns "geometry" and "label".
# - Use column="label" in gdf_pred.plot() to colour by class.
# - gdf_pred.total_bounds returns (xmin, ymin, xmax, ymax) —
#   use it to restore axis limits after geopandas resets them.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 4
# ------------------------------------------------------------
# gdf_pred = create_geojson_from_mask(satellite_img, predictions)
#
# gdf_pred.plot(
#     column="label",
#     cmap=cmap,
#     vmin=1,
#     vmax=10,
#     ax=axes[2],
#     legend=False,
# )
# ------------------------------------------------------------


# %%
# ============================================
# Part 2 — Inference via API
# ============================================
# Note: classes, cmap, legend_elements and label_to_color
# are already defined above and reused in this section.

api_url = "https://funathon-2026-project3.lab.sspcloud.fr"


# %%
# ============================================================
# EXERCISE 1 — Find a satellite image from a GPS point
# ============================================================
#
# Goal: Given a GPS point (latitude, longitude), use the API
# endpoint /find_image to retrieve the filename of the
# Sentinel-2 tile that contains this point.
#
# API endpoint : GET /find_image
# Parameters   :
#   - gps_point (List[float, float]) : [latitude, longitude] in WGS84
#   - nuts_id (str) : NUTS3 region identifier
#   - year    (int) : year of the satellite images (2018–2024)
#
# Steps:
#   1. Choose a GPS point in Luxembourg (e.g. Eurostat offices)
#   2. Call the /find_image endpoint with the correct parameters
#   3. Print the filename returned by the API
# ============================================================

gps_point = __    # TODO: [latitude, longitude] in WGS84 (List[float]), e.g. [49.63, 6.16] for Eurostat
nuts_id   = "__"  # TODO: NUTS3 identifier (str), e.g. "LU000" for Luxembourg
year      = __    # TODO: year of the satellite images (int, between 2018 and 2024)

response_find = requests.get(
    f"{api_url}/__",  # TODO: endpoint name (str), e.g. "find_image"
    params={
        "gps_point": __,  # TODO: [latitude, longitude] defined above
        "nuts_id":   __,  # TODO: NUTS3 identifier defined above
        "year":      __,  # TODO: year defined above
    },
)
response_find.raise_for_status()

image_filename = response_find.json()
print(f"Image found: {image_filename}")

# ------------------------------------------------------------
# HINT — Exercise 1
# ------------------------------------------------------------
# - The Eurostat offices are at latitude=49.63, longitude=6.16.
# - The NUTS3 identifier for Luxembourg is "LU000".
# - The endpoint name is "find_image".
# - requests automatically repeats the parameter for each list
#   value: [49.63, 6.16] → ?gps_point=49.63&gps_point=6.16
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 1
# ------------------------------------------------------------
# gps_point = [49.63339525016761, 6.1689982433356025]
# nuts_id   = "LU000"
# year      = 2024
#
# response_find = requests.get(
#     f"{api_url}/find_image",
#     params={
#         "gps_point": gps_point,
#         "nuts_id":   nuts_id,
#         "year":      year,
#     },
# )
# response_find.raise_for_status()
#
# image_filename = response_find.json()
# print(f"Image found: {image_filename}")
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 — Predict land cover for the found image
# ============================================================
#
# Goal: Using the filename returned by /find_image, call the
# /predict_image endpoint to retrieve the predicted polygons,
# then visualise them on a static plot and an interactive map.
#
# API endpoint : GET /predict_image
# Parameters   :
#   - image    (str)  : S3 image path as returned by /find_image
#   - polygons (bool) : if True, also returns vectorised polygons
#
# The response is a GeoJSON string containing the predicted polygons.
#
# Steps:
#   1. Build the S3 image path (base_url + image_filename)
#   2. Call /predict_image with polygons=True
#   3. Parse the response as a GeoDataFrame
#   4. Load the RGB composite with get_satellite_image()
#   5. Display RGB and polygons side by side
# ============================================================

s3_base_url = (
    "projet-formation/diffusion/funathon/"
    "2026/project3/data/images/__/__/"  # TODO: fill in nuts_id and year
)

image_filepath = s3_base_url + image_filename

response_pred = requests.get(
    f"{api_url}/__",  # TODO: endpoint name (str), e.g. "predict_image"
    params={
        "image":    __,  # TODO: S3 path built above
        "polygons": __,  # TODO: set to True to receive polygons
    },
)
response_pred.raise_for_status()

gdf_pred = gpd.GeoDataFrame.from_features(
    json.loads(response_pred.json())["features"],  # parse GeoJSON string → dict → features
    crs="EPSG:3035",
)

print(f"{len(gdf_pred)} polygons extracted")
print(gdf_pred.head())

# ------------------------------------------------------------
# HINT — Exercise 2
# ------------------------------------------------------------
# - The S3 path follows the pattern:
#   "projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_id}/{year}/"
# - The endpoint name is "predict_image".
# - Set polygons=True to include GeoJSON polygons in the response.
# - response_pred.json() returns a GeoJSON string;
#   json.loads() parses it into a dict before calling from_features().
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2
# ------------------------------------------------------------
# s3_base_url = (
#     "projet-formation/diffusion/funathon/"
#     "2026/project3/data/images/LU000/2024/"
# )
#
# image_filepath = s3_base_url + image_filename
#
# response_pred = requests.get(
#     f"{api_url}/predict_image",
#     params={
#         "image":    image_filepath,
#         "polygons": True,
#     },
# )
# response_pred.raise_for_status()
#
# gdf_pred = gpd.GeoDataFrame.from_features(
#     json.loads(response_pred.json())["features"],
#     crs="EPSG:3035",
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 (continued) — Visualise the prediction
# ============================================================
#
# Goal: Load the RGB composite of the image and display it
# side by side with the predicted polygons.
#
# Steps:
#   1. Build the full HTTPS URL (minio_url + image_filepath)
#   2. Load the satellite image with get_satellite_image()
#   3. Build the RGB composite (bands 4, 3, 2 → indices 3, 2, 1)
#   4. Display the 2 subplots: RGB / polygons
# ============================================================

N_BANDS   = 14
minio_url = "https://minio.lab.sspcloud.fr/"

image_url = minio_url + __  # TODO: S3 image path built above (image_filepath)

si = get_satellite_image(__, n_bands=__)  # TODO: full HTTPS URL, number of bands

rgb = np.transpose(si["array"][[3, 2, 1]], (1, 2, 0)).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

gdf_pred.plot(column="label", cmap=cmap, vmin=1, vmax=10, ax=axes[1], legend=False)
axes[1].set_title("Predicted polygons")
axes[1].set_aspect("equal")
xmin, ymin, xmax, ymax = gdf_pred.total_bounds
axes[1].set_xlim(xmin, xmax)
axes[1].set_ylim(ymin, ymax)
axes[1].axis("off")

fig.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 2 (continued)
# ------------------------------------------------------------
# - Concatenate minio_url and image_filepath to get the full URL.
# - get_satellite_image(url, n_bands=14) returns a dict with
#   key "array" of shape (n_bands, H, W).
# - Bands 4, 3, 2 are at 0-based indices 3, 2, 1.
# - Normalise: divide by np.percentile(rgb, 98) then clip to [0, 1].
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2 (continued)
# ------------------------------------------------------------
# image_url = minio_url + image_filepath
# si = get_satellite_image(image_url, n_bands=N_BANDS)
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 (continued) — Interactive Folium map
# ============================================================
#
# Goal: Display the RGB image overlay and the predicted
# polygons on an interactive Folium map.
#
# Steps:
#   1. Reproject the image bounds to EPSG:4326 with transform_bounds()
#   2. Create a folium.Map centred on the tile
#   3. Add an ImageOverlay with the RGB array
#   4. Reproject gdf_pred to EPSG:4326 and add a GeoJson layer
#      coloured by label using label_to_color
# ============================================================

west, south, east, north = transform_bounds(si["crs"], "EPSG:4326", *si["bounds"])
center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

ImageOverlay(
    image=rgb,
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(m)

gdf_pred_wgs84 = gdf_pred.to_crs("EPSG:4326")

folium.GeoJson(
    gdf_pred_wgs84,
    style_function=lambda feature: {
        "fillColor": __,  # TODO: use label_to_color to colour by label
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(m)

m

# ------------------------------------------------------------
# HINT — Exercise 2 (Folium)
# ------------------------------------------------------------
# - feature["properties"]["label"] gives the integer class ID.
# - label_to_color is a dict {class_id: hex_color}.
# - Use label_to_color.get(..., "#808080") for a safe fallback.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2 (Folium)
# ------------------------------------------------------------
# folium.GeoJson(
#     gdf_pred_wgs84,
#     style_function=lambda feature: {
#         "fillColor": label_to_color.get(feature["properties"]["label"], "#808080"),
#         "color": "black",
#         "weight": 0.5,
#         "fillOpacity": 0.6,
#     },
#     tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
# ).add_to(m)
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 3 — Predict land cover for an entire NUTS3 region
# ============================================================
#
# Goal: Use the /predict_nuts endpoint to retrieve predictions
# for all Sentinel-2 tiles covering a given NUTS3 region and
# year. The server handles caching, so repeated calls are fast.
#
# API endpoint : GET /predict_nuts
# Parameters   :
#   - nuts_id (str) : NUTS3 region identifier
#   - year    (int) : year of the satellite images
#
# The response is a GeoJSON string containing all predicted polygons.
#
# Steps:
#   1. Call /predict_nuts for nuts_id="LU000" and year=2024
#   2. Parse the response as a GeoDataFrame
#   3. Print the number of polygons and the first rows
#   4. Save the GeoDataFrame to a Parquet file
# ============================================================

response_nuts = requests.get(
    f"{api_url}/__",  # TODO: endpoint name (str), e.g. "predict_nuts"
    params={
        "nuts_id": "__",  # TODO: NUTS3 identifier (str), e.g. "LU000"
        "year":    __,    # TODO: year (int, between 2018 and 2024)
    },
)
response_nuts.raise_for_status()

gdf_nuts = gpd.GeoDataFrame.from_features(
    json.loads(response_nuts.json())["features"],  # parse GeoJSON string → dict → features
    crs="EPSG:3035",
)

print(f"{len(gdf_nuts)} polygons received")
print(gdf_nuts.head())

# ------------------------------------------------------------
# HINT — Exercise 3
# ------------------------------------------------------------
# - The endpoint name is "predict_nuts".
# - response_nuts.json() returns a GeoJSON string;
#   json.loads() parses it into a dict before calling from_features().
# - The first call for a new NUTS3/year combination may take
#   several minutes; subsequent calls are fast thanks to the S3 cache.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 3
# ------------------------------------------------------------
# response_nuts = requests.get(
#     f"{api_url}/predict_nuts",
#     params={"nuts_id": "LU000", "year": 2024},
# )
# response_nuts.raise_for_status()
#
# gdf_nuts = gpd.GeoDataFrame.from_features(
#     json.loads(response_nuts.json())["features"],
#     crs="EPSG:3035",
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 3 (continued) — Visualise NUTS3 predictions
# ============================================================
#
# Goal: Display all predicted polygons for the NUTS3 region
# on an interactive Folium map, then save the GeoDataFrame.
#
# Steps:
#   1. Reproject gdf_nuts to EPSG:4326
#   2. Compute the centroid of the region to centre the map
#   3. Create a folium.Map and add a GeoJson layer
#   4. Save gdf_nuts to a Parquet file
# ============================================================

gdf_nuts_wgs84 = gdf_nuts.to_crs("EPSG:4326")
nuts_center = gdf_nuts_wgs84.geometry.centroid.union_all().centroid

m_nuts = folium.Map(location=[nuts_center.y, nuts_center.x], zoom_start=10)

folium.GeoJson(
    __,  # TODO: reprojected NUTS3 GeoDataFrame (gdf_nuts_wgs84)
    style_function=lambda feature: {
        "fillColor": __,  # TODO: use label_to_color to colour by label
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(m_nuts)

m_nuts

# ------------------------------------------------------------
# HINT — Exercise 3 (continued)
# ------------------------------------------------------------
# - Pass gdf_nuts_wgs84 (already reprojected to EPSG:4326) to folium.GeoJson().
# - Same style_function pattern as Exercise 2:
#   label_to_color.get(feature["properties"]["label"], "#808080")
# - Choose a meaningful Parquet filename.
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 3 (continued)
# ------------------------------------------------------------
# folium.GeoJson(
#     gdf_nuts_wgs84,
#     style_function=lambda feature: {
#         "fillColor": label_to_color.get(feature["properties"]["label"], "#808080"),
#         "color": "black",
#         "weight": 0.3,
#         "fillOpacity": 0.6,
#     },
#     tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
# ).add_to(m_nuts)
#
# m_nuts
# ------------------------------------------------------------
