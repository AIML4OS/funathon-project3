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
import torch
from dotenv import load_dotenv
from folium.raster_layers import ImageOverlay
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import transform_bounds

from utils import (
    create_geojson_from_mask,
    get_model,
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
#   2. Call get_model() to load the model from the registry
#   3. Read the run parameters from mlflow.get_run()
#   4. Print the metadata to verify
# ============================================================

model_name = "__"  # TODO: nom du modèle enregistré dans MLflow (str)
model_version = "__"  # TODO: version du modèle souhaitée (str)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

model = get_model(__, __, __)  # TODO: model_name, model_version, mlflow_tracking_uri

run = mlflow.get_run(model.metadata.run_id)

n_bands      = int(run.data.params["n_bands"])
tiles_size   = int(run.data.params["tiles_size"])
augment_size = int(run.data.params["augment_size"])
module_name  = run.data.params["module_name"]

normalization_mean = json.loads(
    mlflow.get_run(model.metadata.run_id).data.params["normalization_mean"]
)[:n_bands]

normalization_std = [
    float(v) for v in eval(
        mlflow.get_run(model.metadata.run_id).data.params["normalization_std"]
    )
][:n_bands]

print(f"n_bands={n_bands}, tiles_size={tiles_size}, augment_size={augment_size}")
print(f"mean={normalization_mean}")
print(f"std={normalization_std}")

# ------------------------------------------------------------
# HINT — Exercise 1
# ------------------------------------------------------------
# - model_name and model_version are strings — check with your
#   team or the MLflow UI which model and version to use
# - get_model(model_name, model_version, mlflow_tracking_uri)
#   returns an mlflow.pyfunc.PyFuncModel object
# - All parameters are stored in run.data.params as strings;
#   cast them to int/float as needed
# - MLFLOW_TRACKING_URI is loaded from the .env file via
#   load_dotenv() + os.getenv()
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 1
# ------------------------------------------------------------
# model_name = "segmentation-sentinel2-model"
# model_version = "2"
# mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
#
# model = get_model(model_name, model_version, mlflow_tracking_uri)
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
#   3. Convert logits to a class mask with torch.argmax()
#   4. Print the mask shape and the set of predicted classes
# ============================================================

image_target = "LU000/2024/4022000_2979190_0_354.tif"

image_path = (
    "https://minio.lab.sspcloud.fr/projet-formation/"
    "diffusion/funathon/2026/project3/data/images/"
    + __  # TODO: chemin relatif de l'image (str), utiliser image_target
)

labeled_satellite_img = predict(
    images=__,            # TODO: URL complète de l'image
    model=__,             # TODO: modèle chargé à l'exercice 1
    tiles_size=__,        # TODO: taille des tuiles récupérée depuis les métadonnées
    augment_size=__,      # TODO: taille d'augmentation récupérée depuis les métadonnées
    n_bands=__,           # TODO: nombre de bandes
    normalization_mean=__, # TODO: moyenne de normalisation
    normalization_std=__,  # TODO: écart-type de normalisation
    module_name=__,        # TODO: nom du module
)

labeled_satellite_img.label = torch.from_numpy(labeled_satellite_img.label)
labeled_satellite_img.label = torch.argmax(labeled_satellite_img.label, dim=0).numpy()

print(f"Mask shape    : {labeled_satellite_img.label.shape}")
print(f"Classes found : {set(labeled_satellite_img.label.flatten().tolist())}")

# ------------------------------------------------------------
# HINT — Exercise 2
# ------------------------------------------------------------
# - Concatenate the base URL with image_target to get image_path
# - predict() returns a SegmentationLabeledSatelliteImage with
#   a .label attribute containing raw logits (shape: n_classes, H, W)
# - torch.argmax(tensor, dim=0) picks the class with highest
#   score for each pixel → shape becomes (H, W)
# - All metadata variables (tiles_size, augment_size, etc.)
#   were retrieved in Exercise 1
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2
# ------------------------------------------------------------
# image_path = (
#     "https://minio.lab.sspcloud.fr/projet-formation/"
#     "diffusion/funathon/2026/project3/data/images/"
#     + image_target
# )
#
# labeled_satellite_img = predict(
#     images=image_path,
#     model=model,
#     tiles_size=tiles_size,
#     augment_size=augment_size,
#     n_bands=n_bands,
#     normalization_mean=normalization_mean,
#     normalization_std=normalization_std,
#     module_name=module_name,
# )
#
# labeled_satellite_img.label = torch.from_numpy(labeled_satellite_img.label)
# labeled_satellite_img.label = torch.argmax(labeled_satellite_img.label, dim=0).numpy()
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
#      to (H, W, 3), then normalize with the 98th percentile
#   2. Create a figure with 2 subplots (RGB / predicted mask)
#   3. Display rgb on axes[0] and the label on axes[1] with cmap
#   4. Add a legend and save the figure
# ============================================================

classes = [
    ("Sealed (1)",                       "#FF0100"),
    ("Woody – needle leaved trees (2)",  "#238B23"),
    ("Woody – broadleaved deciduous (3)","#80FF00"),
    ("Woody – broadleaved evergreen (4)","#00FF00"),
    ("Low-growing woody plants (5)",     "#804000"),
    ("Permanent herbaceous (6)",         "#CCF24E"),
    ("Periodically herbaceous (7)",      "#FEFF80"),
    ("Lichens and mosses (8)",           "#FF81FF"),
    ("Non- and sparsely-vegetated (9)",  "#BFBFBF"),
    ("Water (10)",                       "#0080FF"),
]

cmap = ListedColormap([color for _, color in classes])

label_to_color = {i + 1: color for i, (_, color) in enumerate(classes)}

legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]

# RGB composite — bands 4, 3, 2 → indices 3, 2, 1
rgb = np.transpose(
    labeled_satellite_img.satellite_image.array[[__, __, __]],  # TODO: indices des bandes R, G, B
    (1, 2, 0)
).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(__)           # TODO: afficher le composite RGB
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(__, cmap=__, vmin=1, vmax=10)  # TODO: label, cmap
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=True,
)

plt.tight_layout()
fig.savefig("prediction.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 3
# ------------------------------------------------------------
# - labeled_satellite_img.satellite_image.array has shape
#   (n_bands, H, W); bands are 0-indexed so B4=3, B3=2, B2=1
# - np.transpose(..., (1, 2, 0)) reshapes to (H, W, 3)
# - Normalize: divide by np.percentile(rgb, 98) then
#   np.clip(..., 0, 1) to keep values in [0, 1]
# - Use vmin=1, vmax=10 on imshow for the label so the
#   colormap aligns with the 10 CLC+ classes
# - bbox_inches="tight" ensures the legend is not cropped
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 3
# ------------------------------------------------------------
# rgb = np.transpose(
#     labeled_satellite_img.satellite_image.array[[3, 2, 1]], (1, 2, 0)
# ).astype(np.float32)
# p98 = np.percentile(rgb, 98)
# rgb = np.clip(rgb / p98, 0, 1)
#
# axes[0].imshow(rgb)
# axes[1].imshow(labeled_satellite_img.label, cmap=cmap, vmin=1, vmax=10)
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 4 — Convert the mask to polygons and save
# ============================================================
#
# Goal: Vectorise the predicted mask into a GeoDataFrame of
# polygons (one polygon per connected region of same class),
# display them, and save the result as a parquet file.
#
# Steps:
#   1. Call create_geojson_from_mask() on labeled_satellite_img
#   2. Display 3 subplots: RGB / predicted mask / polygons
#   3. Save the GeoDataFrame to a parquet file
# ============================================================

gdf_pred = create_geojson_from_mask(__)  # TODO: labeled_satellite_img

print(f"{len(gdf_pred)} polygons extracted")
print(gdf_pred.head())

parts = image_target.split("/")
parts[-1] = parts[-1].rsplit(".", 1)[0]
image_target_join = "_".join(parts)
gdf_pred.to_parquet(f"predictions_{image_target_join}.parquet")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(labeled_satellite_img.label, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

gdf_pred.plot(
    column=__,   # TODO: colonne à utiliser pour la couleur (str)
    cmap=__,     # TODO: colormap
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
fig.savefig("prediction_polygons.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 4
# ------------------------------------------------------------
# - create_geojson_from_mask(lsi) takes a
#   SegmentationLabeledSatelliteImage and returns a GeoDataFrame
#   with columns "geometry" and "label"
# - Use column="label" in gdf_pred.plot() to colour by class
# - gdf_pred.total_bounds gives (xmin, ymin, xmax, ymax) to
#   fix the axis limits after geopandas resets them
# - The parquet filename is built from image_target automatically
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 4
# ------------------------------------------------------------
# gdf_pred = create_geojson_from_mask(labeled_satellite_img)
#
# gdf_pred.plot(column="label", cmap=cmap, vmin=1, vmax=10,
#               ax=axes[2], legend=False)
# ------------------------------------------------------------


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
    labeled_satellite_img.satellite_image.crs,
    "EPSG:4326",
    *labeled_satellite_img.satellite_image.bounds,
)
center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

ImageOverlay(
    image=__,                              # TODO: composite RGB normalisé
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(m)

gdf_pred_wgs84 = gdf_pred.to_crs("__")  # TODO: code EPSG cible pour Folium (str)

folium.GeoJson(
    gdf_pred_wgs84,
    style_function=lambda feature: {
        "fillColor": __,  # TODO: utiliser label_to_color pour colorier selon le label
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
# Goal: Given a GPS point (longitude, latitude), use the API
# endpoint /find_image to retrieve the filename of the
# Sentinel-2 tile that contains this point.
#
# API endpoint : GET /find_image
# Parameters   :
#   - lon_gps (float) : longitude in WGS84 (EPSG:4326)
#   - lat_gps (float) : latitude  in WGS84 (EPSG:4326)
#   - nuts_id (str)   : NUTS3 region identifier
#   - year    (int)   : year of the satellite images (2018-2024)
#
# Steps:
#   1. Choose a GPS point located in Luxembourg
#   2. Call the /find_image endpoint with the right parameters
#   3. Print the filename returned by the API
# ============================================================

lon_gps = __  # TODO: longitude en WGS84 (float), ex: 6.13 pour Luxembourg City
lat_gps = __  # TODO: latitude en WGS84 (float), ex: 49.61 pour Luxembourg City
nuts_id = "__"  # TODO: identifiant NUTS3 (str), ex: "LU000" pour le Luxembourg
year    = __  # TODO: année des images satellites (int, entre 2018 et 2024)

response_find = requests.get(
    f"{api_url}/__",  # TODO: nom de l'endpoint à appeler (str), ex: "find_image"
    params={
        "lon_gps": __,  # TODO: longitude définie plus haut
        "lat_gps": __,  # TODO: latitude définie plus haut
        "nuts_id": __,  # TODO: identifiant NUTS3 défini plus haut
        "year":    __,  # TODO: année définie plus haut
    },
)
response_find.raise_for_status()

image_filename = response_find.json()[0]
print(f"Image found: {image_filename}")

# ------------------------------------------------------------
# HINT — Exercise 1
# ------------------------------------------------------------
# - Luxembourg City is around longitude=6.13, latitude=49.61
# - The NUTS3 identifier for Luxembourg is "LU000"
# - The endpoint name is "find_image"
# - The response is a JSON list — take the first element [0]
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 1
# ------------------------------------------------------------
# lon_gps = 6.13
# lat_gps = 49.61
# nuts_id = "LU000"
# year    = 2024
#
# response_find = requests.get(
#     f"{api_url}/find_image",
#     params={
#         "lon_gps": lon_gps,
#         "lat_gps": lat_gps,
#         "nuts_id": nuts_id,
#         "year":    year,
#     },
# )
# response_find.raise_for_status()
#
# image_filename = response_find.json()[0]
# print(f"Image found: {image_filename}")
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 — Predict land cover for the found image
# ============================================================
#
# Goal: Using the filename returned by /find_image, call the
# /predict_image endpoint to get the segmentation mask and
# the predicted polygons for this image.
#
# API endpoint : GET /predict_image
# Parameters   :
#   - image    (str)  : image path as returned by /find_image
#   - polygons (bool) : if True, also returns predicted polygons
#
# The response contains:
#   - "mask"     : list of lists (H x W) with predicted class per pixel
#   - "polygons" : GeoJSON string with the predicted polygons
#
# Steps:
#   1. Call /predict_image with polygons=True
#   2. Parse the mask as a numpy array (key "mask")
#   3. Parse the polygons as a GeoDataFrame (key "polygons")
#   4. Print the mask shape and the number of polygons
# ============================================================

response_pred = requests.get(
    f"{api_url}/__",  # TODO: nom de l'endpoint (str), ex: "predict_image"
    params={
        "image":    __,  # TODO: nom du fichier retourné par /find_image
        "polygons": __,  # TODO: booléen True pour récupérer aussi les polygones
    },
)
response_pred.raise_for_status()

mask = np.array(response_pred.json()["__"])  # TODO: clé de la réponse contenant le masque (str)

gdf_pred = gpd.GeoDataFrame.from_features(
    json.loads(response_pred.json()["__"])["features"],  # TODO: clé contenant les polygones (str)
    crs="EPSG:3035",
)

print(f"Mask shape    : {mask.shape}")
print(f"Classes found : {set(mask.flatten().tolist())}")
print(f"{len(gdf_pred)} polygons extracted")

# ------------------------------------------------------------
# HINT — Exercise 2
# ------------------------------------------------------------
# - The endpoint name is "predict_image"
# - Pass image_filename (from Exercise 1) as the "image" parameter
# - Set polygons=True to get the GeoJSON polygons in the response
# - The response keys are "mask" and "polygons"
# - json.loads() parses the GeoJSON string before calling from_features()
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2
# ------------------------------------------------------------
# response_pred = requests.get(
#     f"{api_url}/predict_image",
#     params={
#         "image":    image_filename,
#         "polygons": True,
#     },
# )
# response_pred.raise_for_status()
#
# mask = np.array(response_pred.json()["mask"])
#
# gdf_pred = gpd.GeoDataFrame.from_features(
#     json.loads(response_pred.json()["polygons"])["features"],
#     crs="EPSG:3035",
# )
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 2 (continued) — Visualise the prediction
# ============================================================
#
# Goal: Load the RGB composite of the image and display it
# side by side with the predicted land cover mask and the
# predicted polygons.
#
# Base URL:
#   https://minio.lab.sspcloud.fr/projet-formation/
#   diffusion/funathon/2026/project3/data/images/
#
# Steps:
#   1. Build the full image URL (base_url + image_filename)
#   2. Load the satellite image with get_satellite_image()
#   3. Build the RGB composite (bands 4, 3, 2 → indices 3, 2, 1)
#   4. Display the 3 subplots: RGB / mask / polygons
# ============================================================

N_BANDS = 14

base_url = (
    "https://minio.lab.sspcloud.fr/projet-formation/"
    "diffusion/funathon/2026/project3/data/images/"
)

image_url = base_url + __  # TODO: nom du fichier image retourné par /find_image

si = get_satellite_image(__, n_bands=__)  # TODO: URL complète de l'image, nombre de bandes

# RGB composite — bands 4, 3, 2 correspond to indices 3, 2, 1
rgb = np.transpose(si["array"][[3, 2, 1]], (1, 2, 0)).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(mask, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

gdf_pred.plot(column="label", cmap=cmap, vmin=1, vmax=10, ax=axes[2], legend=False)
axes[2].set_title("Predicted polygons")
axes[2].set_aspect("equal")
xmin, ymin, xmax, ymax = gdf_pred.total_bounds
axes[2].set_xlim(xmin, xmax)
axes[2].set_ylim(ymin, ymax)
axes[2].axis("off")

fig.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
fig.savefig("prediction_single_image.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------
# HINT — Exercise 2 (continued)
# ------------------------------------------------------------
# - Concatenate base_url and image_filename to get the full URL
# - get_satellite_image(url, n_bands=14) returns a dict with
#   key "array" of shape (n_bands, H, W)
# - Bands 4, 3, 2 are at indices 3, 2, 1 (0-based)
# - Normalize: divide by np.percentile(rgb, 98) then clip to [0, 1]
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2 (continued)
# ------------------------------------------------------------
# image_url = base_url + image_filename
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
#   1. Reproject bounds to WGS84 with transform_bounds()
#   2. Create a folium.Map centred on the tile
#   3. Add an ImageOverlay with the RGB array
#   4. Reproject gdf_pred to EPSG:4326
#   5. Add a GeoJson layer coloured by label using label_to_color
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
        "fillColor": __,  # TODO: utiliser label_to_color pour colorier selon le label du polygone
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
# - feature["properties"]["label"] gives the class ID of a polygon
# - label_to_color is a dict {class_id: hex_color}
# - Use .get() with a fallback: label_to_color.get(..., "#808080")
# ------------------------------------------------------------

# ------------------------------------------------------------
# SOLUTION — Exercise 2 (Folium)
# ------------------------------------------------------------
# style_function=lambda feature: {
#     "fillColor": label_to_color.get(feature["properties"]["label"], "#808080"),
#     "color": "black",
#     "weight": 0.5,
#     "fillOpacity": 0.6,
# }
# ------------------------------------------------------------


# %%
# ============================================================
# EXERCISE 3 — Predict land cover for an entire NUTS3 region
# ============================================================
#
# Goal: Use the /predict_nuts endpoint to retrieve predictions
# for ALL Sentinel-2 tiles covering a given NUTS3 region and
# year. The server handles caching, so repeated calls are fast.
#
# API endpoint : GET /predict_nuts
# Parameters   :
#   - nuts_id (str) : NUTS3 region identifier
#   - year    (int) : year of the satellite images
#
# The response contains a "predictions" key with a GeoJSON string.
#
# Steps:
#   1. Call /predict_nuts for nuts_id="LU000" and year=2024
#   2. Parse the response as a GeoDataFrame (key "predictions")
#   3. Print the number of polygons and the first rows
#   4. Save the GeoDataFrame to a parquet file
# ============================================================

response_nuts = requests.get(
    f"{api_url}/__",  # TODO: nom de l'endpoint (str), ex: "predict_nuts"
    params={
        "nuts_id": "__",  # TODO: identifiant NUTS3 (str), ex: "LU000"
        "year":    __,    # TODO: année (int, entre 2018 et 2024)
    },
)
response_nuts.raise_for_status()

gdf_nuts = gpd.GeoDataFrame.from_features(
    json.loads(response_nuts.json()["__"])["features"],  # TODO: clé de la réponse (str)
    crs="EPSG:3035",
)

print(f"{len(gdf_nuts)} polygons received")
print(gdf_nuts.head())

# ------------------------------------------------------------
# HINT — Exercise 3
# ------------------------------------------------------------
# - The endpoint name is "predict_nuts"
# - The response key is "predictions" (a GeoJSON string)
# - Parse with json.loads(...) then ["features"] before from_features()
# - This call may take several minutes for a full NUTS3 region
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
#     json.loads(response_nuts.json()["predictions"])["features"],
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
#   4. Save gdf_nuts to a parquet file
# ============================================================

gdf_nuts_wgs84 = gdf_nuts.to_crs("EPSG:4326")
nuts_center = gdf_nuts_wgs84.geometry.centroid.unary_union.centroid

m_nuts = folium.Map(location=[nuts_center.y, nuts_center.x], zoom_start=10)

folium.GeoJson(
    __,  # TODO: GeoDataFrame des prédictions NUTS3 reprojeté en WGS84
    style_function=lambda feature: {
        "fillColor": __,  # TODO: utiliser label_to_color pour colorier selon le label
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(m_nuts)

gdf_nuts.to_parquet("__")  # TODO: nom du fichier parquet de sortie, ex: "predictions_LU000_2024.parquet"

m_nuts

# ------------------------------------------------------------
# HINT — Exercise 3 (continued)
# ------------------------------------------------------------
# - Pass gdf_nuts_wgs84 (already reprojected) to folium.GeoJson()
# - Same style_function pattern as Exercise 2:
#   label_to_color.get(feature["properties"]["label"], "#808080")
# - Choose a meaningful filename for the parquet output
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
# gdf_nuts.to_parquet("predictions_LU000_2024.parquet")
# ------------------------------------------------------------


# %%
# ============================================
# API (NUTS3-level predictions)
# ============================================

api_url = "https://funathon-2026-project3.lab.sspcloud.fr"

response = requests.get(
    f"{api_url}/predict_nuts",
    params={"nuts_id": "LU000", "year": 2024},
)
response.raise_for_status()

gdf_nuts = gpd.GeoDataFrame.from_features(
    response.json()["predictions"], crs="EPSG:3035"
)

print(f"{len(gdf_nuts)} polygons received")
print(gdf_nuts.head())

gdf_nuts.to_parquet("predictions_LU000_2024.parquet")
