# %%
# ============================================
# STEP 3 — Inference
# ============================================

# %%
# ============================================
# Load the Trained Model
# ============================================
import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from dotenv import load_dotenv
import requests
import geopandas as gpd
from utils import (
    get_model,
    get_normalization_metrics,
    predict,
    produce_mask,
    create_geojson_from_mask,
)

load_dotenv()

# %%
# =========================
# Load model from MLflow
# =========================

model_name = ___  # TODO: the model name registered
model_version = ___  # TODO: the model version registered
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

model = get_model(model_name, model_version, mlflow_tracking_uri)

# %%
# =========================
# Retrieve model metadata
# =========================

run = mlflow.get_run(model.metadata.run_id)

n_bands = int(run.data.params["n_bands"])
tiles_size = int(run.data.params["tiles_size"])
augment_size = int(run.data.params["augment_size"])
module_name = run.data.params["module_name"]

normalization_mean, normalization_std = get_normalization_metrics(model, n_bands)

print(f"n_bands={n_bands}, tiles_size={tiles_size}, augment_size={augment_size}")
print(f"mean={normalization_mean}")
print(f"std={normalization_std}")


# %%
# ============================================
# Apply the Model
# ============================================

# %%
# =========================
# Predict on a single Sentinel-2 image
# =========================

image_path = (
    "projet-formation/diffusion/funathon/2026/project3/"
    "data/images/LU000/2024/3034500_2011690.tif"
)

labelled_satellite_img = predict(
    images=image_path,
    model=model,
    tiles_size=tiles_size,
    augment_size=augment_size,
    n_bands=n_bands,
    normalization_mean=normalization_mean,
    normalization_std=normalization_std,
    module_name=module_name,
)

# %%
# =========================
# Produce the class mask
# =========================

labelled_satellite_img.label = produce_mask(labelled_satellite_img.label, module_name)

print(f"Mask shape    : {labelled_satellite_img.label.shape}")
print(f"Classes found : {set(labelled_satellite_img.label.flatten().tolist())}")

# %%
# =========================
# Display the prediction
# =========================

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

# RGB composite for visualisation (bands 4, 3, 2)
rgb = np.transpose(
    labelled_satellite_img.satellite_image.array[[3, 2, 1]], (1, 2, 0)
).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(labelled_satellite_img.label, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.35, 0.5),
    frameon=True,
)

plt.tight_layout()
plt.show()

# %%
# =========================
# Convert predictions to polygons
# =========================

gdf_pred = create_geojson_from_mask(labelled_satellite_img)

print(f"{len(gdf_pred)} polygons extracted")
print(gdf_pred.head())

gdf_pred.to_parquet("predictions_LU000_3034500_2011690_2024.parquet")

# %%
# =========================
# Overlay polygons on image
# =========================

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb)

gdf_pred.plot(
    column="label",
    cmap=cmap,
    vmin=1,
    vmax=10,
    alpha=0.5,
    ax=ax,
    legend=False,
)

ax.set_title("Predicted polygons overlay")
ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# ============================================
# API (NUTS3-level predictions)
# ============================================

api_url = "https://projet-formation-api.user.lab.sspcloud.fr"

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
