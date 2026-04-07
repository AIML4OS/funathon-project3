# %%
# ============================================
# STEP 1 — Data Acquisition
# ============================================
#
# In this step you will:
#   1. (Optional) Query the Sentinel Hub Catalog API
#   2. (Optional) Download a Sentinel-2 image with the Processing API and upload to S3
#   3. Read a Sentinel-2 tile via HTTPS and display it
#   4. Explore raster metadata and create a false-colour composite
#   5. Compute and display NDVI
#   6. Build a GeoDataFrame and convert CRS
#   7. Spatial join with NUTS3 regions
#   8. Display a tile on an interactive folium map
#   9. Download a CLC+ label
#  10. Save label to S3 as .npy
#  11. Plot satellite image and label side by side
#
# Each section has TODO comments telling you what to fill in.
# The variable names and imports are already provided.
# ============================================


# %%
# ==============================================
# Exercise 1 (optional) — Query the Sentinel Hub Catalog API
# ==============================================
# Requires a free CDSE account + OAuth client:
#   https://dataspace.copernicus.eu/
#   https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings

# from oauthlib.oauth2 import BackendApplicationClient
# from requests_oauthlib import OAuth2Session

# # Step 1a: Authenticate with OAuth2
# # TODO: Fill in your CDSE OAuth client ID and secret
# client_id = ___
# client_secret = ___

# client = BackendApplicationClient(client_id=client_id)
# oauth = OAuth2Session(client=client)
# # TODO: Fetch the token from the CDSE token endpoint
# # Hint: "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
# token = oauth.fetch_token(
#     token_url=___,
#     client_secret=client_secret,
#     include_client_id=True,
# )

# # Step 1b: Build the Catalog API request body
# catalog_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
# search_body = {
#     "collections": ["sentinel-2-l2a"],  # TODO: ["sentinel-2-l2a"]
#     "datetime": "2021-06-01T00:00:00Z/2021-06-30T23:59:59Z",  # TODO: "2021-06-01T00:00:00Z/2021-06-30T23:59:59Z"
#     "bbox": [
#         22.0,
#         41.0,
#         29.0,
#         44.5,
#     ],  # TODO: Bulgaria bounding box [22.0, 41.0, 29.0, 44.5]
#     "limit": 50,
#     "filter": "eo:cloud_cover < 10",  # TODO: CQL2 cloud cover filter, e.g. "eo:cloud_cover < 10"
#     "filter-lang": "cql2-text",
# }

# # Step 1c: POST to the Catalog API
# response = ___  # TODO: oauth.post(catalog_url, json=search_body)
# results = response.json()

# # Step 1d: Print the number of features
# print(f"Found {len(results['features'])} products")


# # %%
# # ==============================================
# # Exercise 2 (optional) — Download a Sentinel-2 image and upload to S3
# # ==============================================
# # Requires an authenticated OAuth2 session from Exercise 1

# from utils import get_file_system

# # Step 2a: Build the Processing API request
# process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

# evalscript = """
# //VERSION=3
# function setup() {
#   return { input: ["B04", "B03", "B02"], output: { bands: 3, sampleType: "UINT16" } };
# }
# function evaluatePixel(sample) {
#   return [sample.B04, sample.B03, sample.B02];
# }
# """

# request_body = {
#     "input": {
#         "bounds": {
#             "bbox": ___,  # TODO: Bulgaria bounding box [22.0, 41.0, 29.0, 44.5]
#             "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
#         },
#         "data": [
#             {
#                 "type": ___,  # TODO: "sentinel-2-l2a"
#                 "dataFilter": {
#                     "timeRange": {
#                         "from": ___,  # TODO: "2021-06-01T00:00:00Z"
#                         "to": ___,  # TODO: "2021-06-30T23:59:59Z"
#                     },
#                     "maxCloudCoverage": ___,  # TODO: 10
#                 },
#             }
#         ],
#     },
#     "output": {
#         "width": 512,
#         "height": 512,
#         "responses": [
#             {"identifier": "default", "format": {"type": ___}}
#         ],  # TODO: "image/tiff"
#     },
#     "evalscript": evalscript,
# }

# # Step 2b: POST and save locally
# response = ___  # TODO: oauth.post(process_url, json=request_body)
# local_path = "bulgaria_rgb.tif"
# with open(local_path, "wb") as f:
#     f.write(___)  # TODO: response.content

# # Step 2c: Upload to S3
# fs = ___  # TODO: get_file_system()
# fs.put(___, ___)  # TODO: local_path, S3 destination path


# %%
# ==============================================
# Exercise 3 — Read a Sentinel-2 tile and display it
# ==============================================

import rasterio
import numpy as np
import matplotlib.pyplot as plt

tile_url = (
    "https://minio.lab.sspcloud.fr/projet-hackathon-ntts-2025/"
    "data-preprocessed/patchs/CLCplus-Backbone/SENTINEL2/"
    "FRJ27/2018/250/3649890_2331750_0_937.tif"
)

# Step 3a: Open the tile with rasterio and read RGB bands (4, 3, 2)
# TODO: Use rasterio.open(tile_url), then src.read([4, 3, 2]), src.crs, src.bounds
with rasterio.open(tile_url) as src:
    rgb_data = ___
    tile_crs = ___
    tile_bounds = ___

# Step 3b: Transpose to (H, W, 3) and normalize for display
# TODO: np.transpose(rgb_data, (1, 2, 0)), cast to float32,
#       divide by 98th percentile, clip to [0, 1]
rgb = ___

# Step 3c: Display the RGB composite
fig, ax = plt.subplots(figsize=(5, 5))
___  # TODO: ax.imshow(rgb)
ax.set_title("Sentinel-2 RGB composite")
ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# ==============================================
# Exercise 4 — Explore raster metadata and create a false-colour composite
# ==============================================

# Step 4a: Open the tile and print the raster profile
# TODO: Use rasterio.open(tile_url), then print(src.profile)
with rasterio.open(tile_url) as src:
    print(___)  # TODO: src.profile

    # Step 4b: Read false-colour bands (NIR=8, Red=4, Green=3)
    # TODO: src.read([8, 4, 3])
    fc_data = ___

# Step 4c: Normalize for display
# TODO: transpose to (H, W, 3), cast to float32, divide by 98th percentile, clip
fc = np.transpose(fc_data, (1, 2, 0)).astype(np.float32)
p98 = ___  # TODO: np.percentile(fc, 98)
fc = ___  # TODO: np.clip(fc / p98, 0, 1)

# Step 4d: Display side by side with the true-colour RGB
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb)
axes[0].set_title("True colour (B4, B3, B2)")
axes[0].axis("off")
___  # TODO: axes[1].imshow(fc)
axes[1].set_title("False colour (B8, B4, B3)")
axes[1].axis("off")
plt.tight_layout()
plt.show()


# %%
# ==============================================
# Exercise 5 — Compute and display NDVI
# ==============================================

# Step 5a: Read NIR (band 8) and Red (band 4) as float32
# TODO: src.read(8).astype(np.float32) and src.read(4).astype(np.float32)
with rasterio.open(tile_url) as src:
    nir = ___  # TODO: src.read(8).astype(np.float32)
    red = ___  # TODO: src.read(4).astype(np.float32)

# Step 5b: Compute NDVI (handle division by zero)
# TODO: np.where(nir + red == 0, 0, (nir - red) / (nir + red))
ndvi = ___

# Step 5c: Display
fig, ax = plt.subplots(figsize=(6, 5))
im = ___  # TODO: ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("NDVI — FRJ27 (2018)")
ax.axis("off")
___  # TODO: fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
plt.tight_layout()
plt.show()


# %%
# ==============================================
# Exercise 6 — Build a GeoDataFrame and convert CRS
# ==============================================

import geopandas as gpd
from shapely.geometry import box

# Step 6a: Create a box geometry from tile_bounds
# TODO: box(*tile_bounds)
tile_geom = ___

# Step 6b: Build a GeoDataFrame in EPSG:3035
# TODO: gpd.GeoDataFrame({"tile": ["FRJ27"]}, geometry=[tile_geom], crs="EPSG:3035")
gdf = ___

# Step 6c: Convert to WGS84
# TODO: gdf.to_crs("EPSG:4326")
gdf_wgs84 = ___

# Step 6d: Print bounds in both CRS
print("EPSG:3035 bounds:", gdf.total_bounds)
print("EPSG:4326 bounds:", gdf_wgs84.total_bounds)


# %%
# ==============================================
# Exercise 7 — Spatial join with NUTS3 regions
# ==============================================

# Step 7a: Load NUTS3 boundaries (EPSG:3035)
# TODO: gpd.read_file(nuts_url)
nuts_url = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "nuts/geojson/NUTS_RG_01M_2021_3035_LEVL_3.geojson"
)
nuts = ___

# Step 7b: Create a GeoDataFrame for the tile
# TODO: gpd.GeoDataFrame({"tile": ["FRJ27"]}, geometry=[box(*tile_bounds)], crs=tile_crs)
tile_gdf = ___

# Step 7c: Spatial join — find which NUTS3 regions the tile intersects
# TODO: gpd.sjoin(tile_gdf, nuts, predicate="intersects")
joined = ___

# Step 7d: Print matching regions
for _, row in joined.iterrows():
    print(f"NUTS_ID: {row['NUTS_ID']}, NUTS_NAME: {row['NUTS_NAME']}")

# Step 7e: Compute tile area in km²
# TODO: box(*tile_bounds).area / 1e6
area_km2 = ___
print(f"Tile area: {area_km2:.2f} km²")


# %%
# ==============================================
# Exercise 8 — Display a tile on an interactive folium map
# ==============================================

import folium
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds

# Step 8a: Reproject bounds to WGS84
# TODO: transform_bounds(tile_crs, "EPSG:4326", *tile_bounds)
west, south, east, north = ___

# Step 8b: Compute the centre
# TODO: average of south/north and west/east
center_lat = ___
center_lon = ___

# Step 8c: Create the folium map
# TODO: folium.Map(location=[center_lat, center_lon], zoom_start=14)
m = ___

# Step 8d: Add the satellite image overlay
# TODO: ImageOverlay(image=rgb, bounds=[[south, west], [north, east]], opacity=0.7).add_to(m)
___

m


# %%
# ==============================================
# Exercise 9 — Download a CLC+ label
# ==============================================

from utils import download_clcpluslabel, tiff_to_numpy

# These coordinates match the FRJ27 tile above (EPSG:3035)
bbox = [3649890, 2331750, 3652390, 2334250]
year = 2018
filename = "test_label.tif"

# Step 9a: Download the CLC+ label
# TODO: Call download_clcpluslabel(filename, bbox, year)
___

# Step 9b: Convert the downloaded TIFF to a NumPy array
# TODO: Call tiff_to_numpy(filename) and store in `img_array`
# Expected output: a 2D array of integer class IDs (values 1-10)
img_array = ___


# %%
# ==============================================
# Exercise 10 — Save the label as .npy and upload to S3
# ==============================================

from utils import get_file_system

# Step 10a: Save the label array locally as .npy
local_path = "3649890_2331750_0_937.npy"
___  # TODO: np.save(local_path, img_array)

# Step 10b: Initialise the S3 filesystem
# TODO: Call get_file_system()
fs = ___

# Step 10c: Build the S3 destination path
# TODO: Use the training pipeline convention:
# "projet-hackathon-ntts-2025/data-preprocessed/labels/CLCplus-Backbone/SENTINEL2/FRJ27/2018/250/3649890_2331750_0_937.npy"
s3_path = ___

# Step 10d: Upload to S3
# TODO: fs.put(local_path, s3_path)
___

# Step 10e: Verify the upload
# TODO: fs.ls() on the S3 directory
print(fs.ls(___))


# %%
# ==============================================
# Exercise 11 — Plot the image and label side by side
# ==============================================

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Classes & colormap (provided -- do not modify)
classes = [
    ("Sealed (1)", "#FF0100"),
    ("Woody -- needle leaved trees (2)", "#238B23"),
    ("Woody -- Broadleaved deciduous trees (3)", "#80FF00"),
    ("Woody -- Broadleaved evergreen trees (4)", "#00FF00"),
    ("Low-growing woody plants (bushes, shrubs) (5)", "#804000"),
    ("Permanent herbaceous (6)", "#CCF24E"),
    ("Periodically herbaceous (7)", "#FEFF80"),
    ("Lichens and mosses (8)", "#FF81FF"),
    ("Non- and sparsely-vegetated (9)", "#BFBFBF"),
    ("Water (10)", "#0080FF"),
]

cmap = ListedColormap([color for _, color in classes])

# Step 11a: Create a figure with two subplots side by side
# TODO: Call plt.subplots(1, 2, figsize=(12, 6)) and store in `fig, axes`
fig, axes = ___

# Step 11b: Display the Sentinel-2 RGB composite on the left subplot
# TODO: Use axes[0].imshow(rgb), then set a title and turn off axes
___
___
___

# Step 11c: Display the CLC+ label on the right subplot
# TODO: Use axes[1].imshow(img_array, cmap=cmap, vmin=1, vmax=10)
___
___
___

# Step 11d: Add a legend
# TODO: Build legend_elements using Patch(facecolor=color, edgecolor="black", label=label)
#       for each (label, color) in classes, then call fig.legend(...)
legend_elements = [___ for label, color in classes]

fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.35, 0.5),
    frameon=True,
)

plt.tight_layout()
plt.show()
