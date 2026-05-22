"""
Shared utilities for the satellite-segmentation pipeline.

Three families of helpers:

- I/O: read GeoTIFFs from MinIO/S3 over HTTPS via GDAL's `/vsicurl/`
  (`get_satellite_image`, `get_file_system`), and download CLC+ labels from
  the Copernicus ArcGIS REST endpoint (`download_clcpluslabel`).
- Tiling: split a large Sentinel-2 image into model-sized patches
  (`split_image`) and reassemble the per-patch predictions while preserving
  CRS / bounds / transform (`make_mosaic`).
- Inference: normalise inputs with the training-time statistics
  (`preprocess_image`), run the model and upsample logits (`make_prediction`),
  and orchestrate the full pipeline (`predict`). `create_geojson_from_mask`
  vectorises a class mask into polygons via `rasterio.features.shapes`.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from typing import List, Tuple

import albumentations as A
import geopandas as gpd
import mlflow
import numpy as np
import rasterio
import torch
from albumentations.pytorch.transforms import ToTensorV2
from rasterio.features import shapes
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
import os
from s3fs import S3FileSystem


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
    )


def download_label(format_ext, filename, common_params, export_url):
    params = common_params.copy()
    params["format"] = format_ext

    response = requests.get(export_url, params=params, stream=True)

    if response.status_code == 200 and response.headers.get(
        "content-type", ""
    ).startswith("image/"):
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Erreur {format_ext.upper()} : ", response.status_code, response.text)


def download_clcpluslabel(format_ext, bbox_tuple, year):
    export_url = f"https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/CLC_plus/CLMS_CLCplus_RASTER_{year}_010m_eu/ImageServer/exportImage"

    xmin, ymin, xmax, ymax = bbox_tuple

    # 1 pixel = 10 m
    resolution = 10
    size_x = int((xmax - xmin) / resolution)
    size_y = int((ymax - ymin) / resolution)

    bbox_str = f"{xmin},{ymin},{xmax},{ymax}"

    common_params = {
        "f": "image",
        "bbox": bbox_str,
        "bboxSR": "3035",  # Lambert-93
        "imageSR": "3035",  # Lambert-93
        "size": f"{size_x},{size_y}",  # 1 pixel = 10 m
    }

    download_label("tiff", format_ext, common_params, export_url)


def tiff_to_numpy(format_ext):
    img = Image.open(format_ext)
    img_array = np.array(img)
    img_array[(img_array == 254) | (img_array == 255)] = 0

    npy_format_ext = format_ext.replace(".tif", ".npy")
    np.save(npy_format_ext, img_array)
    os.remove(format_ext)

    return img_array


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------


def get_normalization_metrics(
    model: mlflow.pyfunc.PyFuncModel, n_bands: int
) -> Tuple[List[float], List[float]]:
    normalization_mean = json.loads(
        mlflow.get_run(model.metadata.run_id).data.params["normalization_mean"]
    )
    normalization_std = [
        float(v)
        for v in eval(
            mlflow.get_run(model.metadata.run_id).data.params["normalization_std"]
        )
    ]
    return normalization_mean[:n_bands], normalization_std[:n_bands]


def get_satellite_image(image_path: str, n_bands: int) -> dict:
    """
    Read a satellite image with rasterio. Accepts a local path or an HTTPS URL.

    Returns a dict carrying the pixel array plus its geographic metadata
    (`crs`, `bounds`, `transform`) — keeping those alongside the pixels is what
    later lets us turn predictions back into real-world polygons.
    """
    # `/vsicurl/` is GDAL's virtual file system: rasterio streams the bytes it
    # needs from the HTTPS endpoint without downloading the whole GeoTIFF locally.
    if image_path.startswith("https://") or image_path.startswith("http://"):
        file_path = f"/vsicurl/{image_path}"
    else:
        file_path = image_path

    with rasterio.open(file_path) as src:
        return {
            "array": src.read(list(range(1, n_bands + 1))).astype(np.float32),
            "crs": src.crs,
            "bounds": src.bounds,
            "transform": src.transform,
        }


def split_image(image: dict, tile_size: int) -> List[dict]:
    """Découpe une image en tuiles carrées."""
    _, H, W = image["array"].shape
    tiles = []
    for row in range(0, H, tile_size):
        for col in range(0, W, tile_size):
            tile_transform = rasterio.transform.from_origin(
                image["bounds"].left + col * abs(image["transform"].a),
                image["bounds"].top + row * image["transform"].e,
                abs(image["transform"].a),
                abs(image["transform"].e),
            )
            tiles.append(
                {
                    "array": image["array"][
                        :, row : row + tile_size, col : col + tile_size
                    ],
                    "crs": image["crs"],
                    "bounds": rasterio.transform.array_bounds(
                        tile_size, tile_size, tile_transform
                    ),
                    "transform": tile_transform,
                }
            )
    return tiles


def make_mosaic(tiles: List[dict], labels: List[np.ndarray]) -> Tuple[dict, np.ndarray]:
    """
    Reconstruit une image et son masque depuis des tuiles ordonnées.

    Returns:
        Tuple (image dict, label array)
    """
    n_tiles = len(tiles)
    n_cols = int(np.sqrt(n_tiles))
    n_rows = n_tiles // n_cols

    label_rows, array_rows = [], []
    for r in range(n_rows):
        label_rows.append(
            np.concatenate([labels[r * n_cols + c] for c in range(n_cols)], axis=1)
        )
        array_rows.append(
            np.concatenate(
                [tiles[r * n_cols + c]["array"] for c in range(n_cols)], axis=2
            )
        )

    full_label = np.concatenate(label_rows, axis=0)
    full_array = np.concatenate(array_rows, axis=1)

    first, last = tiles[0], tiles[-1]
    full_image = {
        "array": full_array,
        "crs": first["crs"],
        "bounds": rasterio.coords.BoundingBox(
            left=first["bounds"].left,
            bottom=last["bounds"].bottom,
            right=last["bounds"].right,
            top=first["bounds"].top,
        ),
        "transform": first["transform"],
    }
    return full_image, full_label


def get_transform(
    tiles_size: int,
    augment_size: int,
    normalization_mean: List[float],
    normalization_std: List[float],
) -> A.Compose:
    transform_list = [
        A.Normalize(
            max_pixel_value=1.0, mean=normalization_mean, std=normalization_std
        ),
        ToTensorV2(),
    ]
    if augment_size != tiles_size:
        transform_list.insert(0, A.Resize(augment_size, augment_size))
    return A.Compose(transform_list)


def preprocess_image(
    image: dict,
    tiles_size: int,
    augment_size: int,
    normalization_mean: List[float],
    normalization_std: List[float],
) -> torch.Tensor:
    transform = get_transform(
        tiles_size, augment_size, normalization_mean, normalization_std
    )
    arr = image["array"]
    if len(normalization_mean) != arr.shape[0]:
        arr = arr[: len(normalization_mean)]
    return transform(image=np.transpose(arr, [1, 2, 0]))["image"].unsqueeze(dim=0)


@contextmanager
def temporary_raster():
    """Context manager for handling temporary raster files safely."""
    temp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    try:
        temp.close()
        yield temp.name
    finally:
        try:
            os.unlink(temp.name)
        except OSError:
            pass


def create_geojson_from_mask(image: dict, label: np.ndarray) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame from a segmentation mask.

    Args:
        image: dict avec keys array, crs, bounds, transform.
        label: Masque de segmentation (H, W).

    Returns:
        GeoDataFrame avec colonnes geometry et label.
    """
    label = label.astype("uint8")

    metadata = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": image["array"].shape[2],
        "height": image["array"].shape[1],
        "crs": image["crs"],
        "transform": rasterio.transform.from_origin(
            image["bounds"].left, image["bounds"].top, 10, 10
        ),
    }

    with temporary_raster() as temp_tif:
        with rasterio.open(temp_tif, "w+", **metadata) as dst:
            dst.write(label, 1)
            results = [
                {"properties": {"label": int(v)}, "geometry": s}
                for _, (s, v) in enumerate(
                    shapes(label, mask=None, transform=dst.transform)
                )
                if v != 0
            ]

    if results:
        return gpd.GeoDataFrame.from_features(results, crs=image["crs"])
    else:
        return gpd.GeoDataFrame(columns=["geometry", "label"])


def make_prediction(
    image: dict,
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    module_name: str,
) -> Tuple[dict, np.ndarray]:
    """
    Predict the segmentation mask for a single tile.

    Returns:
        Tuple (image dict, label array). The label is a 2D int array of class
        IDs, same H×W as the model's tile_size (after upsampling).
    """
    # Normalisation, optional resize, and HWC→CHW/batch-axis are bundled here
    # because inference *must* reproduce training-time preprocessing exactly.
    normalized = preprocess_image(
        image, tiles_size, augment_size, normalization_mean, normalization_std
    )

    with torch.no_grad():
        prediction = model.predict(normalized.numpy())

    # SegFormer's all-MLP head emits logits at H/4 × W/4. Upsample back to the
    # tile size with bilinear (not nearest) interpolation: we're upsampling
    # continuous per-class scores, not discrete labels.
    if prediction.shape[-2:] != (tiles_size, tiles_size):
        prediction = (
            torch.nn.functional.interpolate(
                torch.from_numpy(prediction),
                size=tiles_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

    # argmax over the class axis: from (n_classes, H, W) of logits to (H, W) of class IDs.
    label = np.argmax(prediction, axis=0).astype(np.int32)
    return image, label


def predict(
    images,
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    module_name: str,
):
    """
    Prédit le(s) masque(s) pour une ou plusieurs images.

    Returns:
        Tuple (image, label) ou liste de tuples.
    """

    def predict_single(image_path):
        si = get_satellite_image(image_path, n_bands)

        if si["array"].shape[1] == tiles_size:
            return make_prediction(
                si,
                model,
                tiles_size,
                augment_size,
                n_bands,
                normalization_mean,
                normalization_std,
                module_name,
            )

        elif si["array"].shape[1] > tiles_size:
            if si["array"].shape[1] % tiles_size != 0:
                raise ValueError("Image dimension must be divisible by tile size.")

            tile_images = split_image(si, tiles_size)
            results = [
                make_prediction(
                    t,
                    model,
                    tiles_size,
                    augment_size,
                    n_bands,
                    normalization_mean,
                    normalization_std,
                    module_name,
                )
                for t in tqdm(tile_images)
            ]
            tile_imgs, tile_labels = zip(*results)
            return make_mosaic(list(tile_imgs), list(tile_labels))

        else:
            raise ValueError("Image dimension must be >= tile size.")

    if isinstance(images, str):
        return predict_single(images)
    else:
        return [
            predict(
                img,
                model,
                tiles_size,
                augment_size,
                n_bands,
                normalization_mean,
                normalization_std,
                module_name,
            )
            for img in tqdm(images)
        ]
