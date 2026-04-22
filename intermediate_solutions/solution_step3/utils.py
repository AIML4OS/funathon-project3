"""
Utils.
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
        float(v) for v in eval(
            mlflow.get_run(model.metadata.run_id).data.params["normalization_std"]
        )
    ]
    return normalization_mean[:n_bands], normalization_std[:n_bands]


def get_satellite_image(image_path: str, n_bands: int) -> dict:
    """
    Lit une image satellite avec rasterio.

    Returns:
        dict avec keys: array, crs, bounds, transform
    """
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
            tiles.append({
                "array": image["array"][:, row:row + tile_size, col:col + tile_size],
                "crs": image["crs"],
                "bounds": rasterio.transform.array_bounds(
                    tile_size, tile_size, tile_transform
                ),
                "transform": tile_transform,
            })
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
        label_rows.append(np.concatenate([labels[r * n_cols + c] for c in range(n_cols)], axis=1))
        array_rows.append(np.concatenate([tiles[r * n_cols + c]["array"] for c in range(n_cols)], axis=2))

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
        A.Normalize(max_pixel_value=1.0, mean=normalization_mean, std=normalization_std),
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
    transform = get_transform(tiles_size, augment_size, normalization_mean, normalization_std)
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
                for _, (s, v) in enumerate(shapes(label, mask=None, transform=dst.transform))
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
    Prédit le masque de segmentation pour une tuile.

    Returns:
        Tuple (image dict, label array)
    """
    normalized = preprocess_image(image, tiles_size, augment_size, normalization_mean, normalization_std)

    with torch.no_grad():
        prediction = model.predict(normalized.numpy())

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
            return make_prediction(si, model, tiles_size, augment_size, n_bands,
                                   normalization_mean, normalization_std, module_name)

        elif si["array"].shape[1] > tiles_size:
            if si["array"].shape[1] % tiles_size != 0:
                raise ValueError("Image dimension must be divisible by tile size.")

            tile_images = split_image(si, tiles_size)
            results = [
                make_prediction(t, model, tiles_size, augment_size, n_bands,
                                normalization_mean, normalization_std, module_name)
                for t in tqdm(tile_images)
            ]
            tile_imgs, tile_labels = zip(*results)
            return make_mosaic(list(tile_imgs), list(tile_labels))

        else:
            raise ValueError("Image dimension must be >= tile size.")

    if isinstance(images, str):
        return predict_single(images)
    else:
        return [predict(img, model, tiles_size, augment_size, n_bands,
                        normalization_mean, normalization_std, module_name)
                for img in tqdm(images)]
