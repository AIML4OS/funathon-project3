"""
Utils.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from typing import List

import albumentations as A
import geopandas as gpd
import mlflow
import numpy as np
import pyarrow.parquet as pq
import rasterio
import torch
from albumentations.pytorch.transforms import ToTensorV2
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from astrovision.plot import make_mosaic
from rasterio.features import shapes
from s3fs import S3FileSystem
from tqdm import tqdm

from app.logger_config import configure_logger

logger = configure_logger()


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=""
    )


def get_model(
    model_name: str, model_version: str, mlflow_tracking_uri: str
) -> mlflow.pyfunc.PyFuncModel:
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.
    Returns:
        model (mlflow.pyfunc.PyFuncModel): The loaded machine learning model.
    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error


def get_normalization_metrics(model: mlflow.pyfunc.PyFuncModel, n_bands: int):
    """
    Retrieves normalization metrics (mean and standard deviation) for the model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        n_bands (int): Number of bands in the satellite image.

    Returns:
        Tuple: A tuple containing normalization mean and standard deviation.
    """
    normalization_mean = json.loads(
        mlflow.get_run(model.metadata.run_id).data.params["normalization_mean"]
    )
    normalization_std_float = eval(
        mlflow.get_run(model.metadata.run_id).data.params["normalization_std"]
    )
    normalization_std = [float(value) for value in normalization_std_float]

    # Extract normalization mean and standard deviation for the number of bands
    normalization_mean, normalization_std = (
        normalization_mean[:n_bands],
        normalization_std[:n_bands],
    )

    return (normalization_mean, normalization_std)


def get_satellite_image(image_path: str, n_bands: int):
    """
    Retrieves a satellite image specified by its path.

    Args:
        image_path (str): Path to the satellite image.
        n_bands (int): Number of bands in the satellite image.

    Returns:
        SatelliteImage: An object representing the satellite image.
    """

    # Load satellite image using the specified path and number of bands
    si = SatelliteImage.from_raster(
        file_path=f"/vsis3/{image_path}",
        dep=None,
        date=None,
        n_bands=n_bands,
    )
    return si


def get_transform(
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
):
    """
    Retrieves the transformation pipeline for image preprocessing.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        tiles_size (int): Size of the satellite image.
        augment_size (int): .
        n_bands: (int): .
        normalization_mean List[float]: .
        normalization_std List[float]: .

    Returns:
        albumentations.Compose: A composition of image transformations.
    """
    # Define the transformation pipeline
    transform_list = [
        A.Normalize(
            max_pixel_value=1.0,
            mean=normalization_mean,
            std=normalization_std,
        ),
        ToTensorV2(),
    ]

    # Add resizing transformation if augment size is different from tiles size
    if augment_size != tiles_size:
        transform_list.insert(0, A.Resize(augment_size, augment_size))

    transform = A.Compose(transform_list)

    return transform


def preprocess_image(
    model: mlflow.pyfunc.PyFuncModel,
    image: SatelliteImage,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
):
    """
    Preprocesses a satellite image using the specified model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        image (SatelliteImage): SatelliteImage object representing the input image.
        tiles_size (int): Size of the satellite image.
        augment_size (int): .
    Returns:
        torch.Tensor: Normalized and preprocessed image tensor.
    """
    # Obtain transformation pipeline
    transform = get_transform(
        model, tiles_size, augment_size, n_bands, normalization_mean, normalization_std
    )

    # Deal when images to pred have more channels than images used during training
    if len(normalization_mean) != image.array.shape[0]:
        image.array = image.array[: len(normalization_mean)]

    # Apply transformation to image
    normalized_si = transform(image=np.transpose(image.array, [1, 2, 0]))["image"].unsqueeze(dim=0)

    return normalized_si


def produce_mask(
    prediction: np.array,
    module_name: str,
):
    """
    Produces mask from prediction array based on the specified model.

    Args:
        prediction (np.array): Array containing the prediction.
        module_name (str): Name of the module used for training.

    Returns:
        np.array: Mask generated from prediction array.
    """

    # Make prediction torch tensor
    prediction = torch.from_numpy(prediction)
    mask = torch.argmax(prediction, dim=0)

    return mask.numpy()


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


def create_geojson_from_mask(lsi: SegmentationLabeledSatelliteImage) -> str:
    """
    Creates a Geopandas from a binary or multiclass mask.
    Args:
        lsi: A SegmentationLabeledSatelliteImage.
    Returns:
        A Geopandas representing the clusters with their respective labels.
    """
    # Convert label to uint8
    lsi.label = lsi.label.astype("uint8")

    # Define the metadata for the raster image
    metadata = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": lsi.label.shape[1],
        "height": lsi.label.shape[0],
        "crs": lsi.satellite_image.crs,
        "transform": rasterio.transform.from_origin(
            lsi.satellite_image.bounds[0], lsi.satellite_image.bounds[3], 10, 10
        ),
    }

    # Use the context manager for temporary file handling
    with temporary_raster() as temp_tif:
        with rasterio.open(temp_tif, "w+", **metadata) as dst:
            dst.write(lsi.label, 1)

            # Process shapes within the same rasterio context
            results = [
                {"properties": {"label": int(v)}, "geometry": s}
                for i, (s, v) in enumerate(shapes(lsi.label, mask=None, transform=dst.transform))
                if v != 0  # Keep only the labels which are not 0
            ]

    # Create and return GeoDataFrame
    if results:
        return gpd.GeoDataFrame.from_features(results)
    else:
        return gpd.GeoDataFrame(columns=["geometry", "label"])


def make_prediction(
    image: SatelliteImage,
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    module_name: str,
):
    """
    Makes a prediction on a satellite image.

    Args:
        image (SatelliteImage): The input satellite image.
        model: The ML model.
        tiles_size: The size of the tiles used during training.
        augment_size: The size of the augmentation used during training.
        n_bands: The number of bands in the satellite image.
        normalization_mean: The mean value used for normalization.
        normalization_std: The standard deviation used for normalization.

    Returns:
        SegmentationLabeledSatelliteImage: The labeled satellite image with the predicted mask.
    """
    # Preprocess the image
    normalized_si = preprocess_image(
        model=model,
        image=image,
        tiles_size=tiles_size,
        augment_size=augment_size,
        n_bands=n_bands,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    # Make prediction using the model
    with torch.no_grad():
        prediction = model.predict(normalized_si.numpy())

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

    return SegmentationLabeledSatelliteImage(image, prediction, logits=True)


def predict(
    images: str | list[str],
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    module_name: str,
):
    """
    Predicts mask for a given satellite image or a given list of given satellite image.

    Args:
        image (str or list): Path to the satellite image(s).
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        tiles_size (int): Size of the satellite image.
        augment_size (int): Size of the augmentation.
        n_bands (int): Number of bands in the satellite image.
        normalization_mean (List[float]): List of mean values for normalization.
        normalization_std (List[float]): List of standard deviation values for normalization.
        module_name (str): Name of the module used for training.

    Returns:
        SegmentationLabeledSatelliteImage or list[SegmentationLabeledSatelliteImage]: The labeled satellite image with the predicted mask.

    Raises:
        ValueError: If the dimension of the image is not divisible by the tile size used during training or if the dimension is smaller than the tile size.

    """

    def predict_single_image(image):
        # Retrieve satellite image
        si = get_satellite_image(image, n_bands)

        if si.array.shape[1] == tiles_size:
            lsi = make_prediction(
                model=model,
                image=si,
                tiles_size=tiles_size,
                augment_size=augment_size,
                n_bands=n_bands,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                module_name=module_name,
            )

        elif si.array.shape[1] > tiles_size:
            if si.array.shape[1] % tiles_size != 0:
                raise ValueError(
                    "The dimension of the image must be divisible by the tiles size used during training."
                )
            else:
                si_splitted = si.split(tiles_size)

                lsi_splitted = [
                    make_prediction(
                        s_si,
                        model,
                        tiles_size,
                        augment_size,
                        n_bands,
                        normalization_mean,
                        normalization_std,
                        module_name,
                    )
                    for s_si in tqdm(si_splitted)
                ]

                lsi = make_mosaic(lsi_splitted, [i for i in range(n_bands)])
        else:
            raise ValueError(
                "The dimension of the image should be equal to or greater than the tile size used during training."
            )
        return lsi

    # Check if input is a str
    if isinstance(images, str):
        return predict_single_image(images)
    else:
        return [
            predict(
                image,
                model,
                tiles_size,
                augment_size,
                n_bands,
                normalization_mean,
                normalization_std,
                module_name,
            )
            for image in tqdm(images)
        ]


def get_filename_to_polygons(dep: str, year: int, fs: S3FileSystem) -> gpd.GeoDataFrame:
    """
    Retrieves the filename to polygons mapping for a given department and year.

    Args:
        dep (str): The department code.
        year (int): The year.
        fs (S3FileSystem): The S3FileSystem object for accessing the data.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filename to polygons mapping.

    """
    # Load the filename to polygons mapping
    data = (
        pq.ParquetDataset(
            "projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/",
            filesystem=fs,
            filters=[("dep", "=", dep), ("year", "=", year)],
        )
        .read()
        .to_pandas()
    )

    # Convert the geometry column to a GeoSeries
    data["geometry"] = gpd.GeoSeries.from_wkt(data["geometry"])
    return gpd.GeoDataFrame(data, geometry="geometry", crs=data.CRS.unique()[0])


def get_cache_path(image: str) -> str:
    """
    Get the cache path for the image.

    Args:
        image (str): The image path.

    Returns:
        str: The cache path.
    """
    assert (
        "MLFLOW_MODEL_NAME" in os.environ
    ), "Please set the MLFLOW_MODEL_NAME environment variable."
    assert (
        "MLFLOW_MODEL_VERSION" in os.environ
    ), "Please set the MLFLOW_MODEL_VERSION environment variable."

    cache_path = os.path.dirname(image.replace(image.split("/")[1], "cache-predictions"))
    image_name = os.path.splitext(os.path.basename(image))[0]
    return f"{cache_path}/{os.getenv("MLFLOW_MODEL_NAME")}/{os.getenv("MLFLOW_MODEL_VERSION")}/{image_name}.npy"


def load_from_cache(
    image: str, n_bands: list[int], filesystem: S3FileSystem
) -> SegmentationLabeledSatelliteImage:
    """
    Load the image and mask from the cache.

    Args:
        image (str): The image path.
        n_bands (list[int]): The number of bands.
        filesystem (s3fs.S3FileSystem): The file system.

    Returns:
        SegmentationLabeledSatelliteImage: The labeled satellite image with the predicted mask.
    """

    mask_path = get_cache_path(image)

    si = get_satellite_image(image, n_bands)
    with filesystem.open(mask_path, "rb") as f:
        mask = np.load(f)

    logits = True if len(mask.shape) >= 3 else False
    return SegmentationLabeledSatelliteImage(si, mask, logits=logits)
