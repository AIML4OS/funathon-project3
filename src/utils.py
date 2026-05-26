"""
Shared utilities for the satellite-segmentation pipeline.
"""

import os
import numpy as np
from PIL import Image
import requests
from s3fs import S3FileSystem


def get_file_system() -> S3FileSystem:
    """
    Return the configured S3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token="",
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


def tiff_to_numpy(format_ext):
    img = Image.open(format_ext)
    img_array = np.array(img)
    img_array[(img_array == 254) | (img_array == 255)] = 0

    npy_format_ext = format_ext.replace(".tif", ".npy")
    np.save(npy_format_ext, img_array)
    os.remove(format_ext)

    return img_array
