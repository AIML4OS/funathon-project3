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
        token=os.environ["AWS_SESSION_TOKEN"]
    )


def download_label(format_ext, filename, common_params, export_url):
    params = common_params.copy()
    params["format"] = format_ext

    response = requests.get(export_url, params=params, stream=True)

    if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/"):
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
        "bboxSR": "3035",   # Lambert-93
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
