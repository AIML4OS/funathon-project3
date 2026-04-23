#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "numpy>=1.26",
#   "rasterio>=1.4",
#   "pandas>=2.0",
#   "pyarrow>=15",
#   "s3fs>=2024.2",
#   "shapely>=2.0",
# ]
# ///
"""
Build a Sentinel-2 / CLC+ training dataset for a NUTS3 region and upload it
to an S3-compatible object store.

Workflow
--------
1. Load the NUTS3 region boundary from the Eurostat GISCO API.
2. Tile the region with a regular 2 500 m × 2 500 m grid (250 px at 10 m).
3. Query the CDSE OData catalogue to find cloud-free Sentinel-2 L2A products.
4. For each product, window-read 14 JP2 band files from CDSE EO S3 via GDAL
   /vsis3/, resampling all bands to 10 m on the fly.
5. Download the matching CLC+ label from the Copernicus ImageServer API.
6. Upload images (14-band GeoTIFF) and labels (.npy) to personal S3.
7. Write a filename2bbox.parquet index to personal S3.

Usage
-----
    uv run final_solution/download_region.py \\
        --nuts FR101 \\
        --year 2021 \\
        --s3-bucket my-bucket \\
        [--s3-prefix sentinel2] \\
        [--workers 4] \\
        [--eo-s3-key KEY] \\
        [--eo-s3-secret SECRET]

Required environment variables (personal S3 credentials):
    AWS_S3_ENDPOINT
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN   (optional)

EO S3 credentials (from https://eodata-s3keysmanager.dataspace.copernicus.eu/):
    EO_S3_ACCESS_KEY_ID      (or --eo-s3-key)
    EO_S3_SECRET_ACCESS_KEY  (or --eo-s3-secret)
"""

import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import rasterio
import s3fs
from rasterio.env import Env as RasterioEnv
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box, shape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUTS_GEOJSON_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "nuts/geojson/NUTS_RG_01M_2021_3035_LEVL_3.geojson"
)
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
IMAGESERVER_URL = (
    "https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/CLC_plus/"
    "CLMS_CLCplus_RASTER_{year}_010m_eu/ImageServer/exportImage"
)
EO_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

PATCH_SIZE_M = 2500   # patch side length in metres
PATCH_SIZE_PX = 250   # patch side length in pixels (10 m resolution)

# Maps band name → (resolution folder, JP2 file suffix)
BAND_MAP: dict[str, tuple[str, str]] = {
    "B01": ("R60m", "B01_60m"),
    "B02": ("R10m", "B02_10m"),
    "B03": ("R10m", "B03_10m"),
    "B04": ("R10m", "B04_10m"),
    "B05": ("R20m", "B05_20m"),
    "B06": ("R20m", "B06_20m"),
    "B07": ("R20m", "B07_20m"),
    "B08": ("R10m", "B08_10m"),
    "B8A": ("R20m", "B8A_20m"),
    "B09": ("R60m", "B09_60m"),
    "B10": ("R60m", "B10_60m"),
    "B11": ("R20m", "B11_20m"),
    "B12": ("R20m", "B12_20m"),
    "SCL": ("R20m", "SCL_20m"),
}

# ---------------------------------------------------------------------------
# NUTS3 region helpers
# ---------------------------------------------------------------------------


def load_nuts3_boundary(nuts_code: str):
    """
    Return the EPSG:3035 shapely geometry for a NUTS3 region.
    The Eurostat GISCO file NUTS_RG_01M_2021_3035_LEVL_3.geojson stores
    coordinates directly in EPSG:3035 (metres).
    """
    print(f"Loading NUTS3 boundary for {nuts_code} …")
    resp = requests.get(NUTS_GEOJSON_URL, timeout=120)
    resp.raise_for_status()
    for feat in resp.json()["features"]:
        if feat["properties"]["NUTS_ID"] == nuts_code:
            return shape(feat["geometry"])
    raise ValueError(f"NUTS3 code '{nuts_code}' not found in Eurostat boundaries.")


def build_tile_grid(nuts_geom, patch_size_m: int = PATCH_SIZE_M) -> pd.DataFrame:
    """
    Create a regular grid of patch_size_m × patch_size_m patches that
    intersect the NUTS3 geometry (EPSG:3035).

    Each row in the returned DataFrame has:
        filename  — "{xmin}_{ymin}_{seq}.tif"
        bbox      — [xmin, ymin, xmax, ymax]
    """
    minx, miny, maxx, maxy = nuts_geom.bounds

    # Snap grid origin to a multiple of patch_size_m
    origin_x = int(minx // patch_size_m) * patch_size_m
    origin_y = int(miny // patch_size_m) * patch_size_m

    patches = []
    x = origin_x
    while x < maxx:
        y = origin_y
        while y < maxy:
            patch_box = box(x, y, x + patch_size_m, y + patch_size_m)
            if patch_box.intersects(nuts_geom):
                patches.append({"xmin": x, "ymin": y,
                                 "xmax": x + patch_size_m, "ymax": y + patch_size_m})
            y += patch_size_m
        x += patch_size_m

    for seq, p in enumerate(patches):
        p["filename"] = f"{p['xmin']}_{p['ymin']}_{seq}.tif"
        p["bbox"] = [p["xmin"], p["ymin"], p["xmax"], p["ymax"]]

    return pd.DataFrame(patches, columns=["filename", "bbox"])


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def bbox_3035_to_wgs84(bbox: list) -> list[float]:
    """Convert an EPSG:3035 bounding box to [west, south, east, north] WGS84."""
    left, bottom, right, top = transform_bounds(
        "EPSG:3035", "EPSG:4326", bbox[0], bbox[1], bbox[2], bbox[3]
    )
    return [left, bottom, right, top]


def union_bbox_3035(bboxes: list[list]) -> list:
    arr = np.array(bboxes)
    return [arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()]


# ---------------------------------------------------------------------------
# Product discovery (OData API)
# ---------------------------------------------------------------------------


def find_s2_products(wgs84_bbox: list[float], year: int, max_cloud: int = 30) -> list[dict]:
    """
    Query OData for Sentinel-2 L2A products that intersect the WGS84 bbox
    and have cloud cover ≤ max_cloud during May–September of year.
    """
    w, s, e, n = wgs84_bbox
    polygon_wkt = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    odata_filter = (
        f"Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}') "
        f"and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud}.0) "
        f"and ContentDate/Start ge {year}-05-01T00:00:00.000Z "
        f"and ContentDate/Start le {year}-09-30T23:59:59.999Z"
    )
    resp = requests.get(
        ODATA_URL,
        params={"$filter": odata_filter, "$orderby": "cloudCover asc", "$top": 200},
        timeout=60,
    )
    resp.raise_for_status()
    return [
        {
            "Name": item["Name"],
            "cloudCover": item.get("cloudCover", 0),
            "GeoFootprint": item.get("GeoFootprint", {}),
        }
        for item in resp.json().get("value", [])
    ]


# ---------------------------------------------------------------------------
# EO S3 helpers
# ---------------------------------------------------------------------------


def build_eo_fs(key: str, secret: str) -> s3fs.S3FileSystem:
    """S3FileSystem for the CDSE EO data bucket (used for directory listing)."""
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": EO_S3_ENDPOINT},
        key=key,
        secret=secret,
    )


def product_s3_prefix(product_name: str) -> str:
    """
    Derive the EO S3 directory prefix from a product name.

    Example input:
        S2A_MSIL2A_20210615T102021_N0300_R065_T32TLT_20210615T135501.SAFE
    Returns:
        eodata/Sentinel-2/MSI/L2A/2021/06/15/S2A_MSIL2A_.../
    """
    date_compact = product_name.split("_")[2]   # e.g. "20210615T102021"
    yyyy, mm, dd = date_compact[:4], date_compact[4:6], date_compact[6:8]
    return f"eodata/Sentinel-2/MSI/L2A/{yyyy}/{mm}/{dd}/{product_name}/"


def find_band_paths(eo_fs: s3fs.S3FileSystem, s3_prefix: str) -> dict[str, str]:
    """
    Navigate the .SAFE directory tree and return /vsis3/ paths for all 14 bands.

    Granule directory name structure: L2A_{tile_id}_{orbit}_{date_compact}
    JP2 filenames:  {tile_id}_{date_compact}_{band_suffix}.jp2
    """
    granule_entries = eo_fs.ls(s3_prefix + "GRANULE/", detail=False)
    if not granule_entries:
        raise RuntimeError(f"No granule found under {s3_prefix}GRANULE/")
    granule_id = granule_entries[0].rstrip("/").split("/")[-1]

    parts = granule_id.split("_")   # ['L2A', 'T32TLT', 'A031234', '20210615T102021']
    tile_id = parts[1]              # 'T32TLT'
    date_compact = parts[3]         # '20210615T102021'

    img_base = f"{s3_prefix}GRANULE/{granule_id}/IMG_DATA"
    band_paths: dict[str, str] = {}
    for band_name, (res_dir, suffix) in BAND_MAP.items():
        jp2_name = f"{tile_id}_{date_compact}_{suffix}.jp2"
        band_paths[band_name] = f"/vsis3/eodata/{img_base}/{res_dir}/{jp2_name}"
    return band_paths


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_patch(
    band_paths: dict[str, str],
    bbox_3035: list,
    eo_s3_key: str,
    eo_s3_secret: str,
) -> np.ndarray:
    """
    Window-read a (14, H, W) uint16 patch from the EO S3 JP2 band files.

    Uses rasterio.env.Env to inject EO S3 credentials into GDAL without
    touching the system env vars used by s3fs for the personal bucket.
    All bands are resampled to 10 m via out_shape.
    """
    xmin, ymin, xmax, ymax = bbox_3035
    h = int((ymax - ymin) / 10)
    w = int((xmax - xmin) / 10)

    env_vars = {
        "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
        "AWS_ACCESS_KEY_ID": eo_s3_key,
        "AWS_SECRET_ACCESS_KEY": eo_s3_secret,
        "AWS_HTTPS": "YES",
        "AWS_VIRTUAL_HOSTING": "FALSE",
    }

    bands_list: list[np.ndarray] = []
    with RasterioEnv(**env_vars):
        for band_name in BAND_MAP:
            with rasterio.open(band_paths[band_name]) as src:
                window = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
                data = src.read(
                    1,
                    window=window,
                    out_shape=(h, w),
                    resampling=rasterio.enums.Resampling.nearest,
                    boundless=True,
                    fill_value=0,
                )
            bands_list.append(data)

    return np.stack(bands_list, axis=0).astype(np.uint16)


def patch_array_to_tiff_bytes(array: np.ndarray, bbox_3035: list) -> bytes:
    """Encode a (14, H, W) uint16 array as a georeferenced GeoTIFF in EPSG:3035."""
    _, h, w = array.shape
    xmin, ymin, xmax, ymax = bbox_3035
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": w,
        "height": h,
        "count": 14,
        "crs": "EPSG:3035",
        "transform": from_bounds(xmin, ymin, xmax, ymax, w, h),
        "compress": "deflate",
    }
    buf = io.BytesIO()
    with MemoryFile(buf) as memfile:
        with memfile.open(**profile) as dst:
            dst.write(array)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# CLC+ label download
# ---------------------------------------------------------------------------


def download_label_array(bbox: list, year: int) -> np.ndarray:
    """
    Download a CLC+ Backbone label from the Copernicus ImageServer for the
    given EPSG:3035 bounding box.  Returns a (H, W) uint8 array with
    nodata values 254/255 mapped to 0.
    """
    xmin, ymin, xmax, ymax = bbox
    size_x = int((xmax - xmin) / 10)
    size_y = int((ymax - ymin) / 10)

    resp = requests.get(
        IMAGESERVER_URL.format(year=year),
        params={
            "f": "image",
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "bboxSR": "3035",
            "imageSR": "3035",
            "size": f"{size_x},{size_y}",
            "format": "tiff",
        },
        timeout=60,
    )
    resp.raise_for_status()

    with MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            label = src.read(1)

    label[(label == 254) | (label == 255)] = 0
    return label


# ---------------------------------------------------------------------------
# Personal S3 helpers
# ---------------------------------------------------------------------------


def build_personal_fs() -> s3fs.S3FileSystem:
    """S3FileSystem for the personal bucket using SSP Cloud env credentials."""
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ.get("AWS_SESSION_TOKEN", ""),
    )


# ---------------------------------------------------------------------------
# Per-tile task
# ---------------------------------------------------------------------------


def process_tile(
    personal_fs: s3fs.S3FileSystem,
    band_paths: dict[str, str],
    year: int,
    filename: str,
    bbox: list,
    img_s3_path: str,
    lbl_s3_path: str,
    eo_s3_key: str,
    eo_s3_secret: str,
) -> tuple[str, bool, str]:
    """
    Extract one Sentinel-2 patch from EO S3 and its CLC+ label, upload both
    to the personal S3 bucket.  Already-uploaded tiles are skipped.
    """
    patch_id = filename.replace(".tif", "")

    if personal_fs.exists(img_s3_path) and personal_fs.exists(lbl_s3_path):
        return patch_id, True, "skipped"

    try:
        array = extract_patch(band_paths, bbox, eo_s3_key, eo_s3_secret)
        tiff_bytes = patch_array_to_tiff_bytes(array, bbox)
        with personal_fs.open(img_s3_path, "wb") as f:
            f.write(tiff_bytes)

        label = download_label_array(bbox, year)
        buf = io.BytesIO()
        np.save(buf, label)
        buf.seek(0)
        with personal_fs.open(lbl_s3_path, "wb") as f:
            f.write(buf.read())

        return patch_id, True, "uploaded"

    except Exception as exc:
        return patch_id, False, str(exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Sentinel-2 / CLC+ dataset for a NUTS3 region "
            "from CDSE EO S3 and upload to a personal S3 bucket."
        )
    )
    parser.add_argument("--nuts", required=True, help="NUTS3 region code, e.g. FR101")
    parser.add_argument(
        "--year", type=int, default=2021,
        help="Acquisition year — must match an available CLC+ edition (default: 2021)",
    )
    parser.add_argument("--s3-bucket", required=True, help="Destination S3 bucket name")
    parser.add_argument(
        "--s3-prefix", default="sentinel2",
        help="Key prefix inside the bucket (default: sentinel2)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--eo-s3-key",
        default=os.environ.get("EO_S3_ACCESS_KEY_ID"),
        help="EO S3 access key (or set EO_S3_ACCESS_KEY_ID)",
    )
    parser.add_argument(
        "--eo-s3-secret",
        default=os.environ.get("EO_S3_SECRET_ACCESS_KEY"),
        help="EO S3 secret key (or set EO_S3_SECRET_ACCESS_KEY)",
    )
    args = parser.parse_args()

    if not args.eo_s3_key or not args.eo_s3_secret:
        parser.error(
            "EO S3 credentials required — provide --eo-s3-key / --eo-s3-secret "
            "or set EO_S3_ACCESS_KEY_ID / EO_S3_SECRET_ACCESS_KEY. "
            "Generate credentials at https://eodata-s3keysmanager.dataspace.copernicus.eu/"
        )

    prefix = f"{args.s3_bucket}/{args.s3_prefix}"

    # ---- Step 1: Build the tile grid from the NUTS3 boundary ---------------
    nuts_geom = load_nuts3_boundary(args.nuts)
    tiles = build_tile_grid(nuts_geom)
    print(f"→ {len(tiles)} patches in the {args.nuts} grid\n")

    # ---- Step 2: Find covering Sentinel-2 products -------------------------
    all_bboxes = list(tiles["bbox"])
    nuts_bbox_wgs84 = bbox_3035_to_wgs84(
        [min(b[0] for b in all_bboxes), min(b[1] for b in all_bboxes),
         max(b[2] for b in all_bboxes), max(b[3] for b in all_bboxes)]
    )
    print(
        f"Region WGS84 bbox: W={nuts_bbox_wgs84[0]:.4f}, S={nuts_bbox_wgs84[1]:.4f}, "
        f"E={nuts_bbox_wgs84[2]:.4f}, N={nuts_bbox_wgs84[3]:.4f}"
    )
    print(f"Querying OData for S2 L2A products (year={args.year}, cloud ≤ 30 %) …")
    products = find_s2_products(nuts_bbox_wgs84, args.year)
    print(f"→ {len(products)} products found\n")

    if not products:
        print("No Sentinel-2 products found — nothing to download.")
        return

    # ---- Step 3: Build filesystems -----------------------------------------
    personal_fs = build_personal_fs()
    eo_fs = build_eo_fs(args.eo_s3_key, args.eo_s3_secret)

    uploaded = skipped = errors = 0

    # ---- Step 4: Process each product --------------------------------------
    for product in products:
        product_name = product["Name"]
        geo_footprint = product.get("GeoFootprint", {})

        # Determine which tiles fall within this product's footprint
        if geo_footprint:
            try:
                product_geom = shape(geo_footprint)
                covered_mask = [
                    box(*bbox_3035_to_wgs84(row["bbox"])).intersects(product_geom)
                    for _, row in tiles.iterrows()
                ]
                covered_tiles = tiles[covered_mask]
            except Exception:
                covered_tiles = tiles  # fallback: try all tiles
        else:
            covered_tiles = tiles

        to_process = covered_tiles[
            [
                not (
                    personal_fs.exists(
                        f"{prefix}/images/{args.nuts}/{args.year}/{row['filename']}"
                    )
                    and personal_fs.exists(
                        f"{prefix}/labels/{args.nuts}/{args.year}/"
                        f"{row['filename'].replace('.tif', '.npy')}"
                    )
                )
                for _, row in covered_tiles.iterrows()
            ]
        ]

        if to_process.empty:
            print(f"  [skip] {product_name[:60]}… — all tiles already uploaded")
            skipped += len(covered_tiles)
            continue

        print(
            f"  Product {product_name[:60]}…\n"
            f"  cloud={product['cloudCover']:.1f}%  →  {len(to_process)} tiles"
        )

        try:
            band_paths = find_band_paths(eo_fs, product_s3_prefix(product_name))
        except Exception as exc:
            print(f"  ✗  Could not resolve band paths: {exc}")
            errors += len(to_process)
            continue

        tasks = [
            dict(
                personal_fs=personal_fs,
                band_paths=band_paths,
                year=args.year,
                filename=row["filename"],
                bbox=row["bbox"],
                img_s3_path=(
                    f"{prefix}/images/{args.nuts}/{args.year}/{row['filename']}"
                ),
                lbl_s3_path=(
                    f"{prefix}/labels/{args.nuts}/{args.year}/"
                    f"{row['filename'].replace('.tif', '.npy')}"
                ),
                eo_s3_key=args.eo_s3_key,
                eo_s3_secret=args.eo_s3_secret,
            )
            for _, row in to_process.iterrows()
        ]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_tile, **t): t["filename"] for t in tasks}
            for future in as_completed(futures):
                patch_id, success, msg = future.result()
                if not success:
                    errors += 1
                    print(f"    ✗  {patch_id}  —  {msg}")
                elif msg == "skipped":
                    skipped += 1
                else:
                    uploaded += 1
                    print(f"    ✓  {patch_id}")

    # ---- Step 5: Write filename2bbox.parquet index -------------------------
    parquet_path = f"{prefix}/images/{args.nuts}/{args.year}/filename2bbox.parquet"
    parquet_buf = io.BytesIO()
    tiles[["filename", "bbox"]].to_parquet(parquet_buf, index=False)
    parquet_buf.seek(0)
    with personal_fs.open(parquet_path, "wb") as f:
        f.write(parquet_buf.read())
    print(f"\nWrote tile index → s3://{parquet_path}")

    print(
        f"\n{'─' * 50}\n"
        f"Done — {uploaded} uploaded, {skipped} skipped "
        f"(already present), {errors} errors.\n"
        f"S3 destination: s3://{prefix}/\n"
        f"  images → images/{args.nuts}/{args.year}/\n"
        f"  labels → labels/{args.nuts}/{args.year}/\n"
        f"  index  → images/{args.nuts}/{args.year}/filename2bbox.parquet\n"
    )


if __name__ == "__main__":
    main()
