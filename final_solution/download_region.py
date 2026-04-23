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
#   "tqdm>=4.66",
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
4. For each product: download all 13 JP2 band files from CDSE EO S3 into
   memory via s3fs, open them as rasterio datasets, then window-read each
   patch (resampling all bands to 10 m on the fly).
   (B10/cirrus is excluded — it is not delivered in L2A products.)
5. Download the matching CLC+ label from the Copernicus ImageServer API.
6. Upload images (13-band GeoTIFF) and labels (.npy) to personal S3.
7. Write a filename2bbox.parquet index to personal S3.

Usage
-----
    uv run final_solution/download_region.py \\
        --nuts FR101 \\
        --year 2021 \\
        --s3-bucket my-bucket \\
        [--s3-prefix sentinel2] \\
        [--label-year 2021] \\
        [--eo-s3-key KEY] \\
        [--eo-s3-secret SECRET]

    # Use 2024 Sentinel-2 imagery with 2021 CLC+ labels (no 2024 edition exists):
    uv run final_solution/download_region.py \\
        --nuts FR101 \\
        --year 2024 \\
        --label-year 2021 \\
        --s3-bucket my-bucket

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

import numpy as np
import pandas as pd
import requests
import rasterio
import s3fs
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box, shape
from tqdm.auto import tqdm

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
            if box(x, y, x + patch_size_m, y + patch_size_m).intersects(nuts_geom):
                patches.append({
                    "xmin": x, "ymin": y,
                    "xmax": x + patch_size_m, "ymax": y + patch_size_m,
                })
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


# ---------------------------------------------------------------------------
# Product discovery (OData API)
# ---------------------------------------------------------------------------


def find_s2_products(wgs84_bbox: list[float], year: int, max_cloud: int = 30) -> list[dict]:
    """
    Query OData for Sentinel-2 L2A products intersecting the WGS84 bbox
    with cloud cover ≤ max_cloud during May–September of year.
    Results are sorted by cloud cover ascending in Python (OData does not
    support $orderby on embedded attribute fields).
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
        f"and ContentDate/Start le {year}-09-30T23:59:59.999Z "
        f"and Online eq true"
    )
    resp = requests.get(
        ODATA_URL,
        params={"$filter": odata_filter, "$top": 200},
        timeout=60,
    )
    resp.raise_for_status()
    results = [
        {
            "Name": item["Name"],
            "cloudCover": item.get("cloudCover", 0),
            "GeoFootprint": item.get("GeoFootprint", {}),
            "S3Path": item.get("S3Path", ""),
        }
        for item in resp.json().get("value", [])
    ]
    return sorted(results, key=lambda p: p["cloudCover"])


# ---------------------------------------------------------------------------
# EO S3 helpers
# ---------------------------------------------------------------------------


def build_eo_fs(key: str, secret: str) -> s3fs.S3FileSystem:
    """S3FileSystem for the CDSE EO data bucket."""
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": EO_S3_ENDPOINT},
        key=key,
        secret=secret,
    )


def product_s3_prefix(product_name: str, s3path: str = "") -> str:
    """
    Return the EO S3 directory prefix for a product (always ends with .SAFE/).

    Uses the OData S3Path when available — reliable for reprocessed products
    stored under L2A_N0500/… instead of L2A/….  The S3Path may point deeper
    than .SAFE (e.g. it sometimes ends with /GRANULE), so we truncate it at
    the .SAFE boundary.  Falls back to constructing the path from the product
    name's sensing date when S3Path is absent.
    """
    if s3path:
        # OData is the source of truth: reprocessed products (e.g. N0500) live
        # under prefixes that cannot be derived from the product name alone.
        path = s3path.lstrip("/")
        safe_end = path.find(".SAFE")
        if safe_end >= 0:
            path = path[: safe_end + len(".SAFE")]
        return path + "/"
    # Fallback for the rare case where OData does not return S3Path: assume
    # the standard layout organised by sensing date.
    date_compact = product_name.split("_")[2]   # e.g. "20210615T102021"
    yyyy, mm, dd = date_compact[:4], date_compact[4:6], date_compact[6:8]
    return f"eodata/Sentinel-2/MSI/L2A/{yyyy}/{mm}/{dd}/{product_name}/"


def find_band_paths(eo_fs: s3fs.S3FileSystem, s3_prefix: str) -> dict[str, str]:
    """
    Navigate the .SAFE directory tree and return s3fs-compatible paths for
    all 14 bands (relative to the EO S3 endpoint, including the bucket name).

    Lists each resolution sub-directory rather than constructing filenames
    from the granule timestamp, because newer processing baselines (N0500+)
    embed a different timestamp in JP2 filenames than in the granule directory name.
    """
    # Why glob and not ls? On the CDSE MinIO backend, `ls("SAFE/GRANULE/")`
    # sometimes returns the parent prefix itself as an entry (trailing slash
    # and all), which would then be parsed as a spurious granule_id called
    # "GRANULE" and produce a double-GRANULE path downstream. Real granule
    # directories always start with "L2A_", so globbing on that prefix is a
    # robust filter.
    granule_matches = eo_fs.glob(f"{s3_prefix}GRANULE/L2A_*")
    if not granule_matches:
        raise RuntimeError(f"No L2A_* granule directory under {s3_prefix}GRANULE/")
    img_base = f"{granule_matches[0].rstrip('/')}/IMG_DATA"

    res_dirs = {res_dir for res_dir, _ in BAND_MAP.values()}
    dir_listings: dict[str, list[str]] = {
        res_dir: eo_fs.ls(f"{img_base}/{res_dir}/", detail=False)
        for res_dir in res_dirs
    }

    band_paths: dict[str, str] = {}
    for band_name, (res_dir, suffix) in BAND_MAP.items():
        matches = [f for f in dir_listings[res_dir] if f.endswith(f"_{suffix}.jp2")]
        if not matches:
            raise RuntimeError(f"No JP2 file for band {band_name} in {img_base}/{res_dir}/")
        band_paths[band_name] = matches[0]
    return band_paths


# ---------------------------------------------------------------------------
# Band file loading and patch extraction
# ---------------------------------------------------------------------------


def open_band_readers(
    eo_fs: s3fs.S3FileSystem, band_paths: dict[str, str]
) -> tuple[list[MemoryFile], dict[str, rasterio.DatasetReader]]:
    """
    Download all 13 JP2 band files from EO S3 into memory via s3fs and open
    them as rasterio dataset readers.

    Keeping the readers open for the duration of a product's patch loop avoids
    repeated MemoryFile allocations and lets GDAL cache the file structure.
    Returns (mem_files, readers) — callers must close both when done.
    """
    mem_files: list[MemoryFile] = []
    readers: dict[str, rasterio.DatasetReader] = {}
    band_bar = tqdm(
        band_paths.items(), desc="    bands", unit="band",
        leave=False, total=len(band_paths),
    )
    for band_name, s3_path in band_bar:
        band_bar.set_postfix_str(s3_path.split("/")[-1])
        # Eagerly stream the whole JP2 (~50 MB) into RAM. We pay this cost
        # once per product, then read hundreds of cheap windows from the
        # in-memory reader — far cheaper than per-patch HTTP round-trips.
        data = eo_fs.cat(s3_path)
        mf = MemoryFile(data)
        mem_files.append(mf)
        readers[band_name] = mf.open()
    return mem_files, readers


def close_band_readers(
    mem_files: list[MemoryFile],
    readers: dict[str, rasterio.DatasetReader],
) -> None:
    for r in readers.values():
        r.close()
    for mf in mem_files:
        mf.close()


def read_patch(
    readers: dict[str, rasterio.DatasetReader], bbox_3035: list
) -> np.ndarray:
    """
    Window-read a (13, H, W) uint16 patch from already-open band readers.
    All bands are resampled to 10 m by specifying out_shape.
    """
    xmin, ymin, xmax, ymax = bbox_3035
    h = int((ymax - ymin) / 10)
    w = int((xmax - xmin) / 10)

    # CRITICAL: tile bboxes are stored in EPSG:3035 (metres, ~3 500 000 range)
    # but Sentinel-2 JP2s are in UTM (EPSG:326xx, ~200 000–800 000 range).
    # Passing 3035 coords to window_from_bounds with a UTM transform would
    # compute a window entirely outside the image → every pixel comes back as
    # fill_value=0. Reproject once here (all bands share the same UTM zone).
    any_src = next(iter(readers.values()))
    left, bottom, right, top = transform_bounds(
        "EPSG:3035", any_src.crs, xmin, ymin, xmax, ymax
    )

    bands_list: list[np.ndarray] = []
    for band_name in BAND_MAP:
        src = readers[band_name]
        window = window_from_bounds(left, bottom, right, top, src.transform)
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
    """Encode a (C, H, W) uint16 array as a georeferenced GeoTIFF in EPSG:3035."""
    c, h, w = array.shape
    xmin, ymin, xmax, ymax = bbox_3035
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": w,
        "height": h,
        "count": c,
        "crs": "EPSG:3035",
        "transform": from_bounds(xmin, ymin, xmax, ymax, w, h),
        "compress": "deflate",
    }
    # rasterio writes into GDAL's /vsimem/ buffer via the outer MemoryFile.
    # The inner context flushes the dataset on close; memfile.read() must be
    # called *after* the inner block and *inside* the outer one to recover
    # the encoded GeoTIFF bytes. (MemoryFile(some_bytesio) would NOT write
    # into that BytesIO — a common gotcha worth remembering.)
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(array)
        return memfile.read()


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
        help="Sentinel-2 acquisition year (default: 2021)",
    )
    parser.add_argument(
        "--label-year", type=int, default=None,
        help=(
            "CLC+ Backbone label year (default: same as --year). "
            "Available editions: 2018, 2021. Use this when --year has no "
            "matching CLC+ edition (e.g. --year 2024 --label-year 2021)."
        ),
    )
    parser.add_argument("--s3-bucket", required=True, help="Destination S3 bucket name")
    parser.add_argument(
        "--s3-prefix", default="sentinel2",
        help="Key prefix inside the bucket (default: sentinel2)",
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

    label_year = args.label_year or args.year
    if label_year != args.year:
        print(f"Note: using CLC+ labels from {label_year} for {args.year} imagery.\n")

    prefix = f"{args.s3_bucket}/{args.s3_prefix}"

    # ---- Step 1: Build tile grid from NUTS3 boundary -----------------------
    # Fetch the EPSG:3035 polygon of the NUTS3 region from Eurostat GISCO and
    # cover it with a regular 2 500 m × 2 500 m grid (→ 250 px at 10 m). The
    # filename of every tile (xmin_ymin_seq.tif) is deterministic, which is
    # what makes the resumability check in step 4 cheap and correct.
    nuts_geom = load_nuts3_boundary(args.nuts)
    tiles = build_tile_grid(nuts_geom)
    print(f"→ {len(tiles)} patches in the {args.nuts} grid\n")

    # ---- Step 2: Find covering Sentinel-2 products -------------------------
    # Query CDSE OData for L2A products intersecting the region's WGS84
    # bounding box over the May–September window of `year` (peak vegetation
    # season, lowest cloud risk). OData lets us combine several filters in
    # one HTTP request: product type, footprint intersection, cloud cover
    # threshold, date range, and online-availability — so the result list
    # only contains products we can actually download.
    all_bboxes = list(tiles["bbox"])
    nuts_bbox_wgs84 = bbox_3035_to_wgs84([
        min(b[0] for b in all_bboxes), min(b[1] for b in all_bboxes),
        max(b[2] for b in all_bboxes), max(b[3] for b in all_bboxes),
    ])
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
    # Two independent S3 clients with different endpoints and different
    # credentials: one read-only for CDSE EO S3, one read/write for the
    # personal SSP Cloud bucket. Using two dedicated S3FileSystem instances
    # (rather than reconfiguring a single one) avoids any credential-set
    # cross-talk during the product loop.
    personal_fs = build_personal_fs()
    eo_fs = build_eo_fs(args.eo_s3_key, args.eo_s3_secret)

    uploaded = skipped = errors = 0

    # ---- Step 4: Process each product --------------------------------------
    # Product-first, tile-second ordering: each product's 13 JP2 bands are
    # shared by hundreds of patches, so we download them *once* per product
    # (`open_band_readers`), then loop over every covered tile reading cheap
    # in-memory windows. Tiles already on personal S3 are skipped here (the
    # resumability mechanism — see the `personal_fs.exists(...)` check below).
    product_bar = tqdm(products, desc="Products", unit="prod")

    def _refresh_totals() -> None:
        product_bar.set_postfix(up=uploaded, skip=skipped, err=errors)

    _refresh_totals()

    for product in product_bar:
        product_name = product["Name"]
        geo_footprint = product.get("GeoFootprint", {})
        product_bar.set_description(f"Products [{product_name[:40]}]")

        # Filter tiles covered by this product's footprint
        if geo_footprint:
            try:
                product_geom = shape(geo_footprint)
                mask = [
                    box(*bbox_3035_to_wgs84(row["bbox"])).intersects(product_geom)
                    for _, row in tiles.iterrows()
                ]
                covered = tiles[mask]
            except Exception:
                covered = tiles
        else:
            covered = tiles

        # Resumability check: a tile is considered "done" iff BOTH the image
        # and the label already exist on personal S3. This lets the script be
        # stopped and restarted freely. Caveat: a present-but-corrupt file
        # (e.g. a tile generated before a bug fix) is treated as done — delete
        # it explicitly to force regeneration.
        to_process = covered[[
            not (
                personal_fs.exists(
                    f"{prefix}/images/{args.nuts}/{args.year}/{row['filename']}"
                )
                and personal_fs.exists(
                    f"{prefix}/labels/{args.nuts}/{args.year}/"
                    f"{row['filename'].replace('.tif', '.npy')}"
                )
            )
            for _, row in covered.iterrows()
        ]]

        if to_process.empty:
            tqdm.write(f"  [skip] {product_name[:60]}… — all tiles already uploaded")
            skipped += len(covered)
            _refresh_totals()
            continue

        tqdm.write(
            f"  Product {product_name[:60]}…  "
            f"cloud={product['cloudCover']:.1f}%  →  {len(to_process)} tiles"
        )

        try:
            band_paths = find_band_paths(
                eo_fs, product_s3_prefix(product_name, product.get("S3Path", ""))
            )
        except Exception as exc:
            tqdm.write(f"  ✗  Could not resolve band paths: {exc}")
            errors += len(to_process)
            _refresh_totals()
            continue

        try:
            mem_files, readers = open_band_readers(eo_fs, band_paths)
        except Exception as exc:
            tqdm.write(f"  ✗  Band download failed: {exc}")
            errors += len(to_process)
            _refresh_totals()
            continue

        try:
            tile_bar = tqdm(
                to_process.iterrows(), desc="    tiles", unit="tile",
                total=len(to_process), leave=False,
            )
            for _, row in tile_bar:
                filename = row["filename"]
                bbox = row["bbox"]
                patch_id = filename.replace(".tif", "")
                img_path = f"{prefix}/images/{args.nuts}/{args.year}/{filename}"
                lbl_path = (
                    f"{prefix}/labels/{args.nuts}/{args.year}/"
                    f"{filename.replace('.tif', '.npy')}"
                )

                try:
                    array = read_patch(readers, bbox)
                    tiff_bytes = patch_array_to_tiff_bytes(array, bbox)
                    with personal_fs.open(img_path, "wb") as f:
                        f.write(tiff_bytes)

                    label = download_label_array(bbox, label_year)
                    buf = io.BytesIO()
                    np.save(buf, label)
                    buf.seek(0)
                    with personal_fs.open(lbl_path, "wb") as f:
                        f.write(buf.read())

                    uploaded += 1

                except Exception as exc:
                    errors += 1
                    tqdm.write(f"    ✗  {patch_id}  —  {exc}")

                _refresh_totals()

        finally:
            close_band_readers(mem_files, readers)

    product_bar.close()

    # ---- Step 5: Write filename2bbox.parquet index -------------------------
    # The index is computed entirely from the deterministic grid (step 1),
    # not from whatever subset of tiles was actually uploaded this run — so
    # re-running the script later (even after a partial interruption) still
    # produces an index that lists every patch in the region. Downstream
    # code uses this parquet to map a tile filename to its EPSG:3035 bbox
    # without needing to parse filenames.
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
