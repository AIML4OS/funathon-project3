"""
Discover and pair (image, label) tiles for training.

`load_data()` resolves the S3 listing of available patches for the requested
NUTS3 / year and applies `filter_indices_from_labels` to drop tiles whose
labels are corrupt or fully no-data (CLC+ `255` sentinel). The pairing is
positional: the i-th image file must match the i-th label file by path
convention — a mismatch here would silently train the model on shifted
labels, which is why the filter step is a hard prerequisite.
"""

import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from src.data.download import download_data
from src.data.filter import filter_indices_from_labels


def get_patchs_labels(
    from_s3: bool,
    nuts: str,
    year: str,
) -> Tuple[List[str], List[str]]:
    """
    Resolve the list of (patch, label) paths for a given NUTS3 / year pair.

    With `from_s3=True`, we only list file names from the bucket's index; the
    actual reads are streamed lazily by callers. With `from_s3=False`, we
    materialise tiles locally under `data/data-preprocessed/` (downloading them
    once via `download_data`) and return their local paths.
    """

    if from_s3:
        url_filenames = (
            f"https://minio.lab.sspcloud.fr/projet-funathon/"
            f"2026/project3/data/images/{nuts}/{year}/filename2bbox.parquet"
        )
        df_filenames = pd.read_parquet(url_filenames)
        patchs = df_filenames.filename.tolist()
        labels = [filename.split('.')[0]+'.npy' for filename in patchs]

    else:
        patchs_path = f"data/data-preprocessed/patchs/{nuts}/{year}"
        labels_path = f"data/data-preprocessed/labels/{nuts}/{year}"

        download_data(patchs_path, labels_path, nuts, year)

        patchs = [
            f"{patchs_path}/{f}"
            for f in os.listdir(patchs_path)
            if Path(f).suffix == ".tif"
        ]

        labels = [
            f"{labels_path}/{f}"
            for f in os.listdir(labels_path)
        ]

    return patchs, labels


def load_data(
    nuts_years: List[str],
) -> Tuple[List[str], List[str]]:

    patches_all = []
    labels_all = []

    for item in nuts_years:
        nuts, year = item.split("_")

        patches, labels = get_patchs_labels(
            from_s3=False,
            nuts=nuts,
            year=year,
        )

        patches.sort()
        labels.sort()

        indices = filter_indices_from_labels(labels, -1.0, 2.0)

        patches_all.extend([patches[i] for i in indices])
        labels_all.extend([labels[i] for i in indices])

    return patches_all, labels_all
