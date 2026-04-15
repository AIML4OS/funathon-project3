# ==========================================================
# data/loading.py
# ==========================================================

import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import subprocess

from data.download import download_data
from data.filter import filter_indices_from_labels


def get_patchs_labels(
    from_s3: bool,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
) -> Tuple[List[str], List[str]]:
    """
    Get paths to patches and labels from S3 or local.
    """

    if from_s3:
        url_filenames = f"https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_3}/{year}/filename2bbox.parquet"
        df_filenames = pd.read_parquet(url_filenames)
        patchs = df_filenames.filename.tolist()
        labels = [filename.split('.')[0]+'.npy' for filename in patchs]

    else:
        patchs_path = (
            f"data/data-preprocessed/patchs/{source}/{dep}/{year}/{tiles_size}"
        )

        labels_path = (
            f"data/data-preprocessed/labels/"
            f"{type_labeler}/{source}/{dep}/{year}/{tiles_size}"
        )

        download_data(
            patchs_path,
            labels_path,
            source,
            dep,
            year,
            tiles_size,
            type_labeler,
        )

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
            source="S2",
            dep=nuts,
            year=year,
            tiles_size="512",
            type_labeler="default",
        )

        patches.sort()
        labels.sort()

        indices = filter_indices_from_labels(labels, -1.0, 2.0)

        patches_all.extend([patches[i] for i in indices])
        labels_all.extend([labels[i] for i in indices])

    return patches_all, labels_all


def format_datasets(args_dict: dict) -> Tuple[List[str], List[str], dict]:
    """
    Validate dataset paths on S3 and extract NUTS + years.
    """

    nuts, years = zip(*[item.split("_") for item in args_dict["datasets"]])
    nuts = [n.upper() for n in nuts]

    for nut, year in zip(nuts, years):
        alias_cmd = [
            "mc", "alias", "set", "public",
            "https://minio.lab.sspcloud.fr",
            "", ""
        ]

        with open("/dev/null", "w") as devnull:
            # set public alias
            subprocess.run(alias_cmd, check=True, stdout=devnull, stderr=devnull)
            patch_cmd = [
                "mc",
                "stat",
                f"public/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts}/{years}/",
            ]
            subprocess.run(patch_cmd, check=True, stdout=devnull, stderr=devnull)  

            if not patch_cmd:
                raise ValueError("S3 path does not exist.")

    args_dict.pop("datasets")

    return list(nuts), list(years), args_dict
