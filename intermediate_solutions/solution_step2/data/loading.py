# ==========================================================
# data/loading.py
# ==========================================================

import os
from pathlib import Path
from typing import List, Tuple

from data.download import get_file_system, download_data
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
        fs = get_file_system()

        patchs = fs.ls(
            f"projet-hackathon-ntts-2025/data-preprocessed/patchs/"
            f"{type_labeler}/{source}/{dep}/{year}/{tiles_size}"
        )

        labels = fs.ls(
            f"projet-hackathon-ntts-2025/data-preprocessed/labels/"
            f"{type_labeler}/{source}/{dep}/{year}/{tiles_size}"
        )

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

    fs = get_file_system()

    for nut, year in zip(nuts, years):
        s3_path = (
            f"s3://projet-hackathon-ntts-2025/data-preprocessed/"
            f"patchs/{args_dict['type_labeler']}/"
            f"{args_dict['source']}/{nut}/{year}"
        )

        if not fs.exists(s3_path):
            raise ValueError(f"S3 path {s3_path} does not exist.")

    args_dict.pop("datasets")

    return list(nuts), list(years), args_dict