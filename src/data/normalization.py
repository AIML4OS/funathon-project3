"""
Compute per-band mean/std from the training tiles.

These statistics must come from the training set (not from each batch) and be
*frozen* afterwards: the same numbers are reused at inference, otherwise the
model sees inputs in a different statistical range from what it learnt on and
predictions degrade silently. The computed values are persisted alongside the
model (MLflow run params) so inference code can restore them exactly.
"""

from typing import List, Tuple
import numpy as np
import yaml

from src.data.loading import get_patchs_labels
from src.data.filter import filter_indices_from_labels


def normalization_params(
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
):
    """
    Load normalization parameters from YAML file.
    """

    params_path = (
        f"data/data-preprocessed/patchs/"
        f"{source}/{dep}/{year}/{tiles_size}/metrics-normalization.yaml"
    )

    with open(params_path) as f:
        params = yaml.safe_load(f)

    return params["mean"], params["std"]


def compute_global_normalization(
    nuts_years: List[str],
    n_bands: int,
) -> Tuple[List[float], List[float]]:

    means = []
    stds = []
    weights = []

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

        indices = filter_indices_from_labels(labels, -1.0, 2.0)

        if len(indices) == 0:
            continue

        mean, std = normalization_params(
            "S2", nuts, year, "512", "default"
        )

        means.append(mean[:n_bands])
        stds.append(std[:n_bands])
        weights.append(len(indices))

    if len(means) == 0:
        raise ValueError("No valid data found for normalization.")

    global_mean = np.average(means, axis=0, weights=weights)

    global_std = np.sqrt(
        np.average([s ** 2 for s in stds], axis=0, weights=weights)
    )

    return global_mean.tolist(), global_std.tolist()
