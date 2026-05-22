"""
Functions to filter data based on labels.
"""

from typing import List
import numpy as np


def label_building_rate(label: np.array, building_label: np.uint8) -> float:
    """
    Compute rate of building annotated pixels in
    segmentation label.

    Args:
        label (np.array): Label.
        building_label (np.uint8): Building label.

    Returns:
        float: Building rate.
    """
    return np.mean(label == building_label)


def load_label(label_path: str) -> np.array:
    """
    Load label.

    Args:
        label_path (str): Label path.

    Returns:
        np.array: Label array.
    """
    return np.load(label_path)


def filter_indices_from_labels(
    label_paths: List[str],
    lower_threshold: float,
    upper_threshold: float,
    type_labeler: str,
) -> List[int]:
    """
    Get indices to be used in a filter to keep only
    data points with a rate of building annotated pixels >
    `lower_threshold` and <= `upper_threshold`.

    Args:
        label_paths (List[str]): Paths to labels.
        lower_threshold (float): Lower threshold on building rate.
        upper_threshold (float): Upper threshold on building rate.
        type_labeler (str): Type of labeler.

    Returns:
        List[int]: Indices of selected labels.
    """

    building_label = 1 if type_labeler == "BDTOPO" else 0

    indices = []
    for idx, path in enumerate(label_paths):
        label = load_label(path)
        building_rate = label_building_rate(label, building_label)
        if (building_rate > lower_threshold) and (building_rate <= upper_threshold):
            indices.append(idx)
    return indices
