"""
Functions to filter data based on labels.
"""

from typing import List
import numpy as np


def filter_indices_from_labels(
    label_paths: List[str],
    lower_threshold: float,
    upper_threshold: float,
) -> List[int]:
    """
    Return label indices whose built-up rate falls in `(lower, upper]`.

    For CLC+ Backbone labels, class id 1 ("Sealed") is the built-up category,
    so we count pixels equal to 1. With permissive thresholds (e.g.
    ``-1.0, 2.0``) the function is a pass-through over all tiles — useful as a
    hook to later restrict training to e.g. urban-heavy patches by tightening
    the bounds.
    """

    # CLC+ Backbone: class 1 = Sealed (the project's built-up equivalent).
    building_label = 1

    indices = []
    for idx, path in enumerate(label_paths):
        label = np.load(path)
        building_rate = np.mean(label == building_label)
        if (building_rate > lower_threshold) and (building_rate <= upper_threshold):
            indices.append(idx)
    return indices
