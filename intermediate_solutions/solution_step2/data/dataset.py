"""
All the __getitem__ functions will return a triplet
image, label, meta_data, with meta_data containing
paths to the non-transformed images or other necessary
information
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from albumentations import Compose
from astrovision.data import SatelliteImage
from torch.utils.data import Dataset

from data.download import get_file_system


class SegmentationDataset(Dataset):
    """
    Dataset class for segmentation.
    """

    def __init__(
        self,
        patchs: List,
        labels: List,
        n_bands: int,
        from_s3: bool,
        transform: Optional[Compose] = None,
    ):
        """
        Constructor.
        """
        self.patchs = patchs
        self.labels = labels
        self.n_bands = n_bands
        self.from_s3 = from_s3
        self.transform = transform

    def __getitem__(self, idx) -> Tuple:
        """
        Getitem fn.

        Args:
            idx: Index of item.

        Returns:
            Tuple: Image, label and metadata.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.from_s3:
            fs = get_file_system()
            si = SatelliteImage.from_raster(
                file_path=f"/vsis3/{self.patchs[idx]}",
                dep=None,
                date=None,
                n_bands=int(self.n_bands),
            )

            label = np.load(fs.open(f"s3://{self.labels[idx]}"))
        else:
            si = SatelliteImage.from_raster(
                file_path=self.patchs[idx], dep=None, date=None, n_bands=int(self.n_bands)
            )

            label = np.load(self.labels[idx])

        # Transforms
        sample = self.transform(image=np.transpose(si.array, [1, 2, 0]), mask=label)
        transformed_image = sample["image"]
        transformed_label = sample["mask"].type(torch.LongTensor)

        metadata = {"path_image": self.patchs[idx], "path_label": self.labels[idx]}
        encoded_inputs = {
            "pixel_values": transformed_image,
            "labels": transformed_label,
            "metadata": metadata,
        }
        return encoded_inputs

    def __len__(self):
        return len(self.patchs)
