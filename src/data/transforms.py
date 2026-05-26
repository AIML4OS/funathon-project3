"""
Albumentations transform pipeline applied to every batch.

Order matters and is fixed: Resize → (optional augmentations) → Normalize →
ToTensorV2. Augmentations must run on raw pixel values *before* normalisation
so that the augmented pixels remain in the same statistical space the mean/std
were measured on. ToTensorV2 is the last step: it converts the HWC NumPy array
to a CHW torch tensor, the layout PyTorch's Conv2d expects.
`max_pixel_value=1.0` tells Albumentations our reflectance values are already
floats in roughly [0, 1] — without this it would assume uint8 [0, 255] and
multiply the std by 255.
"""

from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transform(
    mean: List[float],
    std: List[float],
    augment: bool = False,
    resize: int = 512,
) -> A.Compose:
    """
    Build Albumentations transform pipeline.
    """

    transforms = [
        A.Resize(resize, resize),
    ]

    if augment:
        # Flips are safe: land-cover labels are invariant to horizontal/vertical
        # mirroring. We deliberately avoid rotations and colour jitter — the
        # former break the north-up convention, the latter shift the spectral
        # statistics the normalisation was computed on.
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])

    transforms.extend([
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ])

    return A.Compose(transforms)
