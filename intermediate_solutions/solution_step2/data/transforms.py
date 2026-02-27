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