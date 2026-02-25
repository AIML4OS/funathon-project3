import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from s3fs import S3FileSystem
from typing import Tuple


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )


def format_datasets(args_dict: dict) -> Tuple[str, int]:
    """
    Format datasets.

    Args:
        args_dict (dict): A dictionary containing the command-line arguments.

    Returns:
        Tuple[str, int]: A tuple containing the list of departments and years extracted from the dataset names.

    Raises:
        ValueError: If the S3 path does not exist.

    """
    nuts, years = zip(*[item.split("_") for item in args_dict["datasets"]])
    nuts = [nut.upper() for nut in nuts]
    fs = get_file_system()
    for nut, year in zip(nuts, years):
        s3_path = f"s3://projet-hackathon-ntts-2025/data-preprocessed/patchs/{args_dict['type_labeler']}/{args_dict['source']}/{nut.upper()}/{year}"
        if not fs.exists(s3_path):
            raise ValueError(f"S3 path {s3_path} does not exist.")
    args_dict.pop("datasets")
    return nuts, years, args_dict


def get_trainer(max_epochs: int):
    return pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
    )


def get_dataset(patches, labels, n_bands, transform=None):
    return dataset_dict["segmentation"](patches, labels, n_bands, False, transform)


def get_model(module_name, type_labeler, n_bands):
    return module_dict[module_name](
        n_bands=n_bands,
        logits=False,
        freeze_encoder=False,
        type_labeler=type_labeler,
    )


def get_loss(loss_name, weights=None, label_smoothing=0.0):
    config = loss_dict[loss_name]
    kwargs = config["kwargs"].copy()

    if config["weighted"]:
        kwargs["weights"] = weights

    if config["smoothing"]:
        kwargs["label_smoothing"] = label_smoothing

    return config["loss_function"](**kwargs)


def get_lightning_module(
    module_name,
    type_labeler,
    loss_name,
    n_bands,
    lr,
    weights=None,
    label_smoothing=0.0,
):

    model = get_model(module_name, type_labeler, n_bands)
    loss = get_loss(loss_name, weights, label_smoothing)

    optimizer = torch.optim.AdamW
    optimizer_params = {"lr": lr}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        "monitor": "validation_loss",
        "mode": "min",
        "patience": 2,
    }

    LightningModule = task_dict["segmentation"]

    return LightningModule(
        model=model,
        loss=loss,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        scheduler_interval="epoch",
    )
# -----------------------
# 1️⃣ Normalisation globale
# -----------------------


def compute_global_normalization(nuts_years, n_bands):
    means, stds, weights = [], [], []

    for nuts_year in nuts_years:
        nuts, year = nuts_year.split("_")

        patches, labels = get_patchs_labels(nuts, year)
        indices = filter_indices_from_labels(labels, -1.0, 2.0)

        mean, std = normalization_params(nuts, year)

        means.append(mean[:n_bands])
        stds.append(std[:n_bands])
        weights.append(len(indices))

    global_mean = np.average(means, axis=0, weights=weights)

    global_std = np.sqrt(
        np.average([s**2 for s in stds], axis=0, weights=weights)
    )

    return global_mean.tolist(), global_std.tolist()

# -----------------------
# 2️⃣ Chargement données
# -----------------------


def load_data(nuts_years):
    patches_all, labels_all = [], []

    for nuts_year in nuts_years:
        nuts, year = nuts_year.split("_")

        patches, labels = get_patchs_labels(nuts, year)
        patches.sort()
        labels.sort()

        indices = filter_indices_from_labels(labels, -1.0, 2.0)

        patches_all += [patches[i] for i in indices]
        labels_all += [labels[i] for i in indices]

    return patches_all, labels_all

# -----------------------
# 3️⃣ Transforms
# -----------------------


def build_transform(mean, std, augment, resize):
    transforms = [A.Resize(resize, resize)]

    if augment:
        transforms += [A.HorizontalFlip(), A.VerticalFlip()]

    transforms += [
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)
