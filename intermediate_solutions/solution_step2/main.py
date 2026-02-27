# ==========================================================
# main.py
# ==========================================================

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torch import Generator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data.loading import load_data
from data.normalization import compute_global_normalization
from data.transforms import build_transform
from training.lightning import get_lightning_module
from dataset import SegmentationDataset


# ==========================================================
# 1️⃣ CONFIG
# ==========================================================

CONFIG = {
    "train_regions": [
        "AT332", "BE100", "BE251", "BG322", "CY000", "CZ072",
        "DEA54", "DK041", "EE00A", "EL521", "ES612", "FI1C1",
        "FRJ27", "FRK26", "HR050", "IE061", "ITI32", "LT028",
        "LU000", "LV008", "MT001", "NL33C", "PL414", "PT16I",
        "RO123", "SI035", "SK022"
    ],
    "train_years": ["2018", "2021"],
    "test_regions": ["BE100", "DEA54", "CY000", "LU000"],
    "test_year": "2021",

    "batch_size": 32,
    "test_batch_size": 16,
    "epochs": 20,
    "lr": 1e-3,
    "n_bands": 14,
    "resize": 512,
    "num_workers": os.cpu_count(),
    "seed": 42,
}


# ==========================================================
# 2️⃣ UTILS
# ==========================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)


def build_region_year_list(regions, years):
    return [f"{r}_{y}" for r in regions for y in years]


# ==========================================================
# 3️⃣ MAIN PIPELINE
# ==========================================================


# Reproducibilité
set_seed(CONFIG["seed"])

# -------- Dataset IDs --------
train_ids = build_region_year_list(
    CONFIG["train_regions"],
    CONFIG["train_years"],
)

test_ids = build_region_year_list(
    CONFIG["test_regions"],
    [CONFIG["test_year"]],
)

# -------- Normalisation --------
print("📊 Computing normalization...")
mean, std = compute_global_normalization(
    train_ids,
    CONFIG["n_bands"]
)

# -------- Chargement --------
print("📂 Loading data...")
train_patches, train_labels = load_data(train_ids)
test_patches, test_labels = load_data(test_ids)

# -------- Transforms --------
train_transform = build_transform(
    mean, std, augment=True, resize=CONFIG["resize"]
)

test_transform = build_transform(
    mean, std, augment=False, resize=CONFIG["resize"]
)

# -------- Dataset --------
full_dataset = SegmentationDataset(
    train_patches,
    train_labels,
    CONFIG["n_bands"],
    train_transform,
)

test_dataset = SegmentationDataset(
    test_patches,
    test_labels,
    CONFIG["n_bands"],
    test_transform,
)

# -------- Split train / val --------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=Generator().manual_seed(CONFIG["seed"]),
)

# -------- DataLoaders --------
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["test_batch_size"],
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
)

# -------- Model --------
model = get_lightning_module(
    n_bands=CONFIG["n_bands"],
    lr=CONFIG["lr"],
)

# -------- Trainer --------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model",
    ),
]

trainer = pl.Trainer(
    max_epochs=CONFIG["epochs"],
    accelerator="auto",
    devices="auto",
    callbacks=callbacks,
    log_every_n_steps=10,
)

# -------- Training --------
print("🔥 Training...")
trainer.fit(model, train_loader, val_loader)

print("🧪 Testing...")
trainer.test(model, test_loader)
