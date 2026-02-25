# %%
# ============================================
# STEP 2 — Model training
# ============================================
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from torch import Generator
from utils import (
    compute_global_normalization,
    load_data,
    build_transform
)

nuts_years_train = [
   "AT332_2018",
   "BE100_2018",
   "BE251_2018",
   "BG322_2018",
   "CY000_2018",
   "CZ072_2018",
   "DEA54_2018",
   "DK041_2018",
   "EE00A_2018",
   "EL521_2018",
   "ES612_2018",
   "FI1C1_2018",
   "FRJ27_2018",
   "FRK26_2018",
   "HR050_2018",
   "IE061_2018",
   "ITI32_2018",
   "LT028_2018",
   "LU000_2018",
   "LV008_2018",
   "MT001_2018",
   "NL33C_2018",
   "PL414_2018",
   "PT16I_2018",
   "RO123_2018",
   "SI035_2018",
   "SK022_2018",
   "UKJ22_2018",
   "AT332_2021",
   "BE100_2021",
   "BE251_2021",
   "BG322_2021",
   "CY000_2021",
   "CZ072_2021",
   "DEA54_2021",
   "DK041_2021",
   "EE00A_2021",
   "EL521_2021",
   "ES612_2021",
   "FI1C1_2021",
   "FRJ27_2021",
   "FRK26_2021",
   "HR050_2021",
   "IE061_2021",
   "ITI32_2021",
   "LT028_2021",
   "LU000_2021",
   "LV008_2021",
   "MT001_2021",
   "NL33C_2021",
   "PL414_2021",
   "PT16I_2021",
   "RO123_2021",
   "SI035_2021",
   "SK022_2021"
]

nuts_years_test = ["BE100_2021", "DEA54_2021", "CY000_2021", "LU000_2021"]

# -----------------------
# Config minimale
# -----------------------

CONFIG = {
    "batch_size": 32,
    "epochs": 5,
    "lr": 1e-3,
    "n_bands": 14,
    "resize": 512,
    "cuda": torch.cuda.is_available(),
}

# -----------------------
# Reproducibilité
# -----------------------

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# -----------------------
# Pipeline complet
# -----------------------

mean, std = compute_global_normalization(nuts_years_train, CONFIG["n_bands"])

train_patches, train_labels = load_data(nuts_years_train)
test_patches, test_labels = load_data(nuts_years_test)

train_transform = build_transform(mean, std, True, CONFIG["resize"])
test_transform = build_transform(mean, std, False, CONFIG["resize"])

dataset = get_dataset(train_patches, train_labels, CONFIG["n_bands"], train_transform)
test_dataset = get_dataset(test_patches, test_labels, CONFIG["n_bands"], test_transform)

train_dataset, val_dataset = random_split(
    dataset, [0.8, 0.2], generator=Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

# -----------------------
# 5️⃣ 🔥 TRAIN
# -----------------------

model = get_lightning_module(
    module_name="segformer-b5",
    n_bands=CONFIG["n_bands"],
    lr=CONFIG["lr"],
    cuda=CONFIG["cuda"],
)

trainer = get_trainer(
    earlystop=None,
    checkpoints=None,
    epochs=CONFIG["epochs"],
    num_sanity_val_steps=0,
    accumulate_batch=1,
)

trainer.fit(model, train_loader, val_loader)

# -----------------------
# 6️⃣ TEST
# -----------------------

trainer.test(model, test_loader)