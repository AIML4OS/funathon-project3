# %%
# ============================================
# STEP 1 — Data Acquisition
# ============================================
from astrovision.data.satellite_image import SatelliteImage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from utils import download_clcpluslabel, tiff_to_numpy, get_file_system
import numpy as np

# =========================
# Sentinel-2 image
# =========================


fs = get_file_system()

s3_path = "projet-hackathon-ntts-2025/data-preprocessed/patchs/CLCplus-Backbone/SENTINEL2/BG322/2021/250/5553950_2341530_0_244.tif"
local_path = "5553950_2341530_0_244.tif"

fs.get(s3_path, local_path)  # ne fonctionne pas

satellite_image = SatelliteImage.from_raster(local_path)
normalized_image = satellite_image.normalize()

# (bands, H, W) -> (H, W, bands)
rgb = np.transpose(normalized_image.array, (1, 2, 0))[:, :, [1, 2, 3]]

# =========================
# CLCPlus label
# =========================

bbox_tuple = [5553950, 2341530, 5556450, 2344030]
year = 2021
filename = "test_label.tif"

download_clcpluslabel(filename, bbox_tuple, year)
img_array = tiff_to_numpy(filename)

# =========================
# Classes & colormap
# =========================

classes = [
    ("Sealed (1)", "#FF0100"),
    ("Woody – needle leaved trees (2)", "#238B23"),
    ("Woody – Broadleaved deciduous trees (3)", "#80FF00"),
    ("Woody – Broadleaved evergreen trees (4)", "#00FF00"),
    ("Low-growing woody plants (bushes, shrubs) (5)", "#804000"),
    ("Permanent herbaceous (6)", "#CCF24E"),
    ("Periodically herbaceous (7)", "#FEFF80"),
    ("Lichens and mosses (8)", "#FF81FF"),
    ("Non- and sparsely-vegetated (9)", "#BFBFBF"),
    ("Water (10)", "#0080FF"),
]

cmap = ListedColormap([color for _, color in classes])

# =========================
# Plot
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Sentinel RGB
axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB")
axes[0].axis("off")

# CLCPlus
axes[1].imshow(img_array, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("CLCPlus Label")
axes[1].axis("off")

# Légende globale
legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]

fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.35, 0.5),
    frameon=True
)

plt.tight_layout()
plt.show()

# All the labels for the training :
# s3://projet-hackathon-ntts-2025/data-preprocessed/labels/CLCplus-Backbone/SENTINEL2/


# %%
# ============================================
# STEP 2 — Model training
# ============================================

# YOUR CODE HERE


# %%
# ============================================
# STEP 3 — Inference and Statistics
# ============================================

# YOUR CODE HERE

# %%
# ============================================
# STEP 4 — Deployment
# ============================================

# YOUR CODE HERE