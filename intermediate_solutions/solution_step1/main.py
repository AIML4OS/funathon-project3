# %%
# ============================================
# STEP 1 — Data Acquisition
# ============================================

import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def download_label(format_ext, filename, common_params, export_url):

    """Télécharge une image dans le format spécifié (tiff ou png)"""
    params = common_params.copy()
    params["format"] = format_ext

    response = requests.get(export_url, params=params, stream=True)

    if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/"):
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Erreur {format_ext.upper()} : ", response.status_code, response.text)


bbox_tuple = [5553950, 2341530, 5556450, 2344030]
year = 2021

# Dowload Sentinel Image
### Code

# Dowload CLCPlus label

export_url = f"https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/CLC_plus/CLMS_CLCplus_RASTER_{year}_010m_eu/ImageServer/exportImage"

xmin, ymin, xmax, ymax = bbox_tuple

resolution = 10

# Calcul de la taille en pixels pour garantir 1 pixel = 10 m
size_x = int((xmax - xmin) / resolution)
size_y = int((ymax - ymin) / resolution)

# Construction de la bounding box sous forme de chaîne
bbox_str = f"{xmin},{ymin},{xmax},{ymax}"

# Paramètres communs pour l'export
common_params = {
    "f": "image",
    "bbox": bbox_str,
    "bboxSR": "3035",   # Lambert-93
    "imageSR": "3035",  # Sortie aussi en Lambert-93
    "size": f"{size_x},{size_y}",  # Ajusté automatiquement pour 1 pixel = 10 m
}

filename = "test_label.tif"
download_label("tiff", filename, common_params, export_url)

img = Image.open(filename)
img_array = np.array(img)
img_array[(img_array == 254) | (img_array == 255)] = 0

npy_filename = filename.replace(".tif", ".npy")
np.save(npy_filename, img_array)

# Plot CLCPlus label
# Définition des classes et couleurs (ordre = valeur - 1)
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

# Colormap matplotlib
cmap = ListedColormap([color for _, color in classes])

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(img_array, cmap=cmap, vmin=1, vmax=10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Land Cover CLCPlus")

# Légende
legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]

ax.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout()
plt.show()



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