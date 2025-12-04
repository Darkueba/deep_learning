#!/usr/bin/env python3
"""
PREDICT_NEW_IMAGE.PY
Load best trained ML model and classify a new image, saving a painted class map.
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from data_acquisition import LandsatDataAcquisition
from feature_extraction import FeatureExtractor

# -------------------------------------------------------------------
# Paths (adjust if your saving names are different)
# -------------------------------------------------------------------
BEST_MODEL_PATH = os.path.join(config.RESULTS_DIR, "traditional_ml", "best_model.joblib")
FIG_OUTPUT_PATH = os.path.join(config.FIGURES_DIR, "new_image_classification.png")

CLASS_NAMES = config.CLASS_NAMES  # e.g. ["Water","Forest","Grassland","Urban","Bare Soil"]

CLASS_COLORS = {
    0: [0.2, 0.6, 0.9],   # Water
    1: [0.1, 0.5, 0.1],   # Forest
    2: [0.8, 0.9, 0.2],   # Grassland
    3: [0.8, 0.3, 0.1],   # Urban
    4: [0.7, 0.6, 0.4],   # Bare soil
}

# -------------------------------------------------------------------
# Helper: turn class map into RGB
# -------------------------------------------------------------------
def labels_to_rgb(labels_2d):
    h, w = labels_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cid, col in CLASS_COLORS.items():
        mask = labels_2d == cid
        rgb[mask] = col
    return rgb

# -------------------------------------------------------------------
# 1. Load best model
# -------------------------------------------------------------------
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Best model not found at: {BEST_MODEL_PATH}")

print(f"Loading best model from: {BEST_MODEL_PATH}")
best_clf = joblib.load(BEST_MODEL_PATH)  # can be RF/SVM/etc. [web:338][web:343]

# -------------------------------------------------------------------
# 2. Acquire NEW image (change BBOX in config.py before running)
# -------------------------------------------------------------------
print("\nDownloading new image using current config.BBOX...")
acq = LandsatDataAcquisition()
imagery, metadata, stats = acq.run()
if imagery is None:
    raise RuntimeError("Data acquisition failed")

H, W, C = imagery.shape
print(f"New imagery shape: {imagery.shape}")

# -------------------------------------------------------------------
# 3. Extract features for NEW image
# -------------------------------------------------------------------
print("\nExtracting features for new image...")
fe = FeatureExtractor(imagery)
result = fe.run()

if isinstance(result, tuple):
    if len(result) == 3:
        features, feature_names, _ = result
    else:
        features, feature_names = result
else:
    raise TypeError("FeatureExtractor.run() must return tuple")

print(f"Features shape: {features.shape}")

# -------------------------------------------------------------------
# 4. Predict labels with best model
# -------------------------------------------------------------------
print("\nPredicting classes with best model...")
predictions = best_clf.predict(features)        # shape: (H*W,)
labels_2d = predictions.reshape(H, W)

# -------------------------------------------------------------------
# 5. Build RGB classification map
# -------------------------------------------------------------------
cls_rgb = labels_to_rgb(labels_2d)

# Build an approximate RGB of the satellite (for left panel)
# Swap bands if needed; here assume bands 2,1,0 ~ RGB-like
img_disp = imagery.astype(np.float32)
img_disp /= max(1e-6, img_disp.max())

# -------------------------------------------------------------------
# 6. Plot and save
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img_disp)
axes[0].set_title("Original Satellite Imagery")
axes[0].axis("off")

axes[1].imshow(cls_rgb)
axes[1].set_title("ML Classification Result (Best Model)")
axes[1].axis("off")

legend_elements = [
    mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
    for i in sorted(CLASS_COLORS.keys())
]
axes[1].legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()
os.makedirs(config.FIGURES_DIR, exist_ok=True)
plt.savefig(FIG_OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved classification image to: {FIG_OUTPUT_PATH}")
print("Done.")
