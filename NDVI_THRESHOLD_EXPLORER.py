#!/usr/bin/env python3
"""
NDVI_THRESHOLD_EXPLORER.PY
Interactively adjust NDVI thresholds and visualize labels per pixel.

Usage:
  python ndvi_threshold_explorer.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import json
# ------------------------------------------------------------------
# CONFIGURE THESE PATHS / NAMES TO MATCH YOUR PIPELINE OUTPUT
# ------------------------------------------------------------------
FEATURES_PATH = os.path.join("data", "processed", "features.npy")
FEATURE_NAMES_PATH = os.path.join("data", "processed", "feature_names.json")
IMAGERY_RGB_PATH = os.path.join("data", "processed", "imagery_rgb.npy")  # optional
OUTPUT_FIG_PATH = os.path.join("figures", "ndvi_threshold_explorer.png")

# Class IDs and names (must match your config)
CLASS_NAMES = {
    0: "Water",
    1: "Forest",
    2: "Grassland",
    3: "Urban",
    4: "Bare Soil",
}

# Colors (R,G,B in [0,1])
CLASS_COLORS = {
    0: [0.2, 0.6, 0.9],   # Water
    1: [0.1, 0.5, 0.1],   # Forest
    2: [0.8, 0.9, 0.2],   # Grassland
    3: [0.8, 0.3, 0.1],   # Urban
    4: [0.7, 0.6, 0.4],   # Bare soil
}

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
if not os.path.exists(FEATURES_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
    raise FileNotFoundError("features.npy / feature_names.npy not found. "
                            "Export them from your pipeline first.")

features = np.load(FEATURES_PATH)          # shape: (N, F)
with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
    feature_names = json.load(f)
feature_names = np.array(feature_names)

# Derive image shape from config or infer (assumes full image flattened row-major)
# For example, if you know your imagery is H x W:
# H, W = 512, 512
# Here we try to infer a square image:
N = features.shape[0]
H = int(np.sqrt(N))
W = N // H
if H * W != N:
    raise ValueError(f"Cannot infer image size from N={N}. Set H,W manually.")

# NDVI vector
if "NDVI" in feature_names:
    ndvi_idx = list(feature_names).index("NDVI")
else:
    ndvi_idx = 0
    print("Warning: 'NDVI' not found in feature_names, using index 0")

ndvi = features[:, ndvi_idx].reshape(H, W)

# Optional RGB background
imagery_rgb = None
if os.path.exists(IMAGERY_RGB_PATH):
    imagery_rgb = np.load(IMAGERY_RGB_PATH)  # shape: (H, W, 3)
    if imagery_rgb.shape[0] != H or imagery_rgb.shape[1] != W:
        print("Warning: imagery_rgb shape does not match inferred H,W. Ignoring RGB.")
        imagery_rgb = None

# ------------------------------------------------------------------
# LABELING FUNCTION
# ------------------------------------------------------------------
def compute_labels(ndvi_img, t_w0, t_u0, t_b0, t_g0):
    """
    Map NDVI to labels using 4 thresholds:
      NDVI < t_w0           -> 0 Water
      t_w0 - t_u0           -> 3 Urban
      t_u0 - t_b0           -> 4 Bare Soil
      t_b0 - t_g0           -> 2 Grassland
      NDVI >= t_g0          -> 1 Forest
    """
    labels = np.zeros_like(ndvi_img, dtype=np.int32)

    labels[ndvi_img < t_w0] = 0
    labels[(ndvi_img >= t_w0) & (ndvi_img < t_u0)] = 3
    labels[(ndvi_img >= t_u0) & (ndvi_img < t_b0)] = 4
    labels[(ndvi_img >= t_b0) & (ndvi_img < t_g0)] = 2
    labels[ndvi_img >= t_g0] = 1

    return labels

def labels_to_rgb(labels_2d):
    h, w = labels_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cid, col in CLASS_COLORS.items():
        mask = labels_2d == cid
        rgb[mask] = col
    return rgb

# ------------------------------------------------------------------
# MATPLOTLIB INTERACTIVE FIGURE
# ------------------------------------------------------------------
plt.close("all")
fig, axes = plt.subplots(1, 2 if imagery_rgb is not None else 1, figsize=(14, 6))
if imagery_rgb is not None:
    ax_img, ax_cls = axes
else:
    ax_cls = axes

plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial thresholds (you can adjust)
t_w0_init = -0.05
t_u0_init = 0.15
t_b0_init = 0.35
t_g0_init = 0.55

labels_init = compute_labels(ndvi, t_w0_init, t_u0_init, t_b0_init, t_g0_init)
cls_rgb_init = labels_to_rgb(labels_init)

if imagery_rgb is not None:
    # Normalize RGB for display
    img_disp = imagery_rgb.astype(np.float32)
    img_disp /= max(1e-6, img_disp.max())
    ax_img.imshow(img_disp)
    ax_img.set_title("Original Satellite Imagery")
    ax_img.axis("off")

ax_cls.imshow(cls_rgb_init)
ax_cls.set_title("NDVI-based Labels")
ax_cls.axis("off")

# Legend
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
    for i in sorted(CLASS_NAMES.keys())
]
ax_cls.legend(handles=legend_elements, loc="upper right", fontsize=8)

# ------------------------------------------------------------------
# SLIDERS FOR THRESHOLDS
# ------------------------------------------------------------------
axcolor = "lightgoldenrodyellow"
ax_t_w0 = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
ax_t_u0 = plt.axes([0.1, 0.11, 0.8, 0.03], facecolor=axcolor)
ax_t_b0 = plt.axes([0.1, 0.07, 0.8, 0.03], facecolor=axcolor)
ax_t_g0 = plt.axes([0.1, 0.03, 0.8, 0.03], facecolor=axcolor)

from matplotlib.widgets import Slider, Button

# Sliders (NDVI typically in [-0.2, 1.0], adjust if needed)
t_w0_slider = Slider(ax_t_w0, "Water/Urban", -0.2, 0.3, valinit=t_w0_init, valstep=0.01)
t_u0_slider = Slider(ax_t_u0, "Urban/Bare", -0.1, 0.4, valinit=t_u0_init, valstep=0.01)
t_b0_slider = Slider(ax_t_b0, "Bare/Grass", 0.0, 0.6, valinit=t_b0_init, valstep=0.01)
t_g0_slider = Slider(ax_t_g0, "Grass/Forest", 0.2, 0.9, valinit=t_g0_init, valstep=0.01)

# ------------------------------------------------------------------
# UPDATE FUNCTION
# ------------------------------------------------------------------
def update(val):
    t_w0 = t_w0_slider.val
    t_u0 = t_u0_slider.val
    t_b0 = t_b0_slider.val
    t_g0 = t_g0_slider.val

    # Enforce ordering: t_w0 <= t_u0 <= t_b0 <= t_g0
    if t_u0 < t_w0:
        t_u0 = t_w0
        t_u0_slider.set_val(t_u0)
    if t_b0 < t_u0:
        t_b0 = t_u0
        t_b0_slider.set_val(t_b0)
    if t_g0 < t_b0:
        t_g0 = t_b0
        t_g0_slider.set_val(t_g0)

    labels_new = compute_labels(ndvi, t_w0, t_u0, t_b0, t_g0)
    cls_rgb_new = labels_to_rgb(labels_new)
    ax_cls.images[0].set_data(cls_rgb_new)

    fig.canvas.draw_idle()

t_w0_slider.on_changed(update)
t_u0_slider.on_changed(update)
t_b0_slider.on_changed(update)
t_g0_slider.on_changed(update)

# ------------------------------------------------------------------
# RESET & SAVE BUTTONS
# ------------------------------------------------------------------
reset_ax = plt.axes([0.1, 0.9, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset", color=axcolor, hovercolor="0.9")

save_ax = plt.axes([0.22, 0.9, 0.1, 0.04])
save_button = Button(save_ax, "Save PNG", color=axcolor, hovercolor="0.9")

def reset(event):
    t_w0_slider.reset()
    t_u0_slider.reset()
    t_b0_slider.reset()
    t_g0_slider.reset()

def save(event):
    # Recompute labels with current slider values and save image + labels
    t_w0 = t_w0_slider.val
    t_u0 = t_u0_slider.val
    t_b0 = t_b0_slider.val
    t_g0 = t_g0_slider.val

    labels_final = compute_labels(ndvi, t_w0, t_u0, t_b0, t_g0)
    cls_rgb_final = labels_to_rgb(labels_final)

    # Save colored classification
    plt.figure(figsize=(7, 6))
    plt.imshow(cls_rgb_final)
    plt.axis("off")
    plt.title("NDVI-based Labels (saved)")
    os.makedirs(os.path.dirname(OUTPUT_FIG_PATH), exist_ok=True)
    plt.savefig(OUTPUT_FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved classification PNG to {OUTPUT_FIG_PATH}")

    # Optionally save labels as .npy
    labels_path = OUTPUT_FIG_PATH.replace(".png", "_labels.npy")
    np.save(labels_path, labels_final)
    print(f"Saved label array to {labels_path}")

reset_button.on_clicked(reset)
save_button.on_clicked(save)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("NDVI Threshold Explorer")
    print(" - Adjust sliders to change class thresholds.")
    print(" - Click 'Save PNG' to export the current classification.")
    plt.show()

