# Land Cover Classification Sentinel Hub + Landsat 8

---

## 📌 Overview

This notebook presents the final results of a land cover classification task
performed on Landsat-8 multispectral imagery. The objective was to generate
accurate, pixel-level land cover maps and evaluate model performance using
standard remote sensing metrics.

Land cover classes:
- Water
- Forest
- Grassland
- Urban
- Bare Soil

---

## 🛰️ Final Classification Output

The complete study area was classified at pixel level. The resulting map
clearly separates natural and built environments and highlights dominant land
cover patterns across the region.

**Output: Final Land Cover Map**



---

## 🤖 Model Comparison

Two modeling approaches were evaluated:

- **Random Forest** – traditional machine learning baseline
- **U-Net CNN** – deep learning semantic segmentation model

The comparison focuses on both quantitative accuracy and spatial consistency.

**Key observations:**
- Random Forest provides reliable class separation with low computational cost.
- U-Net produces smoother, more spatially coherent predictions.
- Deep learning improves boundary delineation between adjacent land cover types.

[OUTPUT CELL – IMAGE]
Model performance comparison (OA, Kappa)
(figures/model_comparison.png)

---

## 📊 Accuracy Metrics

Model performance was evaluated using an independent test set. The following
metrics were computed:

- Overall Accuracy (OA)
- Kappa Coefficient
- Precision (User’s Accuracy)
- Recall (Producer’s Accuracy)
- F1-score
- Intersection over Union (IoU)

**General trends:**
- Water and forest classes achieved the highest accuracy.
- Urban and bare soil show moderate confusion due to spectral similarity.
- Vegetation indices significantly improved separability.

[OUTPUT CELL – IMAGE]
Confusion matrices for best-performing models
(figures/confusion_matrices.png)

---

## 🗺️ Spatial Consistency Analysis

Visual inspection of classification maps reveals:

- Clean and continuous delineation of water bodies
- High consistency in forested regions
- Some misclassification between urban and bare soil
- Reduced salt-and-pepper noise in deep learning outputs

[OUTPUT CELL – IMAGE]
Side-by-side spatial comparison (RF vs U-Net)
(figures/spatial_comparison.png)

---

## 🔍 Key Findings

- Spectral indices (NDVI, NDWI, NDBI, SAVI, MNDWI) were critical for accuracy.
- Traditional ML models perform well with limited training data.
- Deep learning excels at capturing spatial structure and context.
- Model choice depends on data availability and computational resources.

---

## ⚠️ Limitations

- Limited ground truth data constrains classification reliability.
- Single-date imagery does not capture seasonal variability.
- Urban and bare soil remain challenging to separate spectrally.

---

## 🚀 Suggested Improvements

- Incorporate multi-temporal imagery
- Increase real ground truth samples
- Add texture-based features
- Use ensemble approaches (RF + U-Net)
- Apply post-processing spatial filters

---

## ✅ Conclusion

This notebook demonstrates that multispectral satellite imagery combined with
machine learning and deep learning techniques can produce accurate and
interpretable land cover classification results. Random Forest serves as a
strong baseline, while U-Net provides superior spatial realism and segmentation
quality.

[OUTPUT CELL – IMAGE]
Final results summary visualization
(figures/final_results_overview.png)
