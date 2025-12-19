# Land Cover Classification Sentinel Hub + Landsat 8

---

## 📌 Overview


<img width="2229" height="1475" alt="01_raw_imagery_visualization" src="https://github.com/user-attachments/assets/0ed22ec0-e0eb-4008-a914-5d437c618bc6" />

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
<img width="1866" height="883" alt="classification_result_ml" src="https://github.com/user-attachments/assets/672159a1-1e20-48da-ac6b-6143239d018a" />



---

## 🤖 Model Comparison

Two modeling approaches were evaluated:

- **Random Forest** – traditional machine learning baseline
- **U-Net CNN** – deep learning semantic segmentation model

The comparison focuses on both quantitative accuracy and spatial consistency.

**Key observations:**
- Random Forest provides reliable class separation with low computational cost.
- U-Net produces smoother, more spatially coherent predictions, but to reach high accuracy it needs more data or tunning.
- Deep learning improves boundary delineation between adjacent land cover types.
  
<img width="1484" height="881" alt="05_ml_accuracy_comparison" src="https://github.com/user-attachments/assets/0c74031e-99fc-4680-9c31-caaf75518e26" />

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


<img width="1783" height="1475" alt="04_confusion_matrices_ml" src="https://github.com/user-attachments/assets/e8738d17-3b91-4930-86d9-c1d87bf78383" />

---

## 🔍 Key Findings

- Spectral indices (NDVI, NDWI, NDBI, SAVI, MNDWI) were critical for accuracy.
- Traditional ML models perform well with limited training data.
- Model choice depends on data availability and computational resources.

---

## ⚠️ Limitations

- Limited ground truth data constrains classification reliability.
- Single-date imagery does not capture seasonal variability.
- Urban and bare soil remain challenging to separate spectrally.

---

## Why Random Forest Can Win
Several common reasons why a Random Forest beats a neural network (including UNet) are:

- **Small or noisy dataset:** With limited training data, deep models overfit easily, while Random Forests are more robust and often generalize better.
- **Handcrafted features vs. raw pixels:** If the Random Forest uses strong engineered features (e.g., spectral indices, texture, domain features) and the UNet learns only from raw images, the classical model can outperform.
- **Suboptimal UNet setup:** Inadequate depth, wrong loss, poor normalization, or insufficient training (too few epochs, bad learning rate, no augmentation) can all severely hurt UNet performance.

---

## 🚀 Suggested Improvements

- Increase U-Net performance.
- Incorporate multi-temporal imagery.
- Increase real ground truth samples.
- Add texture-based features.
- Use ensemble approaches (RF + U-Net).
- Apply post-processing spatial filters.

---

## ✅ Conclusion

This notebook demonstrates that multispectral satellite imagery combined with
machine learning and deep learning techniques can produce accurate and
interpretable land cover classification results. Random Forest serves as a
strong baseline, while U-Net could get even better performane after some tunning and feature ingenering.

