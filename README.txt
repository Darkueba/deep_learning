LAND COVER CLASSIFICATION – RESULTS SUMMARY
Sentinel Hub + Landsat 8
==================================================

PROJECT OUTCOME
--------------------------------------------------
This project successfully produced a multi-class land cover classification
map using Landsat-8 multispectral imagery. The final outputs include pixel-
level classification maps, quantitative accuracy metrics, and comparative
analysis between traditional machine learning and deep learning approaches.

Target land cover classes:
- Water
- Forest
- Grassland
- Urban
- Bare Soil


FINAL CLASSIFICATION RESULTS
--------------------------------------------------
The study area was fully classified at pixel level. Spatial patterns of
vegetation, water bodies, urban regions, and agricultural land are clearly
distinguishable in the final maps.

Final land cover classification map
(figures/final_classification_map.png)


MODEL PERFORMANCE OVERVIEW
--------------------------------------------------
Multiple models were evaluated and compared using standard remote sensing
accuracy metrics.

Evaluated models:
- Random Forest (traditional machine learning)
- U-Net Convolutional Neural Network (deep learning)

Key findings:
- Random Forest provided strong baseline performance with stable results
  across all classes.
- U-Net achieved superior spatial coherence and improved performance on
  complex land cover boundaries.
- Deep learning reduced salt-and-pepper noise common in pixel-based
  classifiers.

[IMAGE PLACEHOLDER]
Model comparison chart (Overall Accuracy & Kappa)
(figures/model_comparison.png)


ACCURACY METRICS (SUMMARY)
--------------------------------------------------
The following metrics were calculated on an independent test set:

- Overall Accuracy (OA)
- Kappa Coefficient
- Per-class Precision (User’s Accuracy)
- Per-class Recall (Producer’s Accuracy)
- F1-Score (per class)
- Intersection over Union (IoU)

General observations:
- Water and forest classes achieved the highest accuracy.
- Urban and bare soil showed moderate confusion due to similar spectral
  characteristics.
- Vegetation-related indices significantly improved class separability.

[IMAGE PLACEHOLDER]
Confusion matrices for best-performing models
(figures/confusion_matrices.png)


SPATIAL QUALITY OF RESULTS
--------------------------------------------------
Visual inspection confirms that:
- Water bodies are spatially consistent and well delineated.
- Forested areas show continuous coverage with minimal fragmentation.
- Urban regions are correctly identified but occasionally mixed with bare
  soil.
- Deep learning outputs exhibit smoother class boundaries and improved
  object-level consistency.

[IMAGE PLACEHOLDER]
Side-by-side comparison: Random Forest vs U-Net
(figures/spatial_comparison.png)


KEY OBSERVATIONS
--------------------------------------------------
- Spectral indices (NDVI, NDWI, NDBI, SAVI, MNDWI) played a critical role in
  improving classification accuracy.
- Traditional ML models are effective with limited data and lower
  computational cost.
- Deep learning models provide superior spatial realism when sufficient
  training data is available.


LIMITATIONS
--------------------------------------------------
- Limited ground truth data affects class boundary accuracy.
- Spectral similarity between urban and bare soil remains a challenge.
- Single-date imagery does not capture seasonal variability.


SUGGESTED IMPROVEMENTS
--------------------------------------------------
- Incorporate multi-temporal imagery to capture seasonal dynamics.
- Increase training data using real ground truth labels.
- Add texture-based features for better urban discrimination.
- Apply ensemble techniques combining ML and deep learning outputs.
- Use post-processing filters to further reduce isolated misclassifications.


CONCLUSION
--------------------------------------------------
The project demonstrates that combining multispectral satellite data with
machine learning and deep learning techniques enables accurate and visually
coherent land cover classification. The results confirm the effectiveness of
Random Forest as a strong baseline and U-Net as a high-performing model for
semantic segmentation in remote sensing applications.

[IMAGE PLACEHOLDER]
Final results overview figure
(figures/final_results_overview.png)
