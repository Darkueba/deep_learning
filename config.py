"""
CONFIGURATION FILE FOR LAND COVER CLASSIFICATION PROJECT
Centralizes all parameters for reproducibility and easy modification
"""

# ============================================================================
# DATA ACQUISITION PARAMETERS
# ============================================================================

# Sentinel Hub Credentials 
SH_CLIENT_ID = ""
SH_CLIENT_SECRET = ""

# Study Area (Bounding Box in WGS84: lon_min, lat_min, lon_max, lat_max)
BBOX = [-4.50, 36.65, -4.30, 36.80]
STUDY_AREA_NAME = "Malaga"

# Time Range for Data


START_DATE = "2025-05-01"
END_DATE = "2025-08-28"

# Image Download Parameters
IMAGE_SIZE = (512, 512)  # Resolution in pixels
SPATIAL_RESOLUTION = 30  # Meters per pixel (Landsat native)
MAX_CLOUD_COVER = 0.2 # 20% max cloud cover

# ============================================================================
# LANDSAT BANDS CONFIGURATION
# ============================================================================

# Landsat 8-9 Bands (OLI Sensor)
BANDS_INFO = {
    'B02': {'name': 'Blue', 'wavelength': '0.45-0.51 μm', 'use': 'Coastal/Aerosol'},
    'B03': {'name': 'Green', 'wavelength': '0.53-0.59 μm', 'use': 'Vegetation'},
    'B04': {'name': 'Red', 'wavelength': '0.64-0.67 μm', 'use': 'Vegetation'},
    'B05': {'name': 'NIR', 'wavelength': '0.85-0.88 μm', 'use': 'Vegetation/Water'},
    'B06': {'name': 'SWIR1', 'wavelength': '1.57-1.65 μm', 'use': 'Moisture'},
    'B07': {'name': 'SWIR2', 'wavelength': '2.11-2.29 μm', 'use': 'Moisture'},
}

# Bands to download (adjust based on needs)
DOWNLOAD_BANDS = ['B04', 'B03', 'B02', 'B05', 'B06', 'B07']  # RGB + NIR + SWIR

# ============================================================================
# LAND COVER CLASSIFICATION CLASSES
# ============================================================================

# Define your land cover classes
CLASS_NAMES = [
    'Water',           # 0 - Lakes, rivers, ponds
    'Forest',          # 1 - Dense vegetation
    'Grassland',       # 2 - Sparse vegetation, meadows
    'Urban/Built-up',  # 3 - Buildings, roads, infrastructure
    'Bare Soil',       # 4 - Agricultural fields, exposed soil
    'Clouds'           # 5 - Cloud pixels (to be masked)
]

NUM_CLASSES = len(CLASS_NAMES) - 1  # Exclude clouds from classification

# ============================================================================
# SPECTRAL INDICES PARAMETERS
# ============================================================================

# Spectral Indices to Calculate
CALCULATE_INDICES = {
    'NDVI': True,   # Normalized Difference Vegetation Index
    'NDWI': True,   # Normalized Difference Water Index
    'NDBI': True,   # Normalized Difference Built-up Index
    'SAVI': True,   # Soil-Adjusted Vegetation Index
    'EVI': False,   # Enhanced Vegetation Index (optional)
    'MNDWI': True,  # Modified NDWI
}

# Feature Selection
USE_ORIGINAL_BANDS = True  # Include original bands as features
USE_SPECTRAL_INDICES = True  # Include calculated indices
USE_TEXTURE_FEATURES = False  # GLCM texture (slower, optional)

# ============================================================================
# MACHINE LEARNING PARAMETERS
# ============================================================================

# Train/Test Split
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# Traditional Classifiers Parameters
CLASSIFIER_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    'svm': {
        'kernel': 'rbf',
        'C': 100,
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced'
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'distance',
        'n_jobs': -1
    },
    'mlc': {
        # Linear Discriminant Analysis (Maximum Likelihood)
        'n_components': None
    }
}

# ============================================================================
# DEEP LEARNING PARAMETERS
# ============================================================================

# CNN Architecture
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = len(DOWNLOAD_BANDS) if USE_ORIGINAL_BANDS else 0
IMG_CHANNELS += sum(CALCULATE_INDICES.values()) if USE_SPECTRAL_INDICES else 0

# Training Parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
CV_FOLDS = 5

# CNN Filter Sizes
CNN_FILTERS = [32, 64, 128, 256]
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

# U-Net Filter Sizes
UNET_FILTERS = [32, 64, 128, 256]
UNET_DROPOUT = 0.3

# Data Augmentation
USE_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'horizontal_flip': True,
    'vertical_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
}

# Patch sizes for spatial context analysis
PATCH_SIZES = [1, 3, 5, 7, 11]

# ============================================================================
# OUTPUT AND RESULTS PARAMETERS
# ============================================================================

# Directory Structure
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
TRAINING_DATA_DIR = 'data/training'
RESULTS_DIR = 'data/results'
FIGURES_DIR = 'figures'
LOGS_DIR = 'logs'

# Output Format
SAVE_MODELS = True
SAVE_VISUALIZATIONS = True
SAVE_CONFUSION_MATRICES = True
SAVE_FEATURE_IMPORTANCE = True

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
VERBOSE = True

# ============================================================================
# RESEARCH & ANALYSIS PARAMETERS
# ============================================================================

# For requirement analysis and literature review
REFERENCES = {
    'maxar_lulc': 'Maxar GLH Land Use/Land Cover (2021)',
    'corine': 'CORINE Land Cover Database',
    'nlcd': 'National Land Cover Database (NLCD)',
    'sentinel2': 'Sentinel-2 Mission (10-60m resolution)',
    'landsat': 'Landsat 8-9 Program (30m resolution)',
}

# Performance Metrics to Track
METRICS_TO_TRACK = [
    'Overall Accuracy',
    'Kappa Coefficient',
    'Precision (Macro)',
    'Recall (Macro)',
    'F1-Score (Macro)',
    'F1-Score (Weighted)',
    'IoU (Jaccard Index)',
    'Training Time',
    'Inference Time'
]

# ============================================================================
# NOTES
# ============================================================================
"""
Configuration Usage:
1. All modules import this config.py
2. Modify parameters here instead of in individual files
3. Easy to track what changed between experiments
4. Reproducible results with documented settings

Example in other files:
    import config
    bbox = config.BBOX
    classes = config.CLASS_NAMES
"""
