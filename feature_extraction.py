"""
MODULE 2: FEATURE EXTRACTION
Calculates spectral indices and prepares features for machine learning
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import generic_filter
import json

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class FeatureExtractor:
    """
    Extracts machine learning features from Landsat-8 bands
    Calculates spectral indices and prepares feature matrix
    """
    
    def __init__(self, imagery):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        imagery : numpy.ndarray
            Shape (H, W, 6) for [B04, B03, B02, B05, B06, B07]
        """
        self.imagery = imagery.astype(float)
        self.height, self.width, self.n_bands = imagery.shape
        
        # Normalize to reflectance [0, 1]
        self.imagery = self.imagery / 10000.0
        
        # Band indices
        self.red_idx = 0    # B04
        self.green_idx = 1  # B03
        self.blue_idx = 2   # B02
        self.nir_idx = 3    # B05
        self.swir1_idx = 4  # B06
        self.swir2_idx = 5  # B07
        
        # Feature list
        self.feature_names = []
        self.features_dict = {}
        
        print("\n" + "="*70)
        print("FEATURE EXTRACTION MODULE")
        print("="*70)
        print(f"\nInput imagery shape: {self.imagery.shape}")
        print(f"Normalized to [0, 1] range")
    
    def _safe_divide(self, numerator, denominator, fill_value=0):
        """
        Safe division avoiding division by zero
        
        Parameters:
        -----------
        numerator : numpy.ndarray
            Numerator array
        denominator : numpy.ndarray
            Denominator array
        fill_value : float
            Value to use where denominator is zero
        
        Returns:
        --------
        numpy.ndarray : Result of safe division
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result[~np.isfinite(result)] = fill_value
        return result
    
    def compute_ndvi(self):
        """
        Normalized Difference Vegetation Index
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1 (>0.5 = healthy vegetation)
        """
        nir = self.imagery[:, :, self.nir_idx]
        red = self.imagery[:, :, self.red_idx]
        ndvi = self._safe_divide(nir - red, nir + red)
        
        self.features_dict['NDVI'] = ndvi
        self.feature_names.append('NDVI')
        
        print(f"✓ NDVI computed - Range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
        return ndvi
    
    def compute_ndwi(self):
        """
        Normalized Difference Water Index
        NDWI = (Green - NIR) / (Green + NIR)
        Range: -1 to 1 (>0.3 = water bodies)
        """
        green = self.imagery[:, :, self.green_idx]
        nir = self.imagery[:, :, self.nir_idx]
        ndwi = self._safe_divide(green - nir, green + nir)
        
        self.features_dict['NDWI'] = ndwi
        self.feature_names.append('NDWI')
        
        print(f"✓ NDWI computed - Range: [{ndwi.min():.3f}, {ndwi.max():.3f}]")
        return ndwi
    
    def compute_ndbi(self):
        """
        Normalized Difference Built-up Index
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        Range: -1 to 1 (>0.0 = urban/built-up)
        """
        swir1 = self.imagery[:, :, self.swir1_idx]
        nir = self.imagery[:, :, self.nir_idx]
        ndbi = self._safe_divide(swir1 - nir, swir1 + nir)
        
        self.features_dict['NDBI'] = ndbi
        self.feature_names.append('NDBI')
        
        print(f"✓ NDBI computed - Range: [{ndbi.min():.3f}, {ndbi.max():.3f}]")
        return ndbi
    
    def compute_savi(self, L=0.5):
        """
        Soil-Adjusted Vegetation Index
        SAVI = ((NIR - Red) / (NIR + Red + L)) × (1 + L)
        L = soil brightness factor (typically 0.5)
        """
        nir = self.imagery[:, :, self.nir_idx]
        red = self.imagery[:, :, self.red_idx]
        savi = self._safe_divide((nir - red) * (1 + L), nir + red + L)
        
        self.features_dict['SAVI'] = savi
        self.feature_names.append('SAVI')
        
        print(f"✓ SAVI computed - Range: [{savi.min():.3f}, {savi.max():.3f}]")
        return savi
    
    def compute_mndwi(self):
        """
        Modified Normalized Difference Water Index
        MNDWI = (Green - SWIR1) / (Green + SWIR1)
        Better at distinguishing water from built-up
        """
        green = self.imagery[:, :, self.green_idx]
        swir1 = self.imagery[:, :, self.swir1_idx]
        mndwi = self._safe_divide(green - swir1, green + swir1)
        
        self.features_dict['MNDWI'] = mndwi
        self.feature_names.append('MNDWI')
        
        print(f"✓ MNDWI computed - Range: [{mndwi.min():.3f}, {mndwi.max():.3f}]")
        return mndwi
    
    def extract_all_features(self):
        """
        Extract all available features
        
        Returns:
        --------
        numpy.ndarray : Feature matrix (H*W, num_features)
        """
        print(f"\n{'='*70}")
        print("EXTRACTING FEATURES")
        print(f"{'='*70}\n")
        
        # Add original bands
        if config.USE_ORIGINAL_BANDS:
            for i, band_name in enumerate(config.DOWNLOAD_BANDS):
                self.features_dict[band_name] = self.imagery[:, :, i]
                self.feature_names.append(band_name)
            print(f"✓ Original bands added ({len(config.DOWNLOAD_BANDS)} bands)")
        
        # Calculate spectral indices
        if config.USE_SPECTRAL_INDICES:
            print("\nCalculating spectral indices:")
            if config.CALCULATE_INDICES.get('NDVI', False):
                self.compute_ndvi()
            if config.CALCULATE_INDICES.get('NDWI', False):
                self.compute_ndwi()
            if config.CALCULATE_INDICES.get('NDBI', False):
                self.compute_ndbi()
            if config.CALCULATE_INDICES.get('SAVI', False):
                self.compute_savi()
            if config.CALCULATE_INDICES.get('MNDWI', False):
                self.compute_mndwi()
        
        # Stack features into 2D array
        features_list = [self.features_dict[name].flatten() for name in self.feature_names]
        features_2d = np.column_stack(features_list)
        
        print(f"\n{'='*70}")
        print(f"Feature Extraction Complete")
        print(f"{'='*70}")
        print(f"Total features: {len(self.feature_names)}")
        print(f"Feature matrix shape: {features_2d.shape}")
        print(f"Feature names: {self.feature_names}")
        
        return features_2d
    
    def visualize_features(self):
        """
        Create visualization of extracted features
        """
        print(f"\n{'='*70}")
        print("VISUALIZING FEATURES")
        print(f"{'='*70}\n")
        
        n_features = len(self.features_dict)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        fig.suptitle('Extracted Features', fontsize=16)
        
        for idx, (feature_name, feature_data) in enumerate(self.features_dict.items()):
            # Normalize for visualization
            data_norm = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min() + 1e-8)
            
            im = axes[idx].imshow(data_norm, cmap='viridis')
            axes[idx].set_title(f'{feature_name}\n[{feature_data.min():.3f}, {feature_data.max():.3f}]')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx])
        
        # Hide unused subplots
        for idx in range(len(self.features_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, '02_extracted_features.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Feature visualization saved: {fig_path}")
        plt.close()
    
    def compute_statistics(self):
        """
        Compute and display feature statistics
        
        Returns:
        --------
        dict : Statistics for each feature
        """
        print(f"\n{'='*70}")
        print("FEATURE STATISTICS")
        print(f"{'='*70}")
        
        stats = {}
        print(f"\n{'Feature':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 65)
        
        for feature_name, feature_data in self.features_dict.items():
            stats[feature_name] = {
                'mean': float(feature_data.mean()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'median': float(np.median(feature_data))
            }
            
            print(f"{feature_name:<15} {stats[feature_name]['mean']:<12.4f} "
                  f"{stats[feature_name]['std']:<12.4f} "
                  f"{stats[feature_name]['min']:<12.4f} "
                  f"{stats[feature_name]['max']:<12.4f}")
        
        return stats
    
    def compute_correlation_matrix(self):
        """
        Compute feature correlation matrix
        
        Returns:
        --------
        numpy.ndarray : Correlation matrix
        """
        print(f"\n{'='*70}")
        print("COMPUTING CORRELATION MATRIX")
        print(f"{'='*70}\n")
        
        features_2d = self.extract_all_features()
        corr_matrix = np.corrcoef(features_2d.T)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(self.feature_names)))
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.set_yticklabels(self.feature_names)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        fig_path = os.path.join(config.FIGURES_DIR, '03_feature_correlation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Correlation matrix saved: {fig_path}")
        plt.close()
        
        return corr_matrix
    
    def save_features(self, features_2d):
        """
        Save extracted features to disk
        
        Parameters:
        -----------
        features_2d : numpy.ndarray
            Feature matrix (H*W, num_features)
        """
        # Save features
        features_path = os.path.join(config.PROCESSED_DATA_DIR, 'features.npy')
        np.save(features_path, features_2d)
        print(f"✓ Features saved: {features_path}")
        
        # Save feature names
        names_path = os.path.join(config.PROCESSED_DATA_DIR, 'feature_names.json')
        with open(names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved: {names_path}")
    
    def run(self):
        """
        Execute complete feature extraction pipeline
        
        Returns:
        --------
        tuple : (features_2d, feature_names, statistics)
        """
        # Extract features
        features_2d = self.extract_all_features()
        
        # Compute statistics
        stats = self.compute_statistics()
        
        # Compute correlation
        corr_matrix = self.compute_correlation_matrix()
        
        # Visualize
        self.visualize_features()
        
        # Save
        self.save_features(features_2d)
        
        print(f"\n{'='*70}")
        print("✓ FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}\n")
        
        return features_2d, self.feature_names, stats


def main():
    """Example usage"""
    # Load pre-downloaded imagery
    data_path = os.path.join(config.RAW_DATA_DIR, 'landsat_raw.npy')
    if os.path.exists(data_path):
        imagery = np.load(data_path)
        
        # Extract features
        fe = FeatureExtractor(imagery)
        features, names, stats = fe.run()
        
        print(f"\nFeature matrix ready for ML training!")
        print(f"Shape: {features.shape}")
    else:
        print(f"✗ Download imagery first: python data_acquisition.py")


if __name__ == "__main__":
    main()
