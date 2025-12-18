"""
MODULE 6: VISUALIZATION
Creates publication-quality figures and analysis plots
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class ClassificationVisualizer:
    """Creates visualizations for classification results"""
    
    def __init__(self):
        """Initialize visualizer"""
        print("\n" + "="*70)
        print("VISUALIZATION MODULE")
        print("="*70)
        
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        
        # Define color map for land cover classes
        self.class_colors = [
            '#4169E1',  # Blue - Water
            '#228B22',  # Forest Green - Forest
            '#ADFF2F',  # Green Yellow - Grassland
            '#FF6347',  # Tomato - Urban
            '#D2B48C',  # Tan - Bare Soil
            '#808080'   # Gray - Clouds
        ]
        
        self.cmap = ListedColormap(self.class_colors[:len(config.CLASS_NAMES)])
    
    def plot_classification_map(self, classification, title='Classification Map'):
        """
        Plot classification map
        
        Parameters:
        -----------
        classification : numpy.ndarray
            2D classification map
        title : str
            Map title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(classification, cmap=self.cmap, vmin=0, vmax=len(config.CLASS_NAMES)-1)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=range(len(config.CLASS_NAMES)))
        cbar.ax.set_yticklabels(config.CLASS_NAMES)
        cbar.set_label('Land Cover Class')
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_feature_importance(self, feature_names, importance_values, classifier_name='Random Forest'):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_names : list
            Feature names
        importance_values : numpy.ndarray
            Importance values
        classifier_name : str
            Classifier name
        """
        # Sort by importance
        sorted_idx = np.argsort(importance_values)[::-1][:15]  # Top 15
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(sorted_idx)), importance_values[sorted_idx], alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title(f'{classifier_name} - Top 15 Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_spectral_signatures(self, imagery, labels, band_names=None):
        """
        Plot spectral signatures per class
        
        Parameters:
        -----------
        imagery : numpy.ndarray
            Image array (H, W, num_bands)
        labels : numpy.ndarray
            Classification map (H, W)
        band_names : list, optional
            Band names
        """
        if band_names is None:
            band_names = config.DOWNLOAD_BANDS
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for class_idx in range(len(config.CLASS_NAMES)-1):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                mean_spectrum = imagery[class_mask].mean(axis=0)
                ax.plot(band_names, mean_spectrum, marker='o', label=config.CLASS_NAMES[class_idx])
        
        ax.set_xlabel('Band')
        ax.set_ylabel('Reflectance')
        ax.set_title('Mean Spectral Signatures per Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def save_figure(self, fig, filename):
        """
        Save figure to disk
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure object
        filename : str
            Output filename
        """
        filepath = os.path.join(config.FIGURES_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved: {filepath}")
        plt.close(fig)


def main():
    """Example usage"""
    vis = ClassificationVisualizer()
    print("Visualization module ready")


if __name__ == "__main__":
    main()
