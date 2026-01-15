import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class ClassificationVisualizer:
    """Creates and saves visualizations for classification results"""

    def __init__(self):
        """Initialize visualizer"""
        print("\n" + "=" * 70)
        print("VISUALIZATION MODULE")
        print("=" * 70)

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

    # ---------------------- helper save ---------------------- #
    def save_figure(self, fig, filename):
        filepath = os.path.join(config.FIGURES_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved: {filepath}")
        plt.close(fig)

    # ------------------ classification map ------------------- #
    def plot_classification_map(self, classification, title='Classification Map',
                                filename='classification_map.png'):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            classification,
            cmap=self.cmap,
            vmin=0,
            vmax=len(config.CLASS_NAMES) - 1
        )
        cbar = plt.colorbar(im, ax=ax, ticks=range(len(config.CLASS_NAMES)))
        cbar.ax.set_yticklabels(config.CLASS_NAMES)
        cbar.set_label('Land Cover Class')
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        self.save_figure(fig, filename)

    # ---------------- feature importance --------------------- #
    def plot_feature_importance(self, feature_names, importance_values,
                                classifier_name='Random Forest',
                                filename='feature_importance.png'):
        sorted_idx = np.argsort(importance_values)[::-1][:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_idx)), importance_values[sorted_idx], alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Importance')
        ax.set_title(f'{classifier_name} - Top 15 Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        self.save_figure(fig, filename)

    # ---------------- spectral signatures -------------------- #
    def plot_spectral_signatures(self, imagery, labels, band_names=None,
                                 filename='spectral_signatures.png'):
        if band_names is None:
            band_names = config.DOWNLOAD_BANDS

        fig, ax = plt.subplots(figsize=(10, 6))
        for class_idx in range(len(config.CLASS_NAMES) - 1):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                mean_spectrum = imagery[class_mask].mean(axis=0)
                ax.plot(
                    band_names,
                    mean_spectrum,
                    marker='o',
                    label=config.CLASS_NAMES[class_idx]
                )

        ax.set_xlabel('Band')
        ax.set_ylabel('Reflectance')
        ax.set_title('Mean Spectral Signatures per Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save_figure(fig, filename)

    # ---------------------- RGB image ------------------------ #
    def plot_rgb_image(self, imagery, rgb_band_indices=(3, 2, 1),
                       filename='rgb_image.png', title='RGB Image'):
        r_idx, g_idx, b_idx = rgb_band_indices
        rgb = np.stack(
            [imagery[:, :, r_idx], imagery[:, :, g_idx], imagery[:, :, b_idx]],
            axis=-1
        )
        rgb_min, rgb_max = rgb.min(), rgb.max()
        if rgb_max > 1.0 or rgb_min < 0.0:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        self.save_figure(fig, filename)

    # ------------------- extracted bands --------------------- #
    def plot_extracted_bands(self, imagery, band_names=None,
                             max_cols=4, filename='extracted_bands.png'):
        h, w, num_bands = imagery.shape

        if band_names is None:
            try:
                band_names = config.DOWNLOAD_BANDS
                if len(band_names) != num_bands:
                    band_names = [f'Band {i}' for i in range(num_bands)]
            except AttributeError:
                band_names = [f'Band {i}' for i in range(num_bands)]
        else:
            if len(band_names) != num_bands:
                raise ValueError("band_names length must match imagery.shape[2]")

        cols = min(max_cols, num_bands)
        rows = int(np.ceil(num_bands / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for i in range(num_bands):
            ax = axes[i]
            band = imagery[:, :, i]
            im = ax.imshow(band, cmap='gray')
            ax.set_title(str(band_names[i]))
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for j in range(num_bands, len(axes)):
            axes[j].axis('off')

        fig.suptitle('Extracted Bands', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self.save_figure(fig, filename)

    # ----------------------- indices ------------------------- #
    def plot_indices(self, indices_dict, filename='indices.png'):
        if not indices_dict:
            print("No indices provided to plot_indices.")
            return

        index_names = list(indices_dict.keys())
        num_indices = len(index_names)
        cols = min(3, num_indices)
        rows = int(np.ceil(num_indices / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for i, name in enumerate(index_names):
            ax = axes[i]
            idx_img = indices_dict[name]
            im = ax.imshow(idx_img, cmap='RdYlGn')
            ax.set_title(name)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for j in range(num_indices, len(axes)):
            axes[j].axis('off')

        fig.suptitle('Calculated Indices', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self.save_figure(fig, filename)

    # ------------------- CNN / U-Net pred -------------------- #
    def plot_cnn_prediction(self, prediction, title='CNN/UNet Prediction',
                            filename='cnn_unet_prediction.png'):
        if prediction.ndim == 3 and prediction.shape[-1] == 1:
            prediction = prediction[:, :, 0]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            prediction,
            cmap=self.cmap,
            vmin=0,
            vmax=len(config.CLASS_NAMES) - 1
        )
        cbar = plt.colorbar(im, ax=ax, ticks=range(len(config.CLASS_NAMES)))
        cbar.ax.set_yticklabels(config.CLASS_NAMES)
        cbar.set_label('Predicted Class')
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        self.save_figure(fig, filename)

    # --------------- convenience: all at once ---------------- #
    def generate_all_visualizations(
        self,
        imagery,
        labels_1d,
        indices_dict,
        cnn_prediction,
        feature_names=None,
        importance_values=None,
        band_names=None,
        rgb_band_indices=(3, 2, 1),
        prefix=''
    ):
        img_height, img_width = imagery.shape[0], imagery.shape[1]
        labels_2d = labels_1d.reshape(img_height, img_width)

        def pf(name):
            return f"{prefix}_{name}" if prefix else name

        self.plot_rgb_image(
            imagery,
            rgb_band_indices=rgb_band_indices,
            filename=pf('rgb_image.png'),
            title='RGB Original Image'
        )
        self.plot_extracted_bands(
            imagery,
            band_names=band_names,
            filename=pf('extracted_bands.png')
        )
        self.plot_indices(
            indices_dict,
            filename=pf('indices.png')
        )
        self.plot_classification_map(
            labels_2d,
            title='ML Classification / Labels',
            filename=pf('ml_classification_map.png')
        )
        self.plot_cnn_prediction(
            cnn_prediction,
            title='CNN / U-Net Prediction',
            filename=pf('cnn_unet_prediction.png')
        )
        self.plot_spectral_signatures(
            imagery,
            labels_2d,
            band_names=band_names,
            filename=pf('spectral_signatures.png')
        )
        if feature_names is not None and importance_values is not None:
            self.plot_feature_importance(
                feature_names,
                importance_values,
                filename=pf('feature_importance.png')
            )
