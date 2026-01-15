"""
EVALUATION.PY - Classification evaluation utilities
"""

import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    import config
except ImportError:
    print("Error: config.py not found")
    raise

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluate and summarize traditional ML classifier results and create visualizations"""

    def __init__(self):
        self.results = {}   # dict: {model_name: { 'Overall Accuracy': float, ...}}
        self.output_dir = os.path.join(config.RESULTS_DIR, "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

        # Directorio de figuras específico para evaluación
        self.figures_dir = os.path.join(config.FIGURES_DIR, "evaluation")
        os.makedirs(self.figures_dir, exist_ok=True)

        # Colormap para clases
        self.class_colors = [
            '#4169E1',  # Blue - Water
            '#228B22',  # Forest Green - Forest
            '#ADFF2F',  # Green Yellow - Grassland
            '#FF6347',  # Tomato - Urban
            '#D2B48C',  # Tan - Bare Soil
            '#808080'   # Gray - Clouds / otros
        ]
        self.cmap = ListedColormap(self.class_colors[:len(config.CLASS_NAMES)])

    # ---------------------------------------------------------
    # RESUMEN NUMÉRICO
    # ---------------------------------------------------------
    def generate_summary_report(self, metrics=None):
        """
        Generate a text/JSON summary of classifier performance.
        """
        # 1) Use provided metrics if given
        if metrics is not None:
            self.results = metrics

        # 2) If still empty, try loading from disk
        if not self.results:
            json_path = os.path.join(self.output_dir, "ml_metrics.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        self.results = json.load(f)
                    logger.info(f"Loaded metrics from {json_path}")
                except Exception as e:
                    logger.warning(f"Could not load metrics from disk: {e}")

        # 3) If still empty, exit gracefully
        if not self.results:
            logger.warning("No classification results available. Skipping summary report.")
            return

        logger.info("Generating summary report...")

        # Best classifier by overall accuracy
        try:
            best_clf_name, best_clf_metrics = max(
                self.results.items(),
                key=lambda x: x[1].get("Overall Accuracy", 0.0)
            )
        except ValueError:
            logger.warning("Results dictionary is empty. Cannot determine best classifier.")
            return

        # Build a simple text summary
        lines = []
        lines.append("=" * 70)
        lines.append("CLASSIFICATION SUMMARY REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Models evaluated:")

        for name, m in self.results.items():
            acc = m.get("Overall Accuracy", None)
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            lines.append(f"  - {name}: Overall Accuracy = {acc_str}")

        lines.append("")
        lines.append(f"Best classifier: {best_clf_name}")
        best_acc = best_clf_metrics.get("Overall Accuracy", None)
        if best_acc is not None:
            lines.append(f"  Overall Accuracy: {best_acc:.4f}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")

        summary_text = "\n".join(lines)

        # Print to console
        print("\n" + summary_text)

        # Save to file
        summary_path = os.path.join(self.output_dir, "classification_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        logger.info(f"Saved summary report to {summary_path}")

        # Save metrics as JSON
        json_path = os.path.join(self.output_dir, "ml_metrics.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved metrics JSON to {json_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics JSON: {e}")

    # ---------------------------------------------------------
    # VISUALIZACIONES
    # ---------------------------------------------------------
    def _save_fig(self, fig, filename):
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    def _plot_rgb_image(self, imagery, rgb_indices=(3, 2, 1), filename="rgb_image.png"):
        r, g, b = rgb_indices
        rgb = np.stack(
            [imagery[:, :, r], imagery[:, :, g], imagery[:, :, b]],
            axis=-1
        )
        vmin, vmax = rgb.min(), rgb.max()
        if vmax > 1.0 or vmin < 0.0:
            rgb = (rgb - vmin) / (vmax - vmin + 1e-8)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.set_title("RGB Original Image")
        ax.axis("off")
        plt.tight_layout()
        self._save_fig(fig, filename)

    def _plot_extracted_bands(self, imagery, band_names=None,
                              max_cols=4, filename="extracted_bands.png"):
        h, w, n_bands = imagery.shape

        if band_names is None:
            try:
                band_names = config.DOWNLOAD_BANDS
                if len(band_names) != n_bands:
                    band_names = [f"Band {i}" for i in range(n_bands)]
            except AttributeError:
                band_names = [f"Band {i}" for i in range(n_bands)]
        else:
            if len(band_names) != n_bands:
                raise ValueError("band_names length must match imagery.shape[2]")

        cols = min(max_cols, n_bands)
        rows = int(np.ceil(n_bands / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        # mapa sencillo: nombre de banda -> colormap típico
        band_cmaps = {
            'B02': 'Blues',   # Blue
            'B03': 'Greens',  # Green
            'B04': 'Reds',    # Red
            'B05': 'Reds',    # NIR (often shown in red)
            'B06': 'Oranges', # SWIR1
            'B07': 'Oranges', # SWIR2
        }

        for i in range(n_bands):
            ax = axes[i]
            band = imagery[:, :, i]
            # normalizar para visualización 0-1
            bmin, bmax = band.min(), band.max()
            band_norm = (band - bmin) / (bmax - bmin + 1e-8)

            bname = str(band_names[i])
            cmap = band_cmaps.get(bname, 'viridis')

            im = ax.imshow(band_norm, cmap=cmap)
            ax.set_title(f'{bname}\n[{bmin:.3f}, {bmax:.3f}]')
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for j in range(n_bands, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Extracted Bands", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save_fig(fig, filename)

    def _plot_indices(self, indices_dict, filename="indices.png"):
        if not indices_dict:
            logger.warning("No indices in indices_dict, skipping index figure.")
            return

        names = list(indices_dict.keys())
        n = len(names)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        # Colormaps típicos por índice
        index_cmaps = {
            'NDVI': 'RdYlGn',   # vegetación: rojo(-) -> verde(+)
            'SAVI': 'RdYlGn',
            'NDWI': 'BrBG',     # agua: marrón(-) -> azul(+)
            'MNDWI': 'BrBG',
            'NDBI': 'OrRd',     # urbano: amarillo->rojo
        }

        for i, name in enumerate(names):
            ax = axes[i]
            idx_img = indices_dict[name]
            vmin, vmax = idx_img.min(), idx_img.max()

            # índices típicamente en [-1,1]
            if name in ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'SAVI']:
                vmin, vmax = -1.0, 1.0

            cmap = index_cmaps.get(name, 'RdYlGn')  # por defecto algo diverging

            im = ax.imshow(idx_img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f'{name}\n[{idx_img.min():.3f}, {idx_img.max():.3f}]')
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Calculated Indices", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save_fig(fig, filename)

    def _plot_classification_map(self, labels_2d, title, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            labels_2d,
            cmap=self.cmap,
            vmin=0,
            vmax=len(config.CLASS_NAMES) - 1
        )
        cbar = plt.colorbar(im, ax=ax, ticks=range(len(config.CLASS_NAMES)))
        cbar.ax.set_yticklabels(config.CLASS_NAMES)
        cbar.set_label("Land Cover Class")
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        self._save_fig(fig, filename)

    def generate_visualizations(
        self,
        imagery,
        features,
        feature_names,
        labels_1d,
        predictions_ml_1d=None,
        indices_dict=None,
        band_names=None,
        rgb_indices=(3, 2, 1),
        unet_prediction_2d=None
    ):
        """
        Generate and save all evaluation visualizations.
        """
        logger.info("Generating evaluation visualizations...")

        H, W = imagery.shape[0], imagery.shape[1]
        labels_2d = labels_1d.reshape(H, W)

        if indices_dict is None:
            indices_dict = {}

        # RGB original
        self._plot_rgb_image(imagery, rgb_indices=rgb_indices, filename="rgb_image.png")

        # Bandas extraídas
        self._plot_extracted_bands(imagery, band_names=band_names, filename="extracted_bands.png")

        # Índices calculados
        self._plot_indices(indices_dict, filename="indices.png")

        # Mapa de etiquetas / GT
        self._plot_classification_map(labels_2d, "Ground Truth / Labels", "labels_map.png")

        # Mapa ML
        if predictions_ml_1d is not None:
            ml_2d = predictions_ml_1d.reshape(H, W)
            self._plot_classification_map(ml_2d, "ML Classification Map", "ml_classification_map.png")

        # Mapa U-Net (si se pasa)
        if unet_prediction_2d is not None:
            self._plot_classification_map(unet_prediction_2d, "U-Net Prediction Map", "unet_prediction_map.png")

        logger.info("Evaluation visualizations generated.")
