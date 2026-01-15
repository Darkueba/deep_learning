#!/usr/bin/env python3
"""
MAIN_PIPELINE.PY - Fixed for segmentation with visualization
Orchestrates complete land cover classification with semantic segmentation
"""

import numpy as np
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import config
except ImportError:
    print("Error: config.py not found")
    raise

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class LandCoverClassificationPipeline:
    """Complete pipeline for land cover segmentation"""

    def __init__(self):
        """Initialize pipeline"""
        print("\n" + "="*70)
        print("LAND COVER CLASSIFICATION PIPELINE - SEGMENTATION")
        print("="*70)

        self.start_time = time.time()
        self.stages = {}

        # Create directories
        for directory in [
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.TRAINING_DATA_DIR,
            config.RESULTS_DIR,
            config.FIGURES_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Placeholders
        self.imagery = None
        self.features = None
        self.feature_names = None
        self.labels = None
        self.indices_dict = {}
        self.predictions_ml = None
        self.unet_pred = None
        self.metrics_ml = None

    # ---------------------------------------------------------
    # Helper: ML classification visualization
    # ---------------------------------------------------------
    def visualize_classification_result(self, labels_2d, imagery=None, title="Land Cover Classification"):
        """
        Create a colored map of classification results and save as image.
        """
        logger.info(f"\nVisualizing classification results...")

        colors = {
            0: [0.2, 0.6, 0.9],      # Water
            1: [0.1, 0.5, 0.1],      # Forest
            2: [0.8, 0.9, 0.2],      # Grassland
            3: [0.8, 0.3, 0.1],      # Urban
            4: [0.7, 0.6, 0.4],      # Bare soil
        }

        class_names = {
            0: "Water",
            1: "Forest",
            2: "Grassland",
            3: "Urban",
            4: "Bare Soil",
        }

        h, w = labels_2d.shape
        colored_output = np.zeros((h, w, 3), dtype=np.float32)

        for class_id, color in colors.items():
            mask = labels_2d == class_id
            colored_output[mask] = color

        fig, axes = plt.subplots(1, 2 if imagery is not None else 1, figsize=(14, 6))

        if imagery is not None:
            if imagery.shape[2] >= 3:
                # imagery: [B04, B03, B02, B05, B06, B07] -> RGB = [0,1,2]
                rgb = imagery[:, :, [0, 1, 2]]
                rgb = np.clip(rgb / max(1e-6, rgb.max()), 0, 1)
            else:
                rgb = np.repeat(imagery[:, :, 0:1], 3, axis=2)

            axes[0].imshow(rgb)
            axes[0].set_title("Original Satellite Imagery")
            axes[0].axis("off")

            axes[1].imshow(colored_output)
            axes[1].set_title(title)
            axes[1].axis("off")
        else:
            ax = axes[0] if isinstance(axes, np.ndarray) else axes
            ax.imshow(colored_output)
            ax.set_title(title)
            ax.axis("off")

        legend_elements = [
            mpatches.Patch(facecolor=colors[i], label=class_names[i])
            for i in sorted(colors.keys())
        ]

        if imagery is not None:
            axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
        else:
            ax = axes[0] if isinstance(axes, np.ndarray) else axes
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        fig_path = os.path.join(config.FIGURES_DIR, 'classification_result_ml.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved classification visualization: {fig_path}")
        plt.close()

        return fig_path

    # ---------------------------------------------------------
    # STAGE 1: Acquisition
    # ---------------------------------------------------------
    def run_stage_1_acquisition(self):
        """Stage 1: Data Acquisition"""
        print(f"\n{'='*70}")
        print("STAGE 1: DATA ACQUISITION")
        print(f"{'='*70}\n")

        from data_acquisition import LandsatDataAcquisition

        acq = LandsatDataAcquisition()
        imagery, metadata, stats = acq.run()

        if imagery is None:
            logger.error("Data acquisition failed")
            return None

        self.imagery = imagery
        self.stages['Data Acquisition'] = {'status': 'Complete', 'shape': imagery.shape}

        return imagery

    # ---------------------------------------------------------
    # STAGE 2: Feature Extraction
    # ---------------------------------------------------------
    def run_stage_2_feature_extraction(self):
        """Stage 2: Feature Extraction"""
        print(f"\n{'='*70}")
        print("STAGE 2: FEATURE EXTRACTION")
        print(f"{'='*70}\n")

        from feature_extraction import FeatureExtractor

        fe = FeatureExtractor(self.imagery)
        result = fe.run()

        # run() -> (features_2d, feature_names, stats, indices_dict)
        if isinstance(result, tuple):
            if len(result) == 4:
                features, feature_names, _, indices_dict = result
            elif len(result) == 3:
                features, feature_names, _ = result
                indices_dict = {}
            elif len(result) == 2:
                features, feature_names = result
                indices_dict = {}
            else:
                logger.error(f"Unexpected return size from FeatureExtractor.run(): {len(result)}")
                raise ValueError(f"Expected 2-4 values, got {len(result)}")
        else:
            logger.error("FeatureExtractor.run() did not return a tuple")
            raise TypeError("FeatureExtractor.run() must return a tuple")

        self.features = features
        self.feature_names = feature_names
        self.indices_dict = indices_dict

        logger.info(f"\nFeatures shape: {features.shape}")
        logger.info(f"Feature names: {feature_names}")
        if indices_dict:
            logger.info(f"Indices available for visualization: {list(indices_dict.keys())}")

        self.stages['Feature Extraction'] = {'status': 'Complete', 'shape': features.shape}

        return features, feature_names

    # ---------------------------------------------------------
    # STAGE 3: Label Generation
    # ---------------------------------------------------------
    def run_stage_3_labels(self):
        """Stage 3: Label Generation"""
        print(f"\n{'='*70}")
        print("STAGE 3: LABEL GENERATION")
        print(f"{'='*70}\n")

        n_samples = self.features.shape[0]
        n_classes = len(config.CLASS_NAMES) - 1

        ndvi_idx = self.feature_names.index('NDVI') if 'NDVI' in self.feature_names else 0
        ndvi = self.features[:, ndvi_idx]

        labels = np.zeros(n_samples, dtype=int)

        # 0 - Water
        labels[ndvi < -0.05] = 0
        # 3 - Urban / built-up
        labels[(ndvi >= -0.05) & (ndvi < 0.19)] = 3
        # 4 - Bare soil / sparse vegetation
        labels[(ndvi >= 0.19) & (ndvi < 0.25)] = 4
        # 2 - Grassland / moderate vegetation
        labels[(ndvi >= 0.25) & (ndvi < 0.35)] = 2
        # 1 - Forest / dense vegetation
        labels[ndvi >= 0.35] = 1

        noise_idx = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
        labels[noise_idx] = np.random.randint(0, n_classes, len(noise_idx))

        self.labels = labels

        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"\nClass distribution:")
        for class_idx, count in sorted(zip(unique, counts)):
            pct = 100 * count / n_samples
            class_name = config.CLASS_NAMES[class_idx] if class_idx < len(config.CLASS_NAMES) else f"Class {class_idx}"
            logger.info(f"  {class_name:<15}: {count:7,} ({pct:5.1f}%)")

        self.stages['Label Generation'] = {'status': 'Complete', 'n_samples': n_samples}

        return labels

    # ---------------------------------------------------------
    # STAGE 4: Traditional ML
    # ---------------------------------------------------------
    def run_stage_4_traditional_ml(self):
        """Stage 4: Traditional ML"""
        print(f"\n{'='*70}")
        print("STAGE 4: TRADITIONAL ML TRAINING")
        print(f"{'='*70}\n")

        from traditional_ml import TraditionalClassifiers

        clf = TraditionalClassifiers()
        X_train, X_test, y_train, y_test = clf.prepare_data(self.features, self.labels)
        clf.train_all(X_train, y_train, use_search=True)
        clf.evaluate_all(X_test, y_test)
        clf.visualize_confusion_matrices()
        clf.visualize_accuracy_comparison()
        clf.save_results()


        self.classifiers_ml = clf.classifiers
        self.metrics_ml = clf.metrics

        # RF feature importance figure
        if 'Random Forest' in clf.classifiers and clf.rf_feature_importances_ is not None:
            importances = clf.rf_feature_importances_
            names = self.feature_names

            idx = np.argsort(importances)[::-1][:15]
            top_names = [names[i] for i in idx]
            top_imps = importances[idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(idx)), top_imps, alpha=0.8)
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels(top_names)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title('Random Forest - Top 15 Feature Importance')
            ax.grid(True, axis='x', alpha=0.3)

            fi_path = os.path.join(config.FIGURES_DIR, 'rf_feature_importance.png')
            plt.tight_layout()
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"✓ RF feature importance saved: {fi_path}")

        # ML classification on full image
        logger.info(f"\nGenerating ML classification visualization...")
        img_height, img_width = self.imagery.shape[0], self.imagery.shape[1]

        if 'Random Forest' in clf.classifiers:
            best_clf_name = 'Random Forest'
            best_clf = clf.classifiers['Random Forest']
        else:
            best_clf_name = list(clf.classifiers.keys())[0]
            best_clf = list(clf.classifiers.values())[0]

        predictions_ml = best_clf.predict(self.features)
        self.predictions_ml = predictions_ml
        unique, counts = np.unique(predictions_ml, return_counts=True)
        for c, n in zip(unique, counts):
            print(f"class {c}: {n} px ({n/len(predictions_ml)*100:.1f}%)")
        print("predictions shape:", predictions_ml.shape)
        print("image H,W:", img_height, img_width)
        print("H*W:", img_height * img_width)

        unique, counts = np.unique(predictions_ml, return_counts=True)
        for c, n in zip(unique, counts):
            print("pred", c, n, f"{n/len(predictions_ml)*100:.1f}%")

        labels_2d_ml = predictions_ml.reshape(img_height, img_width)

        self.visualize_classification_result(
            labels_2d_ml,
            self.imagery,
            title=f"ML Classification Result ({best_clf_name})"
        )

        self.stages['Traditional ML'] = {'status': 'Complete', 'n_models': len(clf.classifiers)}

        return clf

    # ---------------------------------------------------------
    # STAGE 5: Deep Learning
    # ---------------------------------------------------------
    def run_stage_5_deep_learning(self):
        """Stage 5: Deep Learning - Semantic Segmentation"""
        print(f"\n{'='*70}")
        print("STAGE 5: DEEP LEARNING - SEMANTIC SEGMENTATION")
        print(f"{'='*70}\n")

        try:
            from deep_learning import DeepLearningModels, labels_to_rgb, CLASS_COLORS
            from sklearn.model_selection import train_test_split

            logger.info("\n" + "="*70)
            logger.info("PREPARING PATCH-BASED DATA FOR SEGMENTATION")
            logger.info("="*70)

            img_height, img_width = self.imagery.shape[0], self.imagery.shape[1]
            n_channels = self.imagery.shape[2]

            patch_size = 64
            stride = 32  # 50% overlap

            X_patches, y_patches = [], []
            labels_2d_full = self.labels.reshape(img_height, img_width)

            for i in range(0, img_height - patch_size + 1, stride):
                for j in range(0, img_width - patch_size + 1, stride):
                    patch_img = self.imagery[i:i+patch_size, j:j+patch_size, :]
                    patch_lab = labels_2d_full[i:i+patch_size, j:j+patch_size]
                    if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                        continue
                    X_patches.append(patch_img.astype(np.float32))
                    y_patches.append(patch_lab.astype(np.int32))

            if len(X_patches) == 0:
                logger.warning("Could not create patches from imagery, falling back to random demo patches...")
                for _ in range(4):
                    patch_img = np.random.rand(patch_size, patch_size, n_channels).astype(np.float32)
                    patch_lab = np.random.randint(0, len(config.CLASS_NAMES),
                                                  (patch_size, patch_size), dtype=np.int32)
                    X_patches.append(patch_img)
                    y_patches.append(patch_lab)

            X_patches = np.array(X_patches, dtype=np.float32)
            y_patches = np.array(y_patches, dtype=np.int32)

            X_train, X_val, y_train, y_val = train_test_split(
                X_patches, y_patches,
                test_size=0.2,
                random_state=config.RANDOM_STATE
            )

            num_classes = len(config.CLASS_NAMES)
            dl = DeepLearningModels(num_classes=num_classes)

            # Train models
            dl.train_unet(X_train, y_train, X_val, y_val, epochs=15, batch_size=8)
            dl.train_cnn(X_train, y_train, X_val, y_val, epochs=15, batch_size=8)

            # Directory for DL figures
            save_dir = os.path.join(config.FIGURES_DIR, "deep_learning")
            os.makedirs(save_dir, exist_ok=True)

            # Full-image prediction
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            img = self.imagery.astype(np.float32)

            # VISUALIZACIÓN: solo RGB (primeras 3 bandas)
            rgb = img[:, :, :3]
            rgb_disp = rgb / max(1e-6, rgb.max())

            # PREDICCIÓN: usar todas las bandas
            X_full = np.expand_dims(img, axis=0)

            unet_model = dl.models_dl["U-Net"]
            cnn_model = dl.models_dl["CNN"]

            unet_pred = np.argmax(unet_model.predict(X_full, verbose=0)[0], axis=-1)
            cnn_pred = np.argmax(cnn_model.predict(X_full, verbose=0)[0], axis=-1)

            self.unet_pred = unet_pred

            unet_rgb = labels_to_rgb(unet_pred)
            cnn_rgb = labels_to_rgb(cnn_pred)
            gt_rgb = labels_to_rgb(labels_2d_full)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes[0, 0].imshow(rgb_disp);  axes[0, 0].set_title("Original (RGB)"); axes[0, 0].axis("off")
            axes[0, 1].imshow(gt_rgb);    axes[0, 1].set_title("Ground Truth");   axes[0, 1].axis("off")
            axes[1, 0].imshow(unet_rgb);  axes[1, 0].set_title("U-Net");          axes[1, 0].axis("off")
            axes[1, 1].imshow(cnn_rgb);   axes[1, 1].set_title("CNN");            axes[1, 1].axis("off")

            legend_elements = [
                mpatches.Patch(facecolor=CLASS_COLORS[i], label=config.CLASS_NAMES[i])
                for i in sorted(CLASS_COLORS.keys())
            ]
            fig.legend(handles=legend_elements, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0))
            plt.tight_layout()
            full_pred_path = os.path.join(save_dir, "dl_full_image_predictions.png")
            plt.savefig(full_pred_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved full-image predictions to {full_pred_path}")

            # Other DL figures
            dl.plot_predictions(save_dir)
            dl.plot_confusion_matrices(save_dir)
            dl.plot_metrics_comparison(save_dir)
            dl.plot_training_history(save_dir)
            dl.save_results(save_dir)

            self.dl_models = dl.models_dl
            self.metrics_dl = dl.metrics_dl
            self.stages["Deep Learning"] = {"status": "Complete", "n_models": len(dl.models_dl)}

        except ImportError as e:
            logger.warning(f"Deep learning imports failed - skipping deep learning: {str(e)}")
            self.stages["Deep Learning"] = {"status": "Skipped", "reason": str(e)}
        except Exception as e:
            logger.error(f"Deep learning stage failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.stages["Deep Learning"] = {"status": "Failed", "error": str(e)}

    # ---------------------------------------------------------
    # STAGE 6: Evaluation
    # ---------------------------------------------------------
    def run_stage_6_evaluation(self):
        """Stage 6: Evaluation"""
        print(f"\n{'='*70}")
        print("STAGE 6: EVALUATION")
        print(f"{'='*70}\n")

        from evaluation import ClassificationEvaluator

        evaluator = ClassificationEvaluator()

        # Numeric summary
        if hasattr(self, 'metrics_ml') and self.metrics_ml:
            logger.info("Generating summary report from in-memory ML metrics...")
            evaluator.generate_summary_report(self.metrics_ml)
        else:
            logger.info("No ML metrics available, skipping evaluation summary.")

        # Visualizations
        predictions_ml = getattr(self, "predictions_ml", None)
        indices_dict = getattr(self, "indices_dict", {})
        unet_pred = getattr(self, "unet_pred", None)

        if hasattr(self, "imagery") and hasattr(self, "features") and hasattr(self, "labels"):
            evaluator.generate_visualizations(
                imagery=self.imagery,
                features=self.features,
                feature_names=self.feature_names,
                labels_1d=self.labels,
                predictions_ml_1d=predictions_ml,
                indices_dict=indices_dict,
                band_names=getattr(config, "DOWNLOAD_BANDS", None),
                rgb_indices=(0, 1, 2),  # red, green, blue indices in imagery
                unet_prediction_2d=unet_pred
            )
        else:
            logger.warning("Not enough data in pipeline to generate visualizations in evaluation stage.")

        self.evaluator = evaluator
        self.stages['Evaluation'] = {'status': 'Complete'}

    # ---------------------------------------------------------
    # SUMMARY & RUN
    # ---------------------------------------------------------
    def print_summary(self):
        """Print pipeline summary"""
        print(f"\n{'='*70}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*70}\n")

        elapsed_time = time.time() - self.start_time

        print(f"Total Execution Time: {elapsed_time/60:.1f} minutes")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for stage, info in self.stages.items():
            print(f"{stage}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

        print(f"\n{'='*70}")
        print("✓ PIPELINE COMPLETE")
        print(f"✓ Classification results saved to: {config.FIGURES_DIR}")
        print(f"{'='*70}\n")

    def run(self):
        """Execute complete pipeline"""
        try:
            self.run_stage_1_acquisition()
            self.run_stage_2_feature_extraction()
            self.run_stage_3_labels()
            self.run_stage_4_traditional_ml()
            self.run_stage_5_deep_learning()
            self.run_stage_6_evaluation()
            self.print_summary()

            logger.info("✓ All stages completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution"""
        # noqa
    pipeline = LandCoverClassificationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
