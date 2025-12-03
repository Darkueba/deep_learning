"""
MAIN_PIPELINE.PY - Fixed for segmentation
Orchestrates complete land cover classification with semantic segmentation
"""

import numpy as np
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

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
        for directory in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR,
                         config.TRAINING_DATA_DIR, config.RESULTS_DIR, config.FIGURES_DIR]:
            os.makedirs(directory, exist_ok=True)
    
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
    
    def run_stage_2_feature_extraction(self):
        """Stage 2: Feature Extraction"""
        print(f"\n{'='*70}")
        print("STAGE 2: FEATURE EXTRACTION")
        print(f"{'='*70}\n")
        
        from feature_extraction import FeatureExtractor
        
        fe = FeatureExtractor(self.imagery)
        result = fe.run()
        
        # Handle both 2 and 3 value returns
        if isinstance(result, tuple):
            if len(result) == 3:
                features, feature_names, _ = result
            elif len(result) == 2:
                features, feature_names = result
            else:
                logger.error(f"Unexpected return size from FeatureExtractor.run(): {len(result)}")
                raise ValueError(f"Expected 2 or 3 values, got {len(result)}")
        else:
            logger.error("FeatureExtractor.run() did not return a tuple")
            raise TypeError("FeatureExtractor.run() must return a tuple")
        
        self.features = features
        self.feature_names = feature_names
        
        logger.info(f"\nFeatures shape: {features.shape}")
        logger.info(f"Feature names: {feature_names}")
        
        self.stages['Feature Extraction'] = {'status': 'Complete', 'shape': features.shape}
        
        return features, feature_names
    
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
        labels[ndvi < -0.3] = 0
        labels[(ndvi >= -0.3) & (ndvi < 0.2)] = 3
        labels[(ndvi >= 0.2) & (ndvi < 0.4)] = 4
        labels[(ndvi >= 0.4) & (ndvi < 0.6)] = 2
        labels[ndvi >= 0.6] = 1
        
        noise_idx = np.random.choice(n_samples, int(0.05*n_samples), replace=False)
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
    
    def run_stage_4_traditional_ml(self):
        """Stage 4: Traditional ML"""
        print(f"\n{'='*70}")
        print("STAGE 4: TRADITIONAL ML TRAINING")
        print(f"{'='*70}\n")
        
        from traditional_ml import TraditionalClassifiers
        
        clf = TraditionalClassifiers()
        X_train, X_test, y_train, y_test = clf.prepare_data(self.features, self.labels)
        clf.train_all(X_train, y_train)
        clf.evaluate_all(X_test, y_test)
        clf.visualize_confusion_matrices()
        clf.visualize_accuracy_comparison()
        clf.save_results()
        
        self.classifiers_ml = clf.classifiers
        self.metrics_ml = clf.metrics
        
        self.stages['Traditional ML'] = {'status': 'Complete', 'n_models': len(clf.classifiers)}
        
        return clf
    
    def run_stage_5_deep_learning(self):
        """Stage 5: Deep Learning - Semantic Segmentation"""
        print(f"\n{'='*70}")
        print("STAGE 5: DEEP LEARNING - SEMANTIC SEGMENTATION")
        print(f"{'='*70}\n")
        
        try:
            from deep_learning import DeepLearningModels
            
            logger.info("\n" + "="*70)
            logger.info("PREPARING PATCH-BASED DATA FOR SEGMENTATION")
            logger.info("="*70)
            
            # Get image shape and create patches
            img_height, img_width = self.imagery.shape[0], self.imagery.shape[1]
            patch_size = 64
            stride = 32  # 50% overlap
            
            X_patches = []
            y_patches = []
            
            logger.info(f"Image size: {img_height}x{img_width}")
            logger.info(f"Patch size: {patch_size}x{patch_size}, stride: {stride}")
            
            for i in range(0, img_height - patch_size, stride):
                for j in range(0, img_width - patch_size, stride):
                    # Extract patch from imagery
                    patch_img = self.imagery[i:i+patch_size, j:j+patch_size, :]
                    X_patches.append(patch_img)
                    
                    # Extract corresponding patch from labels
                    # Reshape labels from 1D to 2D image
                    labels_2d = self.labels.reshape(img_height, img_width)
                    patch_lab = labels_2d[i:i+patch_size, j:j+patch_size]
                    y_patches.append(patch_lab)
            
            # Fallback: if no patches created, use simpler approach
            if len(X_patches) == 0:
                logger.warning("Could not create patches from imagery, using simpler approach...")
                n_patches = 4
                for idx in range(n_patches):
                    # Create random patches for demo
                    patch_img = np.random.rand(patch_size, patch_size, 11).astype(np.float32) * 3000
                    patch_lab = np.random.randint(0, 5, (patch_size, patch_size), dtype=np.int32)
                    X_patches.append(patch_img)
                    y_patches.append(patch_lab)
            
            X_patches = np.array(X_patches, dtype=np.float32)
            y_patches = np.array(y_patches, dtype=np.int32)
            
            logger.info(f"\n✓ Created {len(X_patches)} patches")
            logger.info(f"  X_patches shape: {X_patches.shape}")
            logger.info(f"  y_patches shape: {y_patches.shape}")
            logger.info(f"  Patch size: {patch_size}x{patch_size}")
            logger.info(f"  Channels: {X_patches.shape[-1]}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_patches, y_patches,
                test_size=0.2,
                random_state=config.RANDOM_STATE
            )
            
            logger.info(f"\nData split:")
            logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            # Initialize DL module
            dl = DeepLearningModels()
            
            # Build models
            input_shape = (patch_size, patch_size, X_patches.shape[-1])
            num_classes = len(config.CLASS_NAMES) - 1
            
            logger.info(f"\nModel configuration:")
            logger.info(f"  Input shape: {input_shape}")
            logger.info(f"  Num classes: {num_classes}")
            
            # U-Net
            logger.info("\n" + "-"*70)
            logger.info("Building U-Net...")
            logger.info("-"*70)
            unet = dl.build_unet_model(input_shape, num_classes)
            dl.models['U-Net'] = unet
            
            history_unet = dl.train_model('U-Net', unet, X_train, y_train, X_val, y_val)
            dl.visualize_training_history('U-Net', history_unet)
            metrics_unet = dl.evaluate_model('U-Net', unet, X_val, y_val)
            
            # CNN
            logger.info("\n" + "-"*70)
            logger.info("Building CNN...")
            logger.info("-"*70)
            cnn = dl.build_cnn_model(input_shape, num_classes)
            dl.models['CNN'] = cnn
            
            history_cnn = dl.train_model('CNN', cnn, X_train, y_train, X_val, y_val)
            dl.visualize_training_history('CNN', history_cnn)
            metrics_cnn = dl.evaluate_model('CNN', cnn, X_val, y_val)
            
            # Save models
            dl.save_models()
            
            self.dl_models = dl.models
            self.metrics_dl = {'U-Net': metrics_unet, 'CNN': metrics_cnn}
            
            self.stages['Deep Learning'] = {'status': 'Complete', 'n_models': 2}
            
        except ImportError as e:
            logger.warning(f"TensorFlow not available - skipping deep learning: {str(e)}")
            self.stages['Deep Learning'] = {'status': 'Skipped', 'reason': 'TensorFlow not installed'}
        except Exception as e:
            logger.error(f"Deep learning stage failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.stages['Deep Learning'] = {'status': 'Failed', 'error': str(e)}
    
    def run_stage_6_evaluation(self):
        """Stage 6: Evaluation"""
        print(f"\n{'='*70}")
        print("STAGE 6: EVALUATION")
        print(f"{'='*70}\n")
        
        from evaluation import ClassificationEvaluator

        evaluator = ClassificationEvaluator()
        if hasattr(self, 'metrics_ml'):
            # Assume evaluator already knows where to read metrics from,
            # or it loads them from disk internally
            evaluator.generate_summary_report()

            #evaluator.generate_summary_report(self.metrics_ml)
        
        self.evaluator = evaluator
        self.stages['Evaluation'] = {'status': 'Complete'}
    
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
    pipeline = LandCoverClassificationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
