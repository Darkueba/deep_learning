"""
MODULE 7: MAIN PIPELINE
Orchestrates entire workflow from data acquisition to evaluation
"""

import numpy as np
import os
import time
from datetime import datetime

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class LandCoverClassificationPipeline:
    """
    Complete end-to-end pipeline for land cover classification
    """
    
    def __init__(self):
        """Initialize pipeline"""
        print("\n" + "="*70)
        print("LAND COVER CLASSIFICATION - MAIN PIPELINE")
        print("="*70)
        
        self.start_time = time.time()
        self.stages = {}
        
        # Create output directories
        for directory in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR,
                         config.TRAINING_DATA_DIR, config.RESULTS_DIR, config.FIGURES_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def run_data_acquisition(self):
        """Stage 1: Download and visualize satellite data"""
        print(f"\n{'='*70}")
        print("STAGE 1: DATA ACQUISITION")
        print(f"{'='*70}\n")
        
        from data_acquisition import LandsatDataAcquisition
        
        acq = LandsatDataAcquisition()
        data, metadata, stats = acq.run()
        
        self.imagery = data
        self.metadata = metadata
        self.stats = stats
        
        self.stages['Data Acquisition'] = {
            'status': 'Complete',
            'output_shape': data.shape
        }
        
        return data
    
    def run_feature_extraction(self):
        """Stage 2: Extract features from imagery"""
        print(f"\n{'='*70}")
        print("STAGE 2: FEATURE EXTRACTION")
        print(f"{'='*70}\n")
        
        from feature_extraction import FeatureExtractor
        
        fe = FeatureExtractor(self.imagery)
        features_2d, feature_names, feature_stats = fe.run()
        
        self.features = features_2d
        self.feature_names = feature_names
        self.feature_stats = feature_stats
        
        self.stages['Feature Extraction'] = {
            'status': 'Complete',
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        
        return features_2d, feature_names
    
    def generate_labels(self):
        """Stage 3: Generate or load labels"""
        print(f"\n{'='*70}")
        print("STAGE 3: LABEL GENERATION")
        print(f"{'='*70}\n")
        
        # Generate synthetic labels for demonstration
        # In production, use real labels from ground truth
        
        print("Generating synthetic labels for demonstration...")
        n_samples = self.features.shape[0]
        n_classes = len(config.CLASS_NAMES) - 1  # Exclude clouds
        
        # Create semi-realistic labels based on NDVI
        ndvi_idx = self.feature_names.index('NDVI') if 'NDVI' in self.feature_names else 0
        ndvi = self.features[:, ndvi_idx]
        
        labels = np.zeros(n_samples, dtype=int)
        
        # Water: low NDVI
        labels[ndvi < -0.3] = 0
        # Urban: medium-low NDVI
        labels[(ndvi >= -0.3) & (ndvi < 0.2)] = 3
        # Bare soil: low-medium NDVI
        labels[(ndvi >= 0.2) & (ndvi < 0.4)] = 4
        # Grassland: medium NDVI
        labels[(ndvi >= 0.4) & (ndvi < 0.6)] = 2
        # Forest: high NDVI
        labels[ndvi >= 0.6] = 1
        
        # Add some noise
        noise_idx = np.random.choice(n_samples, int(0.1*n_samples), replace=False)
        labels[noise_idx] = np.random.randint(0, n_classes, len(noise_idx))
        
        self.labels = labels
        
        print(f"✓ Generated {n_samples} labels")
        print(f"  Classes represented: {np.unique(labels)}")
        
        self.stages['Label Generation'] = {
            'status': 'Complete',
            'n_samples': n_samples,
            'n_classes': n_classes
        }
        
        return labels
    
    def run_traditional_ml(self):
        """Stage 4: Train traditional ML classifiers"""
        print(f"\n{'='*70}")
        print("STAGE 4: TRADITIONAL MACHINE LEARNING")
        print(f"{'='*70}\n")
        
        from traditional_ml import TraditionalClassifiers
        
        clf = TraditionalClassifiers()
        X_train, X_test, y_train, y_test = clf.prepare_data(self.features, self.labels)
        clf.train_all(X_train, y_train)
        metrics = clf.evaluate_all(X_test, y_test)
        
        # Visualizations
        clf.visualize_confusion_matrices()
        clf.visualize_accuracy_comparison()
        clf.save_results()
        clf.print_summary()
        
        self.classifiers_ml = clf.classifiers
        self.metrics_ml = metrics
        
        self.stages['Traditional ML'] = {
            'status': 'Complete',
            'n_classifiers': len(clf.classifiers),
            'best_accuracy': max([m['Overall Accuracy'] for m in metrics.values()])
        }
        
        return clf, metrics
    
    def run_deep_learning(self):
        """Stage 5: Train deep learning models (if TensorFlow available)"""
        print(f"\n{'='*70}")
        print("STAGE 5: DEEP LEARNING")
        print(f"{'='*70}\n")
        
        try:
            from deep_learning import DeepLearningModels
            from sklearn.model_selection import train_test_split
            
            dl = DeepLearningModels()
            
            # Prepare patch data
            X_patches, y_patches = dl.prepare_patch_data(
                self.features, self.labels,
                patch_size=config.IMG_HEIGHT
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_patches, y_patches,
                test_size=0.2,
                random_state=config.RANDOM_STATE
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=config.RANDOM_STATE
            )
            
            input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, len(self.feature_names))
            num_classes = len(config.CLASS_NAMES) - 1
            
            # Train CNN
            cnn = dl.build_cnn_model(input_shape, num_classes)
            history_cnn = dl.train_model('CNN', cnn, X_train, y_train, X_val, y_val)
            dl.visualize_training_history('CNN', history_cnn)
            metrics_cnn = dl.evaluate_model('CNN', cnn, X_test, y_test)
            
            # Train U-Net
            unet = dl.build_unet_model(input_shape, num_classes)
            history_unet = dl.train_model('U-Net', unet, X_train, y_train, X_val, y_val)
            dl.visualize_training_history('U-Net', history_unet)
            metrics_unet = dl.evaluate_model('U-Net', unet, X_test, y_test)
            
            # Save models
            dl.save_models()
            
            self.models_dl = dl.models
            self.metrics_dl = {'CNN': metrics_cnn, 'U-Net': metrics_unet}
            
            self.stages['Deep Learning'] = {
                'status': 'Complete',
                'n_models': 2,
                'best_accuracy': max([m['Overall Accuracy'] for m in self.metrics_dl.values()])
            }
            
        except ImportError:
            print("⚠ TensorFlow not available - skipping deep learning")
            self.stages['Deep Learning'] = {
                'status': 'Skipped',
                'reason': 'TensorFlow not installed'
            }
        
        return None
    
    def run_evaluation(self):
        """Stage 6: Comprehensive evaluation"""
        print(f"\n{'='*70}")
        print("STAGE 6: EVALUATION & COMPARISON")
        print(f"{'='*70}\n")
        
        from evaluation import ClassificationEvaluator
        
        evaluator = ClassificationEvaluator()
        
        # Evaluate ML classifiers
        for clf_name, clf in self.classifiers_ml.items():
            y_pred = clf.predict(self.features)
            evaluator.evaluate_classification(self.labels, y_pred, clf_name)
        
        # Create comparison table and visualizations
        comparison_df = evaluator.create_comparison_table()
        evaluator.visualize_metrics_comparison()
        evaluator.visualize_confusion_matrices()
        evaluator.create_per_class_report()
        evaluator.generate_summary_report()
        
        self.evaluator = evaluator
        
        self.stages['Evaluation'] = {
            'status': 'Complete',
            'n_models_evaluated': len(self.classifiers_ml)
        }
        
        return evaluator
    
    def print_pipeline_summary(self):
        """Print final summary"""
        print(f"\n{'='*70}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*70}\n")
        
        elapsed_time = time.time() - self.start_time
        
        print(f"Execution Time: {elapsed_time/60:.1f} minutes")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for stage, info in self.stages.items():
            print(f"{stage}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        print(f"\n{'='*70}")
        print("✓ PIPELINE COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"Output files saved to:")
        print(f"  Data: {config.RAW_DATA_DIR}/")
        print(f"  Features: {config.PROCESSED_DATA_DIR}/")
        print(f"  Models: {config.RESULTS_DIR}/")
        print(f"  Figures: {config.FIGURES_DIR}/")
    
    def run(self):
        """Execute complete pipeline"""
        try:
            # Stage 1: Data Acquisition
            self.run_data_acquisition()
            
            # Stage 2: Feature Extraction
            self.run_feature_extraction()
            
            # Stage 3: Label Generation
            self.generate_labels()
            
            # Stage 4: Traditional ML
            self.run_traditional_ml()
            
            # Stage 5: Deep Learning
            self.run_deep_learning()
            
            # Stage 6: Evaluation
            self.run_evaluation()
            
            # Summary
            self.print_pipeline_summary()
            
            print("\n✓ All stages completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution"""
    pipeline = LandCoverClassificationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
