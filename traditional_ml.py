"""
MODULE 3: TRADITIONAL MACHINE LEARNING
Trains and evaluates traditional classifiers (RF, SVM, k-NN, LDA)
"""

import numpy as np
import os
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class TraditionalClassifiers:
    """
    Trains and evaluates traditional machine learning classifiers
    Classifiers: Random Forest, SVM, k-NN, LDA
    """
    
    def __init__(self):
        """Initialize classifier collection"""
        print("\n" + "="*70)
        print("TRADITIONAL ML MODULE")
        print("="*70)
        
        self.classifiers = {}
        self.predictions = {}
        self.metrics = {}
        self.confusion_matrices = {}
        
        # Create output directory
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def prepare_data(self, features, labels):
        """
        Prepare data for training
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature matrix (N, num_features)
        labels : numpy.ndarray
            Label vector (N,)
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        print(f"\n{'='*70}")
        print("DATA PREPARATION")
        print(f"{'='*70}")
        X = features
        y = labels

        # --------------------------------------------------
        # OPTIONAL: downsample majority classes BEFORE split
        # --------------------------------------------------
        import numpy as np

        def balance_downsample(X_in, y_in, max_per_class=40000):
            X_list, y_list = [], []
            for c in np.unique(y_in):
                idx = np.where(y_in == c)[0]
                if len(idx) > max_per_class:
                    idx = np.random.choice(idx, max_per_class, replace=False)
                X_list.append(X_in[idx])
                y_list.append(y_in[idx])
            return np.vstack(X_list), np.concatenate(y_list)

        X, y = balance_downsample(X, y, max_per_class=40000)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=labels  # Stratified split
        )
        
        # Standardize features (important for SVM and k-NN)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Class distribution
        print(f"\nClass distribution (training):")
        for class_idx in np.unique(y_train):
            count = np.sum(y_train == class_idx)
            pct = 100 * count / len(y_train)
            print(f"  Class {class_idx}: {count:6d} samples ({pct:5.1f}%)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: RANDOM FOREST")
        print(f"{'='*70}")
        
        params = config.CLASSIFIER_PARAMS['random_forest']
        print(f"\nParameters: {params}")
        
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        self.classifiers['Random Forest'] = rf
        
        # Get feature importance
        feature_importance = rf.feature_importances_
        print(f"✓ Training complete")
        
        return rf, feature_importance
    
    def train_svm(self, X_train, y_train):
        """Train Support Vector Machine classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: SUPPORT VECTOR MACHINE (SVM)")
        print(f"{'='*70}")
        
        params = config.CLASSIFIER_PARAMS['svm']
        print(f"\nParameters: {params}")
        
        svm = SVC(**params)
        svm.fit(X_train, y_train)
        
        self.classifiers['SVM'] = svm
        print(f"✓ Training complete")
        
        return svm
    
    def train_knn(self, X_train, y_train):
        """Train k-Nearest Neighbors classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: k-NEAREST NEIGHBORS (k-NN)")
        print(f"{'='*70}")
        
        params = config.CLASSIFIER_PARAMS['knn']
        print(f"\nParameters: {params}")
        
        knn = KNeighborsClassifier(**params)
        knn.fit(X_train, y_train)
        
        self.classifiers['k-NN'] = knn
        print(f"✓ Training complete")
        
        return knn
    
    def train_lda(self, X_train, y_train):
        """Train Linear Discriminant Analysis classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: LINEAR DISCRIMINANT ANALYSIS (LDA)")
        print(f"{'='*70}")
        
        params = config.CLASSIFIER_PARAMS['mlc']
        print(f"\nParameters: {params}")
        
        lda = LinearDiscriminantAnalysis(**params)
        lda.fit(X_train, y_train)
        
        self.classifiers['LDA'] = lda
        print(f"✓ Training complete")
        
        return lda
    
    def evaluate_classifier(self, clf_name, clf, X_test, y_test):
        """
        Evaluate single classifier
        
        Parameters:
        -----------
        clf_name : str
            Classifier name
        clf : sklearn classifier
            Trained classifier
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test labels
        
        Returns:
        --------
        dict : Metrics dictionary
        """
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Metrics
        oa = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'Overall Accuracy': oa,
            'Kappa': kappa,
            'Precision (Macro)': precision,
            'Recall (Macro)': recall,
            'F1-Score (Macro)': f1,
            'F1-Score (Weighted)': f1_weighted,
            'Confusion Matrix': cm,
            'Classification Report': class_report,
            'Predictions': y_pred
        }
        
        self.metrics[clf_name] = metrics
        self.confusion_matrices[clf_name] = cm
        self.predictions[clf_name] = y_pred
        
        return metrics
    
    def train_all(self, X_train, y_train):
        """
        Train all classifiers
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        """
        print(f"\n{'='*70}")
        print("TRAINING ALL CLASSIFIERS")
        print(f"{'='*70}\n")
        
        # Random Forest
        self.train_random_forest(X_train, y_train)
        
        # SVM
        #self.train_svm(X_train, y_train)
        
        # k-NN
        self.train_knn(X_train, y_train)
        
        # LDA
        self.train_lda(X_train, y_train)
        
        print(f"\n{'='*70}")
        print(f"✓ ALL CLASSIFIERS TRAINED")
        print(f"{'='*70}\n")
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all classifiers
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test labels
        
        Returns:
        --------
        dict : Results dictionary
        """
        print(f"\n{'='*70}")
        print("EVALUATING ALL CLASSIFIERS")
        print(f"{'='*70}\n")
        
        for clf_name, clf in self.classifiers.items():
            print(f"\nEvaluating: {clf_name}...")
            metrics = self.evaluate_classifier(clf_name, clf, X_test, y_test)
            
            print(f"  Overall Accuracy: {metrics['Overall Accuracy']:.4f}")
            print(f"  Kappa:            {metrics['Kappa']:.4f}")
            print(f"  F1-Score:         {metrics['F1-Score (Macro)']:.4f}")
        
        return self.metrics
    
    def visualize_confusion_matrices(self):
        """Create confusion matrix visualizations"""
        print(f"\n{'='*70}")
        print("VISUALIZING CONFUSION MATRICES")
        print(f"{'='*70}\n")
        
        n_classifiers = len(self.confusion_matrices)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        fig.suptitle('Confusion Matrices - Traditional ML', fontsize=14)
        
        for idx, (clf_name, cm) in enumerate(self.confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(clf_name)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, '04_confusion_matrices_ml.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices saved: {fig_path}")
        plt.close()
    
    def visualize_accuracy_comparison(self):
        """Create accuracy comparison chart"""
        print(f"\nVisualizing accuracy comparison...")
        
        classifiers = list(self.metrics.keys())
        accuracies = [self.metrics[clf]['Overall Accuracy'] for clf in classifiers]
        kappas = [self.metrics[clf]['Kappa'] for clf in classifiers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(classifiers))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Overall Accuracy', alpha=0.8)
        ax.bar(x + width/2, kappas, width, label='Kappa Coefficient', alpha=0.8)
        
        ax.set_xlabel('Classifier')
        ax.set_ylabel('Score')
        ax.set_title('Traditional ML Classifier Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, '05_ml_accuracy_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Accuracy comparison saved: {fig_path}")
        plt.close()
    
    def save_results(self):
        """Save results to CSV"""
        print(f"\nSaving results...")
        
        # Create results table
        results_data = []
        for clf_name, metrics in self.metrics.items():
            results_data.append({
                'Classifier': clf_name,
                'Overall Accuracy': metrics['Overall Accuracy'],
                'Kappa': metrics['Kappa'],
                'Precision (Macro)': metrics['Precision (Macro)'],
                'Recall (Macro)': metrics['Recall (Macro)'],
                'F1-Score (Macro)': metrics['F1-Score (Macro)'],
                'F1-Score (Weighted)': metrics['F1-Score (Weighted)']
            })
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(results_data)
        results_path = os.path.join(config.RESULTS_DIR, 'traditional_ml_results.csv')
        df.to_csv(results_path, index=False)
        print(f"✓ Results saved: {results_path}")
        
        # Save models
        models_dir = os.path.join(config.RESULTS_DIR, 'traditional_ml_models')
        os.makedirs(models_dir, exist_ok=True)
        
        for clf_name, clf in self.classifiers.items():
            model_path = os.path.join(models_dir, f'{clf_name.replace(" ", "_").lower()}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)
            print(f"✓ Model saved: {model_path}")
    
    def print_summary(self):
        """Print summary of results"""
        print(f"\n{'='*70}")
        print("TRADITIONAL ML SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"{'Classifier':<20} {'OA':<10} {'Kappa':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for clf_name, metrics in self.metrics.items():
            print(f"{clf_name:<20} {metrics['Overall Accuracy']:<10.4f} "
                  f"{metrics['Kappa']:<10.4f} {metrics['F1-Score (Macro)']:<10.4f}")
        
        # Best classifier
        best_clf = max(self.metrics.items(), key=lambda x: x[1]['Overall Accuracy'])
        print(f"\n✓ Best classifier: {best_clf[0]} ({best_clf[1]['Overall Accuracy']:.4f} OA)")


def main():
    """Example usage"""
    from data_acquisition import LandsatDataAcquisition
    from feature_extraction import FeatureExtractor
    
    # Get data
    print("Loading data...")
    acq = LandsatDataAcquisition()
    data, _, _ = acq.run()
    
    # Extract features
    print("\nExtracting features...")
    fe = FeatureExtractor(data)
    features, names, _ = fe.run()
    
    # Generate synthetic labels for demo
    np.random.seed(42)
    labels = np.random.randint(0, len(config.CLASS_NAMES)-1, size=features.shape[0])
    
    # Train classifiers
    clf = TraditionalClassifiers()
    X_train, X_test, y_train, y_test = clf.prepare_data(features, labels)
    clf.train_all(X_train, y_train)
    metrics = clf.evaluate_all(X_test, y_test)
    
    # Visualize and save
    clf.visualize_confusion_matrices()
    clf.visualize_accuracy_comparison()
    clf.save_results()
    clf.print_summary()


if __name__ == "__main__":
    main()
