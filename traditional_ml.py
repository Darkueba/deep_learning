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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)

# Use non-interactive backend to avoid tkinter threading issues with parallel jobs
import matplotlib
matplotlib.use('Agg')
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
        self.best_params_ = {}

        # Best trackers
        self.best_by_f1_ = None
        self.best_by_oa_ = None

        # RF feature importances for visualization
        self.rf_feature_importances_ = None

        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    def prepare_data(self, features, labels):
        """
        Prepare data for training
        """
        print(f"\n{'='*70}")
        print("DATA PREPARATION")
        print(f"{'='*70}")
        X = features
        y = labels

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")

        print(f"\nClass distribution (training):")
        for class_idx in np.unique(y_train):
            count = np.sum(y_train == class_idx)
            pct = 100 * count / len(y_train)
            print(f"  Class {class_idx}: {count:6d} samples ({pct:5.1f}%)")

        return X_train_scaled, X_test_scaled, y_train, y_test

    # ------------------------------------------------------------------
    # RandomizedSearchCV (ligero)
    # ------------------------------------------------------------------
    def _run_random_search(
        self,
        base_estimator,
        param_distributions,
        X_train,
        y_train,
        scoring="f1_macro",
        cv=3,
        n_iter=10,
        verbose=1,
    ):
        """
        Run a LIGHT RandomizedSearchCV and return best estimator and params.
        """
        print("\nRunning RandomizedSearchCV (light)...")
        print(f"  Scoring: {scoring}")
        print(f"  n_iter:  {n_iter}")
        print(f"  cv:      {cv}")

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=2,
            random_state=config.RANDOM_STATE,
            verbose=verbose,
        )
        search.fit(X_train, y_train)
        print(f"✓ Best params: {search.best_params_}")
        print(f"✓ Best CV score ({scoring}): {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_

    # ------------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------------
    def train_random_forest(self, X_train, y_train, use_search=True):
        """Train Random Forest classifier (with optional light RandomizedSearchCV)"""
        print(f"\n{'='*70}")
        print("TRAINING: RANDOM FOREST")
        print(f"{'='*70}")

        base_params = config.CLASSIFIER_PARAMS['random_forest']
        print(f"\nBase parameters: {base_params}")

        if use_search:
            # espacio reducido
            param_distributions = {
                "n_estimators": [100, 150, 200],
                "max_depth": [None, 15, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False],
            }
            rf_base = RandomForestClassifier(**base_params)
            rf, best_params = self._run_random_search(
                rf_base,
                param_distributions,
                X_train,
                y_train,
                scoring="f1_macro",
                cv=3,
                n_iter=10,
            )
            self.best_params_["Random Forest"] = best_params
        else:
            rf = RandomForestClassifier(**base_params)
            rf.fit(X_train, y_train)

        self.classifiers['Random Forest'] = rf

        # Store feature importance
        self.rf_feature_importances_ = rf.feature_importances_
        print("✓ RF training complete")

        return rf, self.rf_feature_importances_

    # ------------------------------------------------------------------
    # SVM
    # ------------------------------------------------------------------
    def train_svm(self, X_train, y_train, use_search=True):
        """Train Support Vector Machine classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: SUPPORT VECTOR MACHINE (SVM)")
        print(f"{'='*70}")

        base_params = config.CLASSIFIER_PARAMS['svm']
        print(f"\nBase parameters: {base_params}")

        if use_search:
            param_distributions = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear"],
            }
            filtered_base = {
                k: v for k, v in base_params.items()
                if k not in param_distributions
            }
            svm_base = SVC(**filtered_base)
            svm, best_params = self._run_random_search(
                svm_base,
                param_distributions,
                X_train,
                y_train,
                scoring="f1_macro",
                cv=3,
                n_iter=10,
            )
            self.best_params_["SVM"] = best_params
        else:
            svm = SVC(**base_params)
            svm.fit(X_train, y_train)

        self.classifiers['SVM'] = svm
        print("✓ SVM training complete")
        return svm

    # ------------------------------------------------------------------
    # k-NN
    # ------------------------------------------------------------------
    def train_knn(self, X_train, y_train, use_search=True):
        """Train k-Nearest Neighbors classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: k-NEAREST NEIGHBORS (k-NN)")
        print(f"{'='*70}")

        base_params = config.CLASSIFIER_PARAMS['knn']
        print(f"\nBase parameters: {base_params}")

        if use_search:
            param_distributions = {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "p": [1, 2],  # 1: manhattan, 2: euclidean
            }
            filtered_base = {
                k: v for k, v in base_params.items()
                if k not in param_distributions
            }
            knn_base = KNeighborsClassifier(**filtered_base)
            knn, best_params = self._run_random_search(
                knn_base,
                param_distributions,
                X_train,
                y_train,
                scoring="f1_macro",
                cv=3,
                n_iter=10,
            )
            self.best_params_["k-NN"] = best_params
        else:
            knn = KNeighborsClassifier(**base_params)
            knn.fit(X_train, y_train)

        self.classifiers['k-NN'] = knn
        print("✓ k-NN training complete")
        return knn

    # ------------------------------------------------------------------
    # LDA
    # ------------------------------------------------------------------
    def train_lda(self, X_train, y_train, use_search=False):
        """Train Linear Discriminant Analysis classifier"""
        print(f"\n{'='*70}")
        print("TRAINING: LINEAR DISCRIMINANT ANALYSIS (LDA)")
        print(f"{'='*70}")

        params = config.CLASSIFIER_PARAMS['mlc']
        print(f"\nParameters: {params}")

        lda = LinearDiscriminantAnalysis(**params)
        lda.fit(X_train, y_train)

        self.classifiers['LDA'] = lda
        print("✓ LDA training complete")
        return lda

    def evaluate_classifier(self, clf_name, clf, X_test, y_test):
        """
        Evaluate single classifier
        """
        y_pred = clf.predict(X_test)

        oa = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        cm = confusion_matrix(y_test, y_pred)

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

        if (self.best_by_f1_ is None) or (f1 > self.best_by_f1_[1]['F1-Score (Macro)']):
            self.best_by_f1_ = (clf_name, metrics)
        if (self.best_by_oa_ is None) or (oa > self.best_by_oa_[1]['Overall Accuracy']):
            self.best_by_oa_ = (clf_name, metrics)

        return metrics

    def train_all(self, X_train, y_train, use_search=True):
        """
        Train all classifiers
        """
        print(f"\n{'='*70}")
        print("TRAINING ALL CLASSIFIERS")
        print(f"{'='*70}\n")

        # RF, SVM, k-NN con búsqueda ligera
        self.train_random_forest(X_train, y_train, use_search=use_search)
        # self.train_svm(X_train, y_train, use_search=use_search)  # opcional
        self.train_knn(X_train, y_train, use_search=use_search)
        self.train_lda(X_train, y_train, use_search=False)

        print(f"\n{'='*70}")
        print(f"✓ ALL CLASSIFIERS TRAINED")
        print(f"{'='*70}\n")

    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all classifiers
        """
        print(f"\n{'='*70}")
        print("EVALUATING ALL CLASSIFIERS")
        print(f"{'='*70}\n")

        self.best_by_f1_ = None
        self.best_by_oa_ = None

        for clf_name, clf in self.classifiers.items():
            print(f"\nEvaluating: {clf_name}...")
            metrics = self.evaluate_classifier(clf_name, clf, X_test, y_test)

            print(f"  Overall Accuracy: {metrics['Overall Accuracy']:.4f}")
            print(f"  Kappa:            {metrics['Kappa']:.4f}")
            print(f"  F1-Score (Macro): {metrics['F1-Score (Macro)']:.4f}")

            if self.best_by_f1_ and self.best_by_oa_:
                print("  --> Current best by F1: "
                      f"{self.best_by_f1_[0]} "
                      f"(F1={self.best_by_f1_[1]['F1-Score (Macro)']:.4f})")
                print("  --> Current best by OA: "
                      f"{self.best_by_oa_[0]} "
                      f"(OA={self.best_by_oa_[1]['Overall Accuracy']:.4f})")

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
        """Save results to CSV and models"""
        print(f"\nSaving results...")

        results_data = []
        for clf_name, metrics in self.metrics.items():
            row = {
                'Classifier': clf_name,
                'Overall Accuracy': metrics['Overall Accuracy'],
                'Kappa': metrics['Kappa'],
                'Precision (Macro)': metrics['Precision (Macro)'],
                'Recall (Macro)': metrics['Recall (Macro)'],
                'F1-Score (Macro)': metrics['F1-Score (Macro)'],
                'F1-Score (Weighted)': metrics['F1-Score (Weighted)']
            }
            if clf_name in self.best_params_:
                row['Best Params'] = json.dumps(self.best_params_[clf_name])
            results_data.append(row)

        import pandas as pd
        df = pd.DataFrame(results_data)
        results_path = os.path.join(config.RESULTS_DIR, 'traditional_ml_results.csv')
        df.to_csv(results_path, index=False)
        print(f"✓ Results saved: {results_path}")

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

        if not self.metrics:
            print("No metrics available to summarize.")
            return

        print(f"{'Classifier':<20} {'OA':<10} {'Kappa':<10} {'F1-Score':<10}")
        print("-" * 50)

        for clf_name, metrics in self.metrics.items():
            print(f"{clf_name:<20} {metrics['Overall Accuracy']:<10.4f} "
                  f"{metrics['Kappa']:<10.4f} {metrics['F1-Score (Macro)']:<10.4f}")

        print("\nModels sorted by F1-Score (Macro):")
        sorted_by_f1 = sorted(
            self.metrics.items(),
            key=lambda x: x[1]['F1-Score (Macro)'],
            reverse=True
        )
        for name, m in sorted_by_f1:
            print(f"  {name:<20} F1={m['F1-Score (Macro)']:.4f}  "
                  f"OA={m['Overall Accuracy']:.4f}")

        best_f1_name, best_f1_metrics = max(
            self.metrics.items(),
            key=lambda x: x[1]['F1-Score (Macro)']
        )
        best_oa_name, best_oa_metrics = max(
            self.metrics.items(),
            key=lambda x: x[1]['Overall Accuracy']
        )

        print(f"\n✓ Best classifier by F1:  "
              f"{best_f1_name} "
              f"(F1={best_f1_metrics['F1-Score (Macro)']:.4f}, "
              f"OA={best_f1_metrics['Overall Accuracy']:.4f})")
        print(f"✓ Best classifier by OA:   "
              f"{best_oa_name} "
              f"(OA={best_oa_metrics['Overall Accuracy']:.4f}, "
              f"F1={best_oa_metrics['F1-Score (Macro)']:.4f})")

        best_clf = max(self.metrics.items(), key=lambda x: x[1]['Overall Accuracy'])
        print(f"\n✓ Best classifier (by OA): "
              f"{best_clf[0]} ({best_clf[1]['Overall Accuracy']:.4f} OA)")

        if self.best_params_:
            print("\nBest hyperparameters per classifier:")
            for name, params in self.best_params_.items():
                print(f"  {name}: {params}")


def main():
    """Example usage"""
    from data_acquisition import LandsatDataAcquisition
    from feature_extraction import FeatureExtractor

    print("Loading data...")
    acq = LandsatDataAcquisition()
    data, _, _ = acq.run()

    print("\nExtracting features...")
    fe = FeatureExtractor(data)
    features, names, _, indices_dict = fe.run()

    np.random.seed(42)
    labels = np.random.randint(0, len(config.CLASS_NAMES)-1, size=features.shape[0])

    clf = TraditionalClassifiers()
    X_train, X_test, y_train, y_test = clf.prepare_data(features, labels)
    clf.train_all(X_train, y_train, use_search=True)
    metrics = clf.evaluate_all(X_test, y_test)

    clf.visualize_confusion_matrices()
    clf.visualize_accuracy_comparison()
    clf.save_results()
    clf.print_summary()


if __name__ == "__main__":
    main()
