"""
MODULE 5: EVALUATION
Comprehensive accuracy assessment and model comparison
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score, 
    jaccard_score
)

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class ClassificationEvaluator:
    """
    Comprehensive evaluation of classification results
    Calculates 9+ metrics and produces comparison visualizations
    """
    
    def __init__(self):
        """Initialize evaluator"""
        print("\n" + "="*70)
        print("EVALUATION MODULE")
        print("="*70)
        
        self.results = {}
        self.per_class_metrics = {}
        
        # Create output directory
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def evaluate_classification(self, y_true, y_pred, classifier_name):
        """
        Evaluate single classification
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        classifier_name : str
            Name of classifier
        
        Returns:
        --------
        dict : Metrics dictionary
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Calculate metrics
        oa = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # IoU / Jaccard Index
        iou_scores = []
        for class_idx in np.unique(y_true):
            iou = jaccard_score(y_true == class_idx, y_pred == class_idx)
            iou_scores.append(iou)
        mean_iou = np.mean(iou_scores)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'Classifier': classifier_name,
            'Overall Accuracy': oa,
            'Kappa': kappa,
            'Precision (Macro)': macro_precision,
            'Recall (Macro)': macro_recall,
            'F1-Score (Macro)': macro_f1,
            'F1-Score (Weighted)': weighted_f1,
            'Mean IoU': mean_iou,
            'Confusion Matrix': cm,
            'Class Report': class_report
        }
        
        self.results[classifier_name] = metrics
        
        print(f"\n{classifier_name}")
        print(f"  Overall Accuracy: {oa:.4f}")
        print(f"  Kappa:            {kappa:.4f}")
        print(f"  F1-Score:         {macro_f1:.4f}")
        print(f"  Mean IoU:         {mean_iou:.4f}")
        
        return metrics
    
    def create_comparison_table(self):
        """Create model comparison table"""
        print(f"\n{'='*70}")
        print("MODEL COMPARISON TABLE")
        print(f"{'='*70}\n")
        
        comparison_data = []
        for clf_name, metrics in self.results.items():
            comparison_data.append({
                'Classifier': clf_name,
                'Overall Accuracy': metrics['Overall Accuracy'],
                'Kappa': metrics['Kappa'],
                'Precision': metrics['Precision (Macro)'],
                'Recall': metrics['Recall (Macro)'],
                'F1-Score': metrics['F1-Score (Macro)'],
                'Weighted F1': metrics['F1-Score (Weighted)'],
                'Mean IoU': metrics['Mean IoU']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Print table
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(config.RESULTS_DIR, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison table saved: {csv_path}")
        
        return df
    
    def visualize_metrics_comparison(self):
        """Visualize metrics comparison across classifiers"""
        print(f"\nVisualizing metrics comparison...")
        
        metrics_list = ['Overall Accuracy', 'Kappa', 'F1-Score (Macro)', 'Mean IoU']
        classifiers = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        fig.suptitle('Classification Metrics Comparison', fontsize=14)
        
        for idx, metric in enumerate(metrics_list):
            values = [self.results[clf][metric] for clf in classifiers]
            
            ax = axes[idx]
            bars = ax.bar(classifiers, values, alpha=0.7)
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig_path = os.path.join(config.FIGURES_DIR, '07_metrics_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Metrics comparison saved: {fig_path}")
        plt.close()
    
    def visualize_confusion_matrices(self):
        """Visualize all confusion matrices"""
        print(f"\nVisualizing confusion matrices...")
        
        n_classifiers = len(self.results)
        n_cols = min(n_classifiers, 2)
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        if n_classifiers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Confusion Matrices - All Classifiers', fontsize=14)
        
        for idx, (clf_name, metrics) in enumerate(self.results.items()):
            cm = metrics['Confusion Matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{clf_name} (OA={metrics["Overall Accuracy"]:.3f})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, '08_all_confusion_matrices.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices saved: {fig_path}")
        plt.close()
    
    def create_per_class_report(self):
        """Create per-class precision/recall/F1 table"""
        print(f"\nGenerating per-class report...")
        
        # Get per-class metrics for each classifier
        per_class_data = []
        
        for clf_name, metrics in self.results.items():
            class_report = metrics['Class Report']
            
            for class_idx in sorted([k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
                if class_idx.isdigit():
                    class_idx_int = int(class_idx)
                    if class_idx_int < len(config.CLASS_NAMES):
                        class_name = config.CLASS_NAMES[class_idx_int]
                        per_class_data.append({
                            'Classifier': clf_name,
                            'Class': f"{class_name} ({class_idx})",
                            'Precision': class_report[class_idx]['precision'],
                            'Recall': class_report[class_idx]['recall'],
                            'F1-Score': class_report[class_idx]['f1-score'],
                            'Support': int(class_report[class_idx]['support'])
                        })
        
        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            
            # Save to CSV
            csv_path = os.path.join(config.RESULTS_DIR, 'per_class_metrics.csv')
            df_per_class.to_csv(csv_path, index=False)
            print(f"✓ Per-class report saved: {csv_path}")
            
            # Visualize per-class F1 scores
            fig, ax = plt.subplots(figsize=(12, 6))
            
            classifiers = df_per_class['Classifier'].unique()
            classes = df_per_class['Class'].unique()
            
            x = np.arange(len(classes))
            width = 0.15
            
            for idx, clf in enumerate(classifiers):
                data = df_per_class[df_per_class['Classifier'] == clf]
                f1_scores = [data[data['Class'] == c]['F1-Score'].values[0] if c in data['Class'].values else 0 for c in classes]
                ax.bar(x + idx*width, f1_scores, width, label=clf, alpha=0.8)
            
            ax.set_xlabel('Class')
            ax.set_ylabel('F1-Score')
            ax.set_title('Per-Class F1-Scores')
            ax.set_xticks(x + width)
            ax.set_xticklabels(classes, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            fig_path = os.path.join(config.FIGURES_DIR, '09_per_class_f1_scores.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"✓ Per-class F1 scores saved: {fig_path}")
            plt.close()
    
    def generate_summary_report(self):
        """Generate text summary report"""
        print(f"\nGenerating summary report...")
        
        report_path = os.path.join(config.RESULTS_DIR, 'evaluation_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LAND COVER CLASSIFICATION - EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Best classifier
            best_clf = max(self.results.items(), key=lambda x: x[1]['Overall Accuracy'])
            f.write(f"BEST CLASSIFIER: {best_clf[0]}\n")
            f.write(f"  Overall Accuracy: {best_clf[1]['Overall Accuracy']:.4f}\n")
            f.write(f"  Kappa: {best_clf[1]['Kappa']:.4f}\n")
            f.write(f"  F1-Score: {best_clf[1]['F1-Score (Macro)']:.4f}\n\n")
            
            # All classifiers
            f.write("ALL CLASSIFIERS:\n")
            f.write("-" * 70 + "\n")
            for clf_name, metrics in self.results.items():
                f.write(f"\n{clf_name}:\n")
                f.write(f"  Overall Accuracy: {metrics['Overall Accuracy']:.4f}\n")
                f.write(f"  Kappa: {metrics['Kappa']:.4f}\n")
                f.write(f"  Precision: {metrics['Precision (Macro)']:.4f}\n")
                f.write(f"  Recall: {metrics['Recall (Macro)']:.4f}\n")
                f.write(f"  F1-Score: {metrics['F1-Score (Macro)']:.4f}\n")
                f.write(f"  Mean IoU: {metrics['Mean IoU']:.4f}\n")
        
        print(f"✓ Summary report saved: {report_path}")
        
        # Also print to console
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        with open(report_path, 'r') as f:
            print(f.read())


def main():
    """Example usage"""
    evaluator = ClassificationEvaluator()
    
    # Dummy predictions for demo
    y_true = np.random.randint(0, 5, 1000)
    y_pred_rf = np.random.randint(0, 5, 1000)
    y_pred_svm = np.random.randint(0, 5, 1000)
    
    # Evaluate
    evaluator.evaluate_classification(y_true, y_pred_rf, 'Random Forest')
    evaluator.evaluate_classification(y_true, y_pred_svm, 'SVM')
    
    # Generate reports and visualizations
    evaluator.create_comparison_table()
    evaluator.visualize_metrics_comparison()
    evaluator.visualize_confusion_matrices()
    evaluator.create_per_class_report()
    evaluator.generate_summary_report()


if __name__ == "__main__":
    main()
