"""
EVALUATION.PY - Classification evaluation utilities
"""

import os
import logging
import json

try:
    import config
except ImportError:
    print("Error: config.py not found")
    raise

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluate and summarize traditional ML classifier results"""

    def __init__(self):
        self.results = {}   # dict: {model_name: { 'Overall Accuracy': float, ...}}
        self.output_dir = os.path.join(config.RESULTS_DIR, "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_summary_report(self, metrics=None):
        """
        Generate a text/JSON summary of classifier performance.

        Parameters
        ----------
        metrics : dict or None
            If provided, should be the metrics dict from TraditionalClassifiers:
            {
              "Random Forest": {"Overall Accuracy": 0.87, ...},
              "SVM": {...},
              ...
            }
        """
        # 1) Use provided metrics if given
        if metrics is not None:
            self.results = metrics

        # 2) If still empty, try loading from disk (optional, only if you save metrics there)
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

        # Example: find best classifier by overall accuracy
        try:
            best_clf_name, best_clf_metrics = max(
                self.results.items(),
                key=lambda x: x[1].get("Overall Accuracy", 0.0)
            )
        except ValueError:
            # This would only happen if self.results is empty, but we guarded above
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

        # Optionally save metrics as JSON for later use
        json_path = os.path.join(self.output_dir, "ml_metrics.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved metrics JSON to {json_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics JSON: {e}")
