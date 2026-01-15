#!/usr/bin/env python3
"""
DEEP_LEARNING.PY
Deep learning models (U-Net, CNN) with detailed performance analysis and visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
CLASS_NAMES = config.CLASS_NAMES           # lista de nombres de clase
CLASS_COLORS = {
    0: [0.2, 0.6, 0.9],   # Water
    1: [0.1, 0.5, 0.1],   # Forest
    2: [0.8, 0.9, 0.2],   # Grassland
    3: [0.8, 0.3, 0.1],   # Urban
    4: [0.7, 0.6, 0.4],   # Bare soil
}

def labels_to_rgb(labels_2d):
    """Convert label map to RGB for visualization."""
    labels_2d = np.squeeze(labels_2d)
    if labels_2d.ndim != 2:
        raise ValueError(f"labels_to_rgb expects 2D array, got shape {labels_2d.shape}")
    h, w = labels_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cid, col in CLASS_COLORS.items():
        mask = labels_2d == cid
        rgb[mask] = col
    return rgb

# ------------------------------------------------------------------
# U-NET MODEL
# ------------------------------------------------------------------
def build_unet(input_shape, num_classes):
    """Build U-Net model for semantic segmentation."""
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u3 = layers.UpSampling2D((2, 2))(c4)
    u3 = layers.Concatenate()([u3, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u3)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u1 = layers.UpSampling2D((2, 2))(c6)
    u1 = layers.Concatenate()([u1, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c7)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# ------------------------------------------------------------------
# SIMPLE CNN MODEL
# ------------------------------------------------------------------
def build_cnn(input_shape, num_classes):
    """Build simple CNN for pixel-wise classification (converted to dense)."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.Conv2D(num_classes, 1, activation='softmax'),
    ])
    return model

# ------------------------------------------------------------------
# DEEP LEARNING CLASS
# ------------------------------------------------------------------
class DeepLearningModels:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.models_dl = {}
        self.metrics_dl = {}
        self.history_dl = {}

    def prepare_data_for_dl(self, features, labels, test_size=0.3):
        """
        Prepare data for deep learning.
        Features: (N, F) -> (H, W, F) patch-based or flattened
        Labels: (N,) -> (H, W)
        """
        N = features.shape[0]
        F = features.shape[1]

        H = W = int(np.sqrt(N))
        if H * W == N:
            X_img = features.reshape(H, W, F)
            y_img = labels.reshape(H, W)
        else:
            X_img = features
            y_img = labels

        split_idx = int(0.7 * H)
        X_train = X_img[:split_idx, :, :]
        y_train = y_img[:split_idx, :]
        X_val = X_img[split_idx:, :, :]
        y_val = y_img[split_idx:, :]

        print(f"Training shape: {X_train.shape}, {y_train.shape}")
        print(f"Val shape: {X_val.shape}, {y_val.shape}")

        return X_train, X_val, y_train, y_val

    def train_unet(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
        """Train U-Net model."""
        print(f"\n{'='*70}")
        print("TRAINING U-NET")
        print(f"{'='*70}")

        input_shape = X_train.shape[1:]
        model = build_unet(input_shape, self.num_classes)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        self.models_dl['U-Net'] = model
        self.history_dl['U-Net'] = history
        self._evaluate_model(model, X_val, y_val, 'U-Net')

        return model

    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
        """Train simple CNN model."""
        print(f"\n{'='*70}")
        print("TRAINING CNN")
        print(f"{'='*70}")

        input_shape = X_train.shape[1:]
        model = build_cnn(input_shape, self.num_classes)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        self.models_dl['CNN'] = model
        self.history_dl['CNN'] = history
        self._evaluate_model(model, X_val, y_val, 'CNN')

        return model

    def _evaluate_model(self, model, X_val, y_val, model_name):
        """Evaluate model and compute detailed metrics."""
        print(f"\n{'='*70}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*70}")

        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=-1)

        y_val_flat = y_val.flatten()
        y_pred_flat = y_pred.flatten()

        overall_acc = accuracy_score(y_val_flat, y_pred_flat)
        print(f"\nOverall Accuracy: {overall_acc:.4f}")

        precision_macro = precision_score(y_val_flat, y_pred_flat, average='macro', zero_division=0)
        recall_macro = recall_score(y_val_flat, y_pred_flat, average='macro', zero_division=0)
        f1_macro = f1_score(y_val_flat, y_pred_flat, average='macro', zero_division=0)

        print(f"Macro Precision: {precision_macro:.4f}")
        print(f"Macro Recall: {recall_macro:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")

        print(f"\nPer-class metrics:")
        for cls in range(self.num_classes):
            p = precision_score(y_val_flat, y_pred_flat, labels=[cls], average='macro', zero_division=0)
            r = recall_score(y_val_flat, y_pred_flat, labels=[cls], average='macro', zero_division=0)
            f = f1_score(y_val_flat, y_pred_flat, labels=[cls], average='macro', zero_division=0)
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
            print(f"  Class {cls} ({name}): P={p:.3f} R={r:.3f} F1={f:.3f}")

        cm = confusion_matrix(y_val_flat, y_pred_flat, labels=range(self.num_classes))

        self.metrics_dl[model_name] = {
            'overall_accuracy': overall_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_val': y_val,
        }

        print(f"\n✓ {model_name} evaluation complete.")

    def _to_2d_label(self, arr):
        """Ensure arr is a single 2D label map (H, W)."""
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[0]
        raise ValueError(f"Unexpected label array shape: {arr.shape}")

    def plot_predictions(self, save_dir=None):
        """Plot predictions from each model as colored maps."""
        if not self.metrics_dl:
            print("No models evaluated yet.")
            return

        num_models = len(self.metrics_dl)
        fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))

        if num_models == 1:
            axes = [axes]

        first_metrics = list(self.metrics_dl.values())[0]
        y_val_full = first_metrics['y_val']
        y_val_2d = self._to_2d_label(y_val_full)

        ax_idx = 0
        axes[ax_idx].imshow(labels_to_rgb(y_val_2d))
        axes[ax_idx].set_title("Ground Truth Labels")
        axes[ax_idx].axis('off')
        ax_idx += 1

        for model_name, metrics in self.metrics_dl.items():
            y_pred_full = metrics['y_pred']
            y_pred_2d = self._to_2d_label(y_pred_full)

            axes[ax_idx].imshow(labels_to_rgb(y_pred_2d))
            axes[ax_idx].set_title(f"{model_name} Predictions\n(Acc: {metrics['overall_accuracy']:.3f})")
            axes[ax_idx].axis('off')
            ax_idx += 1

        legend_elements = [
            mpatches.Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
            for i in sorted(CLASS_COLORS.keys())
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, 'dl_predictions.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved predictions plot to {out_path}")
        plt.close()

    def plot_confusion_matrices(self, save_dir=None):
        """Plot confusion matrices for each model."""
        if not self.metrics_dl:
            print("No models evaluated yet.")
            return

        num_models = len(self.metrics_dl)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

        if num_models == 1:
            axes = [axes]

        class_labels = CLASS_NAMES  # lista de nombres

        for ax_idx, (model_name, metrics) in enumerate(self.metrics_dl.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axes[ax_idx],
                xticklabels=class_labels, yticklabels=class_labels
            )
            axes[ax_idx].set_title(f"{model_name} Confusion Matrix")
            axes[ax_idx].set_ylabel("True Label")
            axes[ax_idx].set_xlabel("Predicted Label")

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, 'dl_confusion_matrices.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrices to {out_path}")
        plt.close()

    def plot_metrics_comparison(self, save_dir=None):
        """Compare accuracy and F1 across all models."""
        if not self.metrics_dl:
            print("No models evaluated yet.")
            return

        model_names = list(self.metrics_dl.keys())
        accuracies = [self.metrics_dl[m]['overall_accuracy'] for m in model_names]
        f1_scores = [self.metrics_dl[m]['f1_macro'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, accuracies, width, label='Overall Accuracy', color='steelblue')
        ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='coral')

        ax.set_ylabel('Score')
        ax.set_title('Deep Learning Models Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, 'dl_metrics_comparison.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved metrics comparison to {out_path}")
        plt.close()

    def plot_training_history(self, save_dir=None):
        """Plot training history (loss and accuracy over epochs)."""
        if not self.history_dl:
            print("No training history available.")
            return

        num_models = len(self.history_dl)
        fig, axes = plt.subplots(num_models, 2, figsize=(12, 4 * num_models))

        if num_models == 1:
            axes = [axes]

        for ax_idx, (model_name, history) in enumerate(self.history_dl.items()):
            axes[ax_idx][0].plot(history.history['loss'], label='Train Loss')
            axes[ax_idx][0].plot(history.history['val_loss'], label='Val Loss')
            axes[ax_idx][0].set_title(f"{model_name} Loss")
            axes[ax_idx][0].set_xlabel("Epoch")
            axes[ax_idx][0].set_ylabel("Loss")
            axes[ax_idx][0].legend()
            axes[ax_idx][0].grid(True, alpha=0.3)

            axes[ax_idx][1].plot(history.history['accuracy'], label='Train Accuracy')
            axes[ax_idx][1].plot(history.history['val_accuracy'], label='Val Accuracy')
            axes[ax_idx][1].set_title(f"{model_name} Accuracy")
            axes[ax_idx][1].set_xlabel("Epoch")
            axes[ax_idx][1].set_ylabel("Accuracy")
            axes[ax_idx][1].legend()
            axes[ax_idx][1].grid(True, alpha=0.3)
  
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, 'dl_training_history.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {out_path}")
        plt.close()

    def save_results(self, save_dir=None):
        """Save models and metrics, and generate all DL figures."""
        if save_dir is None:
            save_dir = os.path.join(config.RESULTS_DIR, 'deep_learning')

        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.models_dl.items():
            model_path = os.path.join(save_dir, f'{model_name.lower()}_model.h5')
            model.save(model_path)
            print(f"Saved {model_name} to {model_path}")

        self.plot_predictions(save_dir)
        self.plot_confusion_matrices(save_dir)
        self.plot_metrics_comparison(save_dir)
        self.plot_training_history(save_dir)


if __name__ == "__main__":
    print("Deep Learning module ready.")
