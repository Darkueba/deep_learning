"""
DEEP_LEARNING.PY - Fixed segmentation module
CNN and U-Net for pixel-wise land cover segmentation
"""

import numpy as np
import os
import logging
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, 
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

try:
    import config
except ImportError:
    print("Error: config.py not found")
    raise

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DeepLearningModels:
    """Deep learning models for land cover segmentation"""
    
    def __init__(self):
        """Initialize DL module"""
        print("\n" + "="*70)
        print("DEEP LEARNING MODULE")
        print("="*70)
        
        self.models = {}
        self.histories = {}
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def build_unet_model(self, input_shape, num_classes):
        """
        Build U-Net for semantic segmentation
        
        Input: (64, 64, 11)
        Output: (64, 64, num_classes)
        
        Args:
            input_shape: tuple (height, width, channels)
            num_classes: number of land cover classes
        
        Returns:
            Compiled Keras Model
        """
        logger.info(f"\nBuilding U-Net model...")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Output classes: {num_classes}")
        
        inputs = Input(shape=input_shape)
        
        # Encoder
        c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
        c1 = Conv2D(32, (3,3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2,2))(c1)
        
        c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
        c2 = Conv2D(64, (3,3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2,2))(c2)
        
        c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
        c3 = Conv2D(128, (3,3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2,2))(c3)
        
        c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
        c4 = Conv2D(256, (3,3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D((2,2))(c4)
        
        # Bottleneck
        c5 = Conv2D(512, (3,3), activation='relu', padding='same')(p4)
        c5 = Conv2D(512, (3,3), activation='relu', padding='same')(c5)
        
        # Decoder
        u6 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c5)
        u6 = Concatenate()([u6, c4])
        c6 = Conv2D(256, (3,3), activation='relu', padding='same')(u6)
        c6 = Conv2D(256, (3,3), activation='relu', padding='same')(c6)
        
        u7 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c6)
        u7 = Concatenate()([u7, c3])
        c7 = Conv2D(128, (3,3), activation='relu', padding='same')(u7)
        c7 = Conv2D(128, (3,3), activation='relu', padding='same')(c7)
        
        u8 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c7)
        u8 = Concatenate()([u8, c2])
        c8 = Conv2D(64, (3,3), activation='relu', padding='same')(u8)
        c8 = Conv2D(64, (3,3), activation='relu', padding='same')(c8)
        
        u9 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c8)
        u9 = Concatenate()([u9, c1])
        c9 = Conv2D(32, (3,3), activation='relu', padding='same')(u9)
        c9 = Conv2D(32, (3,3), activation='relu', padding='same')(c9)
        
        # Output layer: per-pixel class probabilities
        outputs = Conv2D(num_classes, (1,1), activation='softmax', padding='same')(c9)
        
        model = Model(inputs=inputs, outputs=outputs, name='U-Net')
        
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✓ U-Net model built successfully")
        logger.info(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def build_cnn_model(self, input_shape, num_classes):
        """
        Build CNN for semantic segmentation
        Similar to U-Net but simpler (no skip connections)
        
        Input: (64, 64, 11)
        Output: (64, 64, num_classes)
        
        Args:
            input_shape: tuple (height, width, channels)
            num_classes: number of land cover classes
        
        Returns:
            Compiled Keras Model
        """
        logger.info(f"\nBuilding CNN model for segmentation...")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Output classes: {num_classes}")
        
        inputs = Input(shape=input_shape)
        
        x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2))(x)
        
        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2))(x)
        
        x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
        
        # Upsample back to original size
        x = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(x)
        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        
        x = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(x)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
        
        # Output: per-pixel class probabilities
        outputs = Conv2D(num_classes, (1,1), activation='softmax', padding='same')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNN-Segmentation')
        
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✓ CNN model built successfully")
        logger.info(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val):
        """
        Train a segmentation model
        
        Args:
            model_name: str, name of model (e.g., 'U-Net')
            model: compiled Keras model
            X_train: training patches (N, 64, 64, 11)
            y_train: training masks (N, 64, 64) with integer class IDs
            X_val: validation patches (M, 64, 64, 11)
            y_val: validation masks (M, 64, 64)
        
        Returns:
            Training history
        """
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}\n")
        
        logger.info(f"Training {model_name}...")
        logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=8,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.histories[model_name] = history
        logger.info(f"✓ {model_name} training complete")
        
        return history
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate a model
        
        Args:
            model_name: str
            model: trained model
            X_test: test patches (N, 64, 64, 11)
            y_test: test masks (N, 64, 64)
        
        Returns:
            dict with metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        
        logger.info(f"  Loss: {loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def visualize_training_history(self, model_name, history):
        """Plot training history"""
        logger.info(f"Creating training history visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, f'{model_name.lower()}_training_history.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {fig_path}")
        plt.close()
    
    def save_models(self):
        """Save trained models"""
        logger.info(f"\nSaving models...")
        models_dir = os.path.join(config.RESULTS_DIR, 'deep_learning_models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            path = os.path.join(models_dir, f'{model_name.lower().replace("-", "_")}.h5')
            model.save(path)
            logger.info(f"  ✓ Saved: {path}")


def main():
    pass


if __name__ == "__main__":
    main()
