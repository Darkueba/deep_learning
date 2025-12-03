"""
MODULE 4: DEEP LEARNING
Implements CNN and U-Net architectures for semantic segmentation
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠ TensorFlow not available - skipping deep learning")
    TENSORFLOW_AVAILABLE = False

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class DeepLearningModels:
    """
    Implements deep learning models for semantic segmentation
    Models: CNN, U-Net
    """
    
    def __init__(self):
        """Initialize deep learning module"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠ TensorFlow not installed - install with: pip install tensorflow")
            return
        
        print("\n" + "="*70)
        print("DEEP LEARNING MODULE")
        print("="*70)
        
        self.models = {}
        self.histories = {}
        self.predictions = {}
        
        # Create output directory
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def build_cnn_model(self, input_shape, num_classes):
        """
        Build simple CNN model
        
        Parameters:
        -----------
        input_shape : tuple
            Input shape (height, width, channels)
        num_classes : int
            Number of output classes
        
        Returns:
        --------
        keras.Model : Compiled model
        """
        print(f"\n{'='*70}")
        print("BUILDING CNN MODEL")
        print(f"{'='*70}")
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(config.DROPOUT_RATE),
            
            # Global pooling and dense
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✓ CNN model built")
        model.summary()
        
        return model
    
    def build_unet_model(self, input_shape, num_classes):
        """
        Build U-Net model for semantic segmentation
        
        Parameters:
        -----------
        input_shape : tuple
            Input shape (height, width, channels)
        num_classes : int
            Number of output classes
        
        Returns:
        --------
        keras.Model : Compiled model
        """
        print(f"\n{'='*70}")
        print("BUILDING U-NET MODEL")
        print(f"{'='*70}")
        
        inputs = keras.Input(shape=input_shape)
        
        # Encoder
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D(2)(c1)
        
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D(2)(c2)
        
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(2)(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
        
        # Decoder
        u3 = layers.UpSampling2D(2)(c4)
        u3 = layers.Concatenate()([u3, c3])
        c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
        c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(c5)
        
        u2 = layers.UpSampling2D(2)(c5)
        u2 = layers.Concatenate()([u2, c2])
        c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
        c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(c6)
        
        u1 = layers.UpSampling2D(2)(c6)
        u1 = layers.Concatenate()([u1, c1])
        c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
        c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(c7)
        
        # Output
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c7)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✓ U-Net model built")
        model.summary()
        
        return model
    
    def prepare_patch_data(self, features_2d, labels, patch_size=64):
        """
        Prepare patch-based data for training
        
        Parameters:
        -----------
        features_2d : numpy.ndarray
            Feature matrix (H*W, num_features)
        labels : numpy.ndarray
            Labels (H*W,)
        patch_size : int
            Size of patches
        
        Returns:
        --------
        tuple : (X_patches, y_patches)
        """
        print(f"\nPreparing patch-based data (patch_size={patch_size})...")
        
        # Reshape to 3D
        height = int(np.sqrt(features_2d.shape[0]))
        width = height
        num_features = features_2d.shape[1]
        
        features_3d = features_2d.reshape(height, width, num_features)
        labels_2d = labels.reshape(height, width)
        
        X_patches = []
        y_patches = []
        
        # Extract patches
        for i in range(0, height - patch_size + 1, patch_size//2):
            for j in range(0, width - patch_size + 1, patch_size//2):
                patch = features_3d[i:i+patch_size, j:j+patch_size, :]
                label_patch = labels_2d[i:i+patch_size, j:j+patch_size]
                
                if patch.shape == (patch_size, patch_size, num_features):
                    X_patches.append(patch)
                    y_patches.append(label_patch)
        
        X_patches = np.array(X_patches)
        y_patches = np.array(y_patches)
        
        print(f"✓ Created {X_patches.shape[0]} patches")
        print(f"  Patch shape: {X_patches.shape[1:]}")
        
        return X_patches, y_patches
    
    def train_model(self, model_name, model, X_train, y_train, X_val=None, y_val=None):
        """
        Train model
        
        Parameters:
        -----------
        model_name : str
            Model name
        model : keras.Model
            Model to train
        X_train : numpy.ndarray
            Training data
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray, optional
            Validation data
        y_val : numpy.ndarray, optional
            Validation labels
        """
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(
                os.path.join(config.RESULTS_DIR, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        print(f"✓ Training complete")
        
        return history
    
    def visualize_training_history(self, model_name, history):
        """Visualize training history"""
        print(f"\nVisualizing training history for {model_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, f'06_{model_name}_training_history.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history saved: {fig_path}")
        plt.close()
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate model on test set
        
        Parameters:
        -----------
        model_name : str
            Model name
        model : keras.Model
            Trained model
        X_test : numpy.ndarray
            Test data
        y_test : numpy.ndarray
            Test labels
        
        Returns:
        --------
        dict : Metrics dictionary
        """
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test, verbose=0).argmax(axis=-1).flatten()
        y_test_flat = y_test.flatten()
        
        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
        
        oa = accuracy_score(y_test_flat, y_pred)
        f1 = f1_score(y_test_flat, y_pred, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_test_flat, y_pred)
        
        metrics = {
            'Overall Accuracy': oa,
            'F1-Score': f1,
            'Kappa': kappa
        }
        
        print(f"  Overall Accuracy: {oa:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Kappa: {kappa:.4f}")
        
        return metrics
    
    def save_models(self):
        """Save trained models"""
        print(f"\nSaving models...")
        
        for model_name, model in self.models.items():
            model_path = os.path.join(config.RESULTS_DIR, f'{model_name}.h5')
            model.save(model_path)
            print(f"✓ Model saved: {model_path}")


def main():
    """Example usage"""
    if not TENSORFLOW_AVAILABLE:
        print("⚠ TensorFlow not available - skipping deep learning demo")
        return
    
    from data_acquisition import LandsatDataAcquisition
    from feature_extraction import FeatureExtractor
    
    # Get data
    print("Loading data...")
    acq = LandsatDataAcquisition()
    data, _, _ = acq.run()
    
    # Extract features
    print("\nExtracting features...")
    fe = FeatureExtractor(data)
    features_2d, names, _ = fe.run()
    
    # Generate synthetic labels
    np.random.seed(42)
    labels = np.random.randint(0, len(config.CLASS_NAMES)-1, size=features_2d.shape[0])
    
    # Initialize module
    dl = DeepLearningModels()
    
    # Prepare patch data
    X_patches, y_patches = dl.prepare_patch_data(features_2d, labels, patch_size=64)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_patches, y_patches,
        test_size=0.2,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42
    )
    
    # Build and train CNN
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, len(names))
    num_classes = len(config.CLASS_NAMES) - 1
    
    cnn = dl.build_cnn_model(input_shape, num_classes)
    history_cnn = dl.train_model('CNN', cnn, X_train, y_train, X_val, y_val)
    dl.visualize_training_history('CNN', history_cnn)
    
    # Evaluate
    metrics_cnn = dl.evaluate_model('CNN', cnn, X_test, y_test)
    
    print(f"\n{'='*70}")
    print("✓ DEEP LEARNING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
