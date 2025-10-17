"""
LA-Breast DCE-MRI Dataset Training Script
Trains a CNN model for breast MRI classification and saves it as h5 file
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import re
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MRIDataLoader:
    """Load and preprocess MRI images from LA-Breast DCE-MRI Dataset"""
    
    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.class_mapping = {
            'RN': 0,  # Normal
            'R1': 1,  # Benign
            'R2': 1,  # Benign
            'R3': 1,  # Benign
            'R4': 2,  # Malignant
            'R5': 2,  # Malignant
            'R6': 2,  # Malignant
            'R7': 2,  # Malignant
            'R8': 2,  # Malignant
            'R9': 2,  # Malignant
            'R10': 2, # Malignant
            'R11': 2, # Malignant
            'R12': 2  # Malignant
        }
        self.class_names = ['Normal', 'Benign', 'Malignant']
        
    def extract_label_from_filename(self, filename):
        """Extract label from filename (e.g., Mri_130_R1_IM-0360-0168.tiff -> R1)"""
        # Pattern to match R followed by number or RN
        pattern = r'_(R\d+|RN)_'
        match = re.search(pattern, filename)
        if match:
            label = match.group(1)
            return self.class_mapping.get(label, None)
        return None
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single MRI image"""
        try:
            # Load TIFF image
            img = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            img = img.resize(self.img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_dataset(self, data_dir):
        """Load all images and labels from a directory"""
        images = []
        labels = []
        filenames = []
        
        # Get all TIFF files
        image_files = list(Path(data_dir).rglob('*.tiff')) + list(Path(data_dir).rglob('*.tif'))
        
        print(f"Found {len(image_files)} images in {data_dir}")
        
        for img_path in image_files:
            # Extract label from filename
            label = self.extract_label_from_filename(img_path.name)
            
            if label is None:
                print(f"Warning: Could not extract label from {img_path.name}")
                continue
            
            # Load and preprocess image
            img_array = self.load_and_preprocess_image(str(img_path))
            
            if img_array is not None:
                images.append(img_array)
                labels.append(label)
                filenames.append(img_path.name)
        
        print(f"Successfully loaded {len(images)} images")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for cls, count in zip(unique, counts):
            print(f"  {self.class_names[cls]}: {count} images")
        
        return X, y, filenames


class MRIClassifier:
    """CNN-based MRI Breast Cancer Classifier"""
    
    def __init__(self, img_size=(224, 224), num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = ['Normal', 'Benign', 'Malignant']
        
    def build_model(self):
        """Build CNN model architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        print("\nModel compiled successfully!")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """Train the model"""
        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'mri_breast_classifier_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\nEvaluating model on test set...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
    
    def plot_training_history(self, save_path='mri_training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path='mri_confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - MRI Breast Cancer Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath='mri_breast_classifier.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
        
        # Save metadata
        metadata = {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'trained_at': datetime.now().isoformat(),
            'model_architecture': 'CNN',
            'framework': 'TensorFlow/Keras'
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, filepath='mri_breast_classifier.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.img_size = tuple(metadata['img_size'])
                self.num_classes = metadata['num_classes']
                self.class_names = metadata['class_names']
            print(f"Model metadata loaded from {metadata_path}")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("LA-Breast DCE-MRI Dataset Training Pipeline")
    print("=" * 80)
    
    # Configuration
    DATASET_PATH = "LA-Breast DCE-MRI Dataset/breas_data/breast_data/d5"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 50
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    
    # Initialize data loader
    print("\n[1/6] Initializing data loader...")
    data_loader = MRIDataLoader(DATASET_PATH, img_size=IMG_SIZE)
    
    # Load dataset
    print("\n[2/6] Loading MRI dataset...")
    X, y, filenames = data_loader.load_dataset(DATASET_PATH)
    
    if len(X) == 0:
        print("Error: No images loaded. Please check the dataset path.")
        return
    
    print(f"\nDataset loaded: {len(X)} images")
    print(f"Image shape: {X[0].shape}")
    
    # Split dataset
    print("\n[3/6] Splitting dataset...")
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Initialize classifier
    print("\n[4/6] Building model...")
    classifier = MRIClassifier(img_size=IMG_SIZE, num_classes=3)
    classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    
    # Train model
    print("\n[5/6] Training model...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate model
    print("\n[6/6] Evaluating model...")
    results = classifier.evaluate(X_test, y_test)
    
    # Plot results
    print("\nGenerating visualizations...")
    classifier.plot_training_history('mri_training_history.png')
    classifier.plot_confusion_matrix(np.array(results['confusion_matrix']), 
                                    'mri_confusion_matrix.png')
    
    # Save model
    print("\nSaving model...")
    classifier.save_model('mri_breast_classifier.h5')
    
    # Save training results
    training_results = {
        'dataset': 'LA-Breast DCE-MRI',
        'total_images': len(X),
        'train_images': len(X_train),
        'val_images': len(X_val),
        'test_images': len(X_test),
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'test_metrics': {
            'accuracy': results['test_accuracy'],
            'precision': results['test_precision'],
            'recall': results['test_recall'],
            'loss': results['test_loss']
        },
        'confusion_matrix': results['confusion_matrix'],
        'class_names': classifier.class_names,
        'trained_at': datetime.now().isoformat()
    }
    
    with open('mri_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Test Precision: {results['test_precision']:.4f}")
    print(f"  Test Recall: {results['test_recall']:.4f}")
    print(f"\nModel saved as: mri_breast_classifier.h5")
    print(f"Training results saved as: mri_training_results.json")
    print(f"Visualizations saved as: mri_training_history.png, mri_confusion_matrix.png")


if __name__ == "__main__":
    main()
