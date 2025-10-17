"""
BUSI-WHU Breast Cancer Dataset Training Script
Trains a neural network model for breast cancer classification using clinical features
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BUSIDataProcessor:
    """Process BUSI-WHU breast cancer clinical data"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the CSV data"""
        print(f"Loading data from {self.csv_path}...")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Drop ID column (not useful for prediction)
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Separate features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Drop unnamed/empty columns
        X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
        X = X.dropna(axis=1, how='all')
        
        # Convert diagnosis to binary: M (Malignant) = 1, B (Benign) = 0
        y = (y == 'M').astype(int)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"\nFeatures: {len(self.feature_names)}")
        print(f"Class distribution:")
        print(f"  Benign (0): {(y == 0).sum()} samples")
        print(f"  Malignant (1): {(y == 1).sum()} samples")
        
        return X.values, y.values
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled


class BreastCancerClassifier:
    """Neural Network for Breast Cancer Classification"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.class_names = ['Benign', 'Malignant']
        
    def build_model(self):
        """Build deep neural network architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            
            # First hidden layer
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Second hidden layer
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Third hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Fourth hidden layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
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
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        print("\nModel compiled successfully!")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'busi_breast_classifier_best.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\nEvaluating model on test set...")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall, test_auc = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        return {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_auc': float(test_auc),
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_probs.tolist(),
            'true_labels': y_test.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            }
        }
    
    def plot_training_history(self, save_path='busi_training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[0, 2].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
        axes[0, 2].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[0, 2].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 0].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 0].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 0].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(self.history.history['auc'], label='Train AUC', linewidth=2)
        axes[1, 1].plot(self.history.history['val_auc'], label='Val AUC', linewidth=2)
        axes[1, 1].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, save_path='busi_confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - BUSI Breast Cancer Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, roc_data, save_path='busi_roc_curve.png'):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_data["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - BUSI Breast Cancer Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath='busi_breast_classifier.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'class_names': self.class_names,
            'trained_at': datetime.now().isoformat(),
            'model_type': 'Deep Neural Network',
            'framework': 'TensorFlow/Keras',
            'dataset': 'BUSI-WHU Breast Cancer'
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, filepath='busi_breast_classifier.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.input_dim = metadata['input_dim']
                self.class_names = metadata['class_names']
            print(f"Model metadata loaded from {metadata_path}")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("BUSI-WHU Breast Cancer Dataset Training Pipeline")
    print("=" * 80)
    
    # Configuration
    CSV_PATH = "BUSI-WHU/data.csv"
    EPOCHS = 100
    BATCH_SIZE = 32
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    LEARNING_RATE = 0.001
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    
    # Initialize data processor
    print("\n[1/7] Initializing data processor...")
    data_processor = BUSIDataProcessor(CSV_PATH)
    
    # Load and preprocess data
    print("\n[2/7] Loading and preprocessing data...")
    X, y = data_processor.load_and_preprocess_data()
    
    if len(X) == 0:
        print("Error: No data loaded. Please check the CSV file.")
        return
    
    print(f"\nDataset loaded: {len(X)} samples with {X.shape[1]} features")
    
    # Split dataset
    print("\n[3/7] Splitting dataset...")
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n[4/7] Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled = data_processor.scale_features(
        X_train, X_val, X_test
    )
    
    # Initialize classifier
    print("\n[5/7] Building model...")
    classifier = BreastCancerClassifier(input_dim=X_train_scaled.shape[1])
    classifier.build_model()
    classifier.compile_model(learning_rate=LEARNING_RATE)
    
    # Train model
    print("\n[6/7] Training model...")
    history = classifier.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate model
    print("\n[7/7] Evaluating model...")
    results = classifier.evaluate(X_test_scaled, y_test)
    
    # Plot results
    print("\nGenerating visualizations...")
    classifier.plot_training_history('busi_training_history.png')
    classifier.plot_confusion_matrix(np.array(results['confusion_matrix']), 
                                    'busi_confusion_matrix.png')
    classifier.plot_roc_curve(results['roc_curve'], 'busi_roc_curve.png')
    
    # Save model
    print("\nSaving model...")
    classifier.save_model('busi_breast_classifier.h5')
    
    # Save scaler
    import joblib
    joblib.dump(data_processor.scaler, 'busi_scaler.pkl')
    print("Scaler saved to busi_scaler.pkl")
    
    # Save training results
    training_results = {
        'dataset': 'BUSI-WHU Breast Cancer',
        'total_samples': len(X),
        'num_features': X.shape[1],
        'feature_names': data_processor.feature_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'test_metrics': {
            'accuracy': results['test_accuracy'],
            'precision': results['test_precision'],
            'recall': results['test_recall'],
            'auc': results['test_auc'],
            'loss': results['test_loss']
        },
        'confusion_matrix': results['confusion_matrix'],
        'class_names': classifier.class_names,
        'trained_at': datetime.now().isoformat()
    }
    
    with open('busi_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\nFinal Test Results:")
    print(f"  Accuracy:  {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {results['test_precision']:.4f}")
    print(f"  Recall:    {results['test_recall']:.4f}")
    print(f"  AUC:       {results['test_auc']:.4f}")
    print(f"\nFiles generated:")
    print(f"  - busi_breast_classifier.h5 (trained model)")
    print(f"  - busi_breast_classifier_metadata.json (model metadata)")
    print(f"  - busi_scaler.pkl (feature scaler)")
    print(f"  - busi_training_results.json (training results)")
    print(f"  - busi_training_history.png (training curves)")
    print(f"  - busi_confusion_matrix.png (confusion matrix)")
    print(f"  - busi_roc_curve.png (ROC curve)")
    print("=" * 80)


if __name__ == "__main__":
    main()
