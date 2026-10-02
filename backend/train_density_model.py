"""
Mammogram Density Assessment Training Script
Trains a CNN model to predict breast density from mammogram images
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MammogramDensityDataLoader:
    """Load and preprocess mammogram density images"""
    
    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.train_csv = os.path.join(dataset_path, 'train.csv')
        self.test_csv = os.path.join(dataset_path, 'test.csv')
        
    def load_data(self):
        """Load training and test data"""
        print("Loading data...")
        
        # Load CSV files
        train_df = pd.read_csv(self.train_csv)
        print(f"\nTraining samples: {len(train_df)}")
        print(f"Density range: {train_df['Density'].min():.1f} - {train_df['Density'].max():.1f}")
        print(f"Mean density: {train_df['Density'].mean():.2f}")
        print(f"Std density: {train_df['Density'].std():.2f}")
        
        # Distribution of density
        print(f"\nDensity distribution:")
        print(train_df['Density'].describe())
        
        return train_df
    
    def create_density_categories(self, density):
        """Convert density percentage to BI-RADS categories"""
        if density < 25:
            return 0  # Almost entirely fatty (A)
        elif density < 50:
            return 1  # Scattered fibroglandular densities (B)
        elif density < 75:
            return 2  # Heterogeneously dense (C)
        else:
            return 3  # Extremely dense (D)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            return img_array
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None


class MammogramDensityModel:
    """CNN model for mammogram density prediction"""
    
    def __init__(self, img_size=(224, 224), mode='regression'):
        self.img_size = img_size
        self.mode = mode  # 'regression' or 'classification'
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN architecture"""
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
        ])
        
        if self.mode == 'regression':
            model.add(layers.Dense(1, activation='linear'))  # Output: density percentage
        else:
            model.add(layers.Dense(4, activation='softmax'))  # Output: 4 BI-RADS categories
        
        self.model = model
        
        print("\nModel Architecture:")
        self.model.summary()
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        if self.mode == 'regression':
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae', 'mse']
            )
        else:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print("\nModel compiled successfully!")
    
    def create_generators(self, train_df, val_df, batch_size=32):
        """Create data generators"""
        dataset_path = "Mammogram Density Assessment Dataset"
        train_image_dir = os.path.join(dataset_path, "train/images")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        if self.mode == 'regression':
            train_generator = train_datagen.flow_from_dataframe(
                train_df,
                directory=train_image_dir,
                x_col='Filename',
                y_col='Density',
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='raw',
                shuffle=True
            )
            
            val_generator = val_datagen.flow_from_dataframe(
                val_df,
                directory=train_image_dir,
                x_col='Filename',
                y_col='Density',
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='raw',
                shuffle=False
            )
        else:
            train_generator = train_datagen.flow_from_dataframe(
                train_df,
                directory=train_image_dir,
                x_col='Filename',
                y_col='Category',
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='raw',
                shuffle=True
            )
            
            val_generator = val_datagen.flow_from_dataframe(
                val_df,
                directory=train_image_dir,
                x_col='Filename',
                y_col='Category',
                target_size=self.img_size,
                batch_size=batch_size,
                class_mode='raw',
                shuffle=False
            )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model"""
        print(f"\nStarting training for {epochs} epochs...")
        
        callbacks = [
            ModelCheckpoint(
                'density_model_best.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
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
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def evaluate(self, val_generator, val_df):
        """Evaluate the model"""
        print("\nEvaluating model...")
        
        # Get predictions
        predictions = self.model.predict(val_generator, verbose=1)
        
        if self.mode == 'regression':
            y_true = val_df['Density'].values
            y_pred = predictions.flatten()
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            print(f"\nRegression Metrics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R² Score: {r2:.4f}")
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'predictions': y_pred.tolist()[:100],  # Save first 100
                'true_values': y_true.tolist()[:100]
            }
        else:
            y_true = val_df['Category'].values
            y_pred = np.argmax(predictions, axis=1)
            
            accuracy = np.mean(y_true == y_pred)
            
            print(f"\nClassification Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': float(accuracy),
                'predictions': y_pred.tolist()[:100],
                'true_values': y_true.tolist()[:100]
            }
    
    def plot_training_history(self, save_path='density_training_history.png'):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE or Accuracy
        if self.mode == 'regression':
            axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
            axes[1].plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
            axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('MAE')
        else:
            axes[1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
            axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
            axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Accuracy')
        
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to {save_path}")
        plt.close()
    
    def plot_predictions(self, results, save_path='density_predictions.png'):
        """Plot predictions vs true values"""
        if self.mode != 'regression':
            return
        
        y_true = results['true_values']
        y_pred = results['predictions']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=50)
        axes[0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('True Density (%)', fontsize=12)
        axes[0].set_ylabel('Predicted Density (%)', fontsize=12)
        axes[0].set_title('Predictions vs True Values', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.array(y_pred) - np.array(y_true)
        axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Prediction Error (%)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath='mammogram_density_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
        
        metadata = {
            'img_size': list(self.img_size),
            'mode': self.mode,
            'trained_at': datetime.now().isoformat(),
            'model_type': 'CNN (MobileNetV2)',
            'framework': 'TensorFlow/Keras',
            'dataset': 'Mammogram Density Assessment'
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("Mammogram Density Assessment Training Pipeline")
    print("=" * 80)
    
    # Configuration
    DATASET_PATH = "Mammogram Density Assessment Dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 50
    MODE = 'regression'  # or 'classification'
    VAL_SPLIT = 0.15
    
    # Initialize data loader
    print("\n[1/6] Loading dataset...")
    data_loader = MammogramDensityDataLoader(DATASET_PATH, img_size=IMG_SIZE)
    train_df = data_loader.load_data()
    
    if len(train_df) == 0:
        print("Error: No data loaded")
        return
    
    # Add category column if using classification
    if MODE == 'classification':
        train_df['Category'] = train_df['Density'].apply(data_loader.create_density_categories)
        print(f"\nCategory distribution:")
        print(train_df['Category'].value_counts().sort_index())
    
    # Split data
    print("\n[2/6] Splitting dataset...")
    train_data, val_data = train_test_split(
        train_df, 
        test_size=VAL_SPLIT, 
        random_state=42,
        stratify=train_df['Category'] if MODE == 'classification' else None
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Initialize model
    print("\n[3/6] Building model...")
    model = MammogramDensityModel(img_size=IMG_SIZE, mode=MODE)
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Create generators
    print("\n[4/6] Creating data generators...")
    train_gen, val_gen = model.create_generators(train_data, val_data, batch_size=BATCH_SIZE)
    
    # Train model
    print("\n[5/6] Training model...")
    history = model.train(train_gen, val_gen, epochs=EPOCHS)
    
    # Evaluate model
    print("\n[6/6] Evaluating model...")
    results = model.evaluate(val_gen, val_data)
    
    # Plot results
    print("\nGenerating visualizations...")
    model.plot_training_history('density_training_history.png')
    if MODE == 'regression':
        model.plot_predictions(results, 'density_predictions.png')
    
    # Save model
    print("\nSaving model...")
    model.save_model('mammogram_density_model.h5')
    
    # Save training results
    training_results = {
        'dataset': 'Mammogram Density Assessment',
        'total_samples': len(train_df),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'mode': MODE,
        'metrics': results,
        'trained_at': datetime.now().isoformat()
    }
    
    with open('density_training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Pipeline Completed!")
    print("=" * 80)
    
    if MODE == 'regression':
        print(f"\nFinal Results:")
        print(f"  MAE: {results['mae']:.4f}%")
        print(f"  RMSE: {results['rmse']:.4f}%")
        print(f"  R² Score: {results['r2_score']:.4f}")
    else:
        print(f"\nFinal Accuracy: {results['accuracy']:.4f}")
    
    print(f"\nFiles generated:")
    print(f"  - mammogram_density_model.h5")
    print(f"  - mammogram_density_model_metadata.json")
    print(f"  - density_model_best.h5")
    print(f"  - density_training_results.json")
    print(f"  - density_training_history.png")
    if MODE == 'regression':
        print(f"  - density_predictions.png")


if __name__ == "__main__":
    main()
