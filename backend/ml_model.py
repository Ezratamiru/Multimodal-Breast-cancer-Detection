import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import pickle
from datetime import datetime

class MammogramClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['Benign', 'Malignant']
        
    def create_model(self):
        """Create a CNN model for mammogram classification"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Rescaling
            layers.Rescaling(1./255),
            
            # Convolutional blocks
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data_dir):
        """Prepare data generators for training"""
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def train(self, data_dir, epochs=50):
        """Train the model"""
        print("Preparing data...")
        train_gen, val_gen, test_gen = self.prepare_data(data_dir)
        
        print("Creating model...")
        if self.model is None:
            self.create_model()
        
        print(f"Model summary:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_mammogram_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, data_dir):
        """Evaluate the model on test data"""
        _, _, test_gen = self.prepare_data(data_dir)
        
        print("Evaluating model...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(test_gen, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_gen)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_gen.classes
        
        # Classification report
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=self.class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        results = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'f1_score': report['macro avg']['f1-score']
        }
        
        print(f"Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        return results
    
    def predict_image(self, image_path):
        """Predict a single image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_array)[0][0]
        predicted_class = int(prediction > 0.5)
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        result = {
            'prediction': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probability_malignant': float(prediction),
            'probability_benign': float(1 - prediction)
        }
        
        return result
    
    def save_model(self, filepath='mammogram_classifier.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'class_names': self.class_names,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath.replace('.h5', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='mammogram_classifier.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.img_size = tuple(metadata['img_size'])
                self.batch_size = metadata['batch_size']
                self.class_names = metadata['class_names']
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training script"""
    # Initialize classifier
    classifier = MammogramClassifier(img_size=(224, 224), batch_size=32)
    
    # Data directory
    data_dir = "Mammogram-20250924T132236Z-1-001/Mammogram"
    
    print("Starting mammogram classification training...")
    
    # Train the model
    history = classifier.train(data_dir, epochs=50)
    
    # Evaluate the model
    results = classifier.evaluate(data_dir)
    
    # Save the model
    classifier.save_model('mammogram_classifier.h5')
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
