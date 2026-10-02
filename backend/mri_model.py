"""LA-Breast DCE-MRI Model Inference Module"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import os
from typing import Union, Dict, List

class MRIBreastCancerPredictor:
    """Predictor for MRI breast cancer classification using extracted features"""
    
    def __init__(self, model_path='LA-Breast DCE-MRI Dataset/my_breast_cancer_model.h5'):
        self.model_path = model_path
        self.model = None
        self.num_features = 76
        self.class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4']
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"✅ Loaded MRI model from {self.model_path}")
        return True
    
    def predict(self, features: Union[List[float], np.ndarray, Dict[str, float]]) -> Dict:
        """Predict breast cancer from extracted features
        
        Args:
            features: Either a list/array of 76 features or a dict with feature names
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert features to numpy array
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        elif isinstance(features, list):
            feature_array = np.array(features)
        else:
            feature_array = features
        
        # Validate shape
        if feature_array.shape[0] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {feature_array.shape[0]}")
        
        # Reshape for prediction
        feature_array = feature_array.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(feature_array, verbose=0)[0]
        
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])
        
        result = {
            'prediction': self.class_names[predicted_class],
            'prediction_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'Normal': float(predictions[0]),
                'Benign': float(predictions[1]),
                'Malignant': float(predictions[2])
            },
            'risk_level': self._get_risk_level(predicted_class, confidence)
        }
        
        return result
    
    def _get_risk_level(self, pred_class, confidence):
        """Determine risk level based on class"""
        risk_mapping = {
            0: 'Low',
            1: 'Low-Moderate',
            2: 'Moderate',
            3: 'Moderate-High',
            4: 'High'
        }
        return risk_mapping.get(pred_class, 'Unknown')
    
    def load_features_from_csv(self, csv_path: str, row_index: int = 0) -> np.ndarray:
        """Load features from CSV file
        
        Args:
            csv_path: Path to CSV file
            row_index: Row index to load features from
        """
        df = pd.read_csv(csv_path)
        # Get only numeric columns and convert to float
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = df.iloc[row_index][numeric_cols].values.astype(float)[:self.num_features]
        return features
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'input_features': self.num_features,
            'model_type': 'Dense Neural Network',
            'framework': 'TensorFlow/Keras',
            'task': 'MRI Feature-based Classification',
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'note': 'Model requires 76 pre-extracted features, not raw images'
        }

if __name__ == "__main__":
    predictor = MRIBreastCancerPredictor()
    try:
        predictor.load_model()
        print("✅ Model loaded successfully!")
        info = predictor.get_model_info()
        print(f"\nModel Info:")
        print(f"  Input Features: {info['input_features']}")
        print(f"  Classes: {info['classes']}")
        print(f"  Note: {info['note']}")
        
        # Test with sample features from CSV
        csv_path = "LA-Breast DCE-MRI Dataset/breas_data/breast_data/metadata/test.csv"
        if os.path.exists(csv_path):
            print(f"\n🔬 Testing with sample from CSV...")
            features = predictor.load_features_from_csv(csv_path, row_index=0)
            result = predictor.predict(features)
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
