"""
BUSI-WHU Breast Cancer Model Inference Module
Provides prediction capabilities using the trained model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os
from typing import Dict, List, Union

class BUSIBreastCancerPredictor:
    """Predictor for BUSI breast cancer classification"""
    
    def __init__(self, model_path='busi_breast_classifier.h5', 
                 scaler_path='busi_scaler.pkl',
                 metadata_path='busi_breast_classifier_metadata.json'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.model = None
        self.scaler = None
        self.metadata = None
        self.class_names = ['Benign', 'Malignant']
        self.feature_names = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]
        
    def load_model(self):
        """Load the trained model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        
        # Load model
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Loaded BUSI model from {self.model_path}")
        
        # Load scaler
        self.scaler = joblib.load(self.scaler_path)
        print(f"✅ Loaded scaler from {self.scaler_path}")
        
        # Load metadata if available
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.class_names = self.metadata.get('class_names', self.class_names)
            print(f"✅ Loaded metadata from {self.metadata_path}")
        
        return True
    
    def predict(self, features: Union[Dict[str, float], List[float], np.ndarray]) -> Dict:
        """
        Make prediction on clinical features
        
        Args:
            features: Can be:
                - Dict mapping feature names to values
                - List of 30 feature values in order
                - NumPy array of 30 features
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert features to numpy array
        if isinstance(features, dict):
            # Extract features in correct order
            feature_array = np.array([features.get(fname, 0.0) for fname in self.feature_names])
        elif isinstance(features, list):
            feature_array = np.array(features)
        elif isinstance(features, np.ndarray):
            feature_array = features
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")
        
        # Ensure correct shape
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        
        # Handle case where extra column exists
        if feature_array.shape[1] == 31:
            feature_array = feature_array[:, :30]
        
        # Validate feature count
        if feature_array.shape[1] != 30:
            raise ValueError(f"Expected 30 features, got {feature_array.shape[1]}")
        
        # Scale features
        features_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]
        prediction_class = int(prediction_prob > 0.5)
        
        # Prepare result
        result = {
            'prediction': self.class_names[prediction_class],
            'prediction_class': prediction_class,
            'probability_benign': float(1 - prediction_prob),
            'probability_malignant': float(prediction_prob),
            'confidence': float(max(prediction_prob, 1 - prediction_prob)),
            'risk_level': self._get_risk_level(prediction_prob)
        }
        
        return result
    
    def _get_risk_level(self, malignant_prob: float) -> str:
        """Determine risk level based on malignancy probability"""
        if malignant_prob < 0.3:
            return 'Low'
        elif malignant_prob < 0.6:
            return 'Moderate'
        elif malignant_prob < 0.8:
            return 'High'
        else:
            return 'Very High'
    
    def predict_batch(self, features_batch: List[Union[Dict, List, np.ndarray]]) -> List[Dict]:
        """Make predictions for multiple samples"""
        results = []
        for features in features_batch:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (approximation using model weights)
        Note: This is a simplified approach
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get weights from first layer
        first_layer_weights = self.model.layers[0].get_weights()[0]
        
        # Calculate average absolute weight for each feature
        importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Normalize
        importance = importance / np.sum(importance)
        
        # Create feature importance dictionary
        feature_importance = {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importance)
        }
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return feature_importance
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'loaded': False, 'message': 'Model not loaded'}
        
        info = {
            'loaded': True,
            'model_path': self.model_path,
            'input_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'model_type': 'Deep Neural Network',
            'framework': 'TensorFlow/Keras'
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BUSIBreastCancerPredictor()
    
    # Load model
    try:
        predictor.load_model()
        print("\n✅ Model loaded successfully!")
        
        # Get model info
        info = predictor.get_model_info()
        print(f"\nModel Info:")
        print(f"  Features: {info['input_features']}")
        print(f"  Classes: {info['class_names']}")
        
        # Example prediction with sample features
        sample_features = {
            "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
            "area_mean": 1001, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
            "concavity_mean": 0.3001, "concave points_mean": 0.1471,
            "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
            "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589,
            "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904,
            "concavity_se": 0.05373, "concave points_se": 0.01587,
            "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
            "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6,
            "area_worst": 2019, "smoothness_worst": 0.1622, "compactness_worst": 0.6656,
            "concavity_worst": 0.7119, "concave points_worst": 0.2654,
            "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
        }
        
        print("\n🔬 Making prediction on sample data...")
        result = predictor.predict(sample_features)
        print(f"\nPrediction Results:")
        print(f"  Diagnosis: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probability (Benign): {result['probability_benign']:.2%}")
        print(f"  Probability (Malignant): {result['probability_malignant']:.2%}")
        
        # Get feature importance
        print("\n📊 Top 10 Most Important Features:")
        importance = predictor.get_feature_importance()
        for i, (feature, imp) in enumerate(list(importance.items())[:10], 1):
            print(f"  {i}. {feature}: {imp:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
