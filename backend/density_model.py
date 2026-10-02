"""
Mammogram Density Assessment Inference Module
Provides prediction capabilities for breast density estimation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os
from typing import Union, Dict

class MammogramDensityPredictor:
    """Predictor for mammogram breast density assessment"""
    
    def __init__(self, model_path='mammogram_density_model.h5',
                 metadata_path='mammogram_density_model_metadata.json'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.img_size = (224, 224)
        
        # BI-RADS density categories
        self.density_categories = {
            'A': 'Almost entirely fatty (< 25%)',
            'B': 'Scattered fibroglandular densities (25-50%)',
            'C': 'Heterogeneously dense (50-75%)',
            'D': 'Extremely dense (> 75%)'
        }
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model without compiling (avoids metric deserialization issues)
        self.model = keras.models.load_model(self.model_path, compile=False)
        # Recompile with simple config for inference
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(f"✅ Loaded density model from {self.model_path}")
        
        # Load metadata if available
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.img_size = tuple(self.metadata.get('img_size', [224, 224]))
            print(f"✅ Loaded metadata from {self.metadata_path}")
        
        return True
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image_input: Can be:
                - Path to image file (str)
                - PIL Image
                - NumPy array
        
        Returns:
            Preprocessed image array
        """
        # Load image if path provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype('uint8'))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def get_birads_category(self, density: float) -> str:
        """Convert density percentage to BI-RADS category"""
        if density < 25:
            return 'A'
        elif density < 50:
            return 'B'
        elif density < 75:
            return 'C'
        else:
            return 'D'
    
    def get_risk_assessment(self, density: float) -> str:
        """Assess cancer risk based on breast density"""
        if density < 25:
            return 'Low'
        elif density < 50:
            return 'Moderate'
        elif density < 75:
            return 'High'
        else:
            return 'Very High'
    
    def get_recommendations(self, density: float) -> list:
        """Get clinical recommendations based on density"""
        category = self.get_birads_category(density)
        
        recommendations = {
            'A': [
                "Standard mammography screening is appropriate",
                "Continue routine annual or biennial screening"
            ],
            'B': [
                "Standard mammography screening is appropriate",
                "Consider supplemental screening if at high risk"
            ],
            'C': [
                "Discuss supplemental screening options with physician",
                "Consider ultrasound or MRI if at elevated risk",
                "More frequent monitoring may be beneficial"
            ],
            'D': [
                "Supplemental screening strongly recommended",
                "Consider breast MRI or ultrasound in addition to mammography",
                "Increased surveillance recommended",
                "Discuss with radiologist about optimal screening strategy"
            ]
        }
        
        return recommendations.get(category, [])
    
    def predict(self, image_input: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Predict breast density from mammogram image
        
        Args:
            image_input: Image as file path, PIL Image, or numpy array
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        img_array = self.preprocess_image(image_input)
        
        # Make prediction
        density_pred = self.model.predict(img_array, verbose=0)[0][0]
        
        # Ensure prediction is within valid range
        density_pred = float(np.clip(density_pred, 0, 100))
        
        # Get BI-RADS category
        category = self.get_birads_category(density_pred)
        
        # Prepare result
        result = {
            'density_percentage': round(density_pred, 2),
            'birads_category': category,
            'category_description': self.density_categories[category],
            'risk_assessment': self.get_risk_assessment(density_pred),
            'recommendations': self.get_recommendations(density_pred),
            'interpretation': self._get_interpretation(density_pred, category)
        }
        
        return result
    
    def _get_interpretation(self, density: float, category: str) -> str:
        """Generate clinical interpretation"""
        interpretations = {
            'A': f"The breast tissue shows minimal density ({density:.1f}%). This composition allows for excellent visualization of masses and other abnormalities.",
            'B': f"The breast tissue shows scattered areas of fibroglandular density ({density:.1f}%). Masses should be well visualized.",
            'C': f"The breast tissue is heterogeneously dense ({density:.1f}%). This may lower the sensitivity of mammography and obscure small masses.",
            'D': f"The breast tissue is extremely dense ({density:.1f}%). This significantly lowers mammography sensitivity and may obscure masses."
        }
        
        return interpretations.get(category, f"Breast density estimated at {density:.1f}%.")
    
    def predict_batch(self, image_inputs: list) -> list:
        """Make predictions for multiple images"""
        results = []
        for img_input in image_inputs:
            try:
                result = self.predict(img_input)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'loaded': False, 'message': 'Model not loaded'}
        
        info = {
            'loaded': True,
            'model_path': self.model_path,
            'input_size': self.img_size,
            'model_type': 'CNN (MobileNetV2)',
            'framework': 'TensorFlow/Keras',
            'task': 'Breast Density Assessment (Regression)',
            'output': 'Density percentage (0-100%)',
            'birads_categories': self.density_categories
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = MammogramDensityPredictor()
    
    # Load model
    try:
        predictor.load_model()
        print("\n✅ Model loaded successfully!")
        
        # Get model info
        info = predictor.get_model_info()
        print(f"\nModel Info:")
        print(f"  Input Size: {info['input_size']}")
        print(f"  Model Type: {info['model_type']}")
        print(f"  Task: {info['task']}")
        
        # Test prediction on a sample image
        dataset_path = "Mammogram Density Assessment Dataset/train/images"
        if os.path.exists(dataset_path):
            sample_images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')][:3]
            
            if sample_images:
                print(f"\n🔬 Testing predictions on {len(sample_images)} sample images...")
                
                for img_filename in sample_images:
                    img_path = os.path.join(dataset_path, img_filename)
                    result = predictor.predict(img_path)
                    
                    print(f"\n📊 Results for {img_filename}:")
                    print(f"  Density: {result['density_percentage']:.1f}%")
                    print(f"  BI-RADS Category: {result['birads_category']}")
                    print(f"  Description: {result['category_description']}")
                    print(f"  Risk Assessment: {result['risk_assessment']}")
                    print(f"  Recommendations:")
                    for rec in result['recommendations']:
                        print(f"    - {rec}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
