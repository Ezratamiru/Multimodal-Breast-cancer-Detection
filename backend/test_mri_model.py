"""Test script for MRI model inference"""

from mri_model import MRIBreastCancerPredictor
import os
from pathlib import Path

def test_mri_model():
    print("=" * 60)
    print("Testing MRI Breast Cancer Classification Model")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MRIBreastCancerPredictor()
    
    # Load model
    try:
        predictor.load_model()
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get model info
    info = predictor.get_model_info()
    print(f"Model Info:")
    print(f"  Type: {info['model_type']}")
    print(f"  Classes: {info['classes']}")
    print(f"  Input Size: {info['input_size']}")
    print()
    
    # Find sample images
    data_dir = "LA-Breast DCE-MRI Dataset/breas_data/breast_data"
    if not os.path.exists(data_dir):
        print(f"⚠️ Data directory not found: {data_dir}")
        return
    
    # Get some sample images
    image_files = list(Path(data_dir).rglob('*.tiff'))[:5]
    
    if not image_files:
        print("⚠️ No TIFF images found in dataset")
        return
    
    print(f"Testing on {len(image_files)} sample images:\n")
    
    for img_path in image_files:
        try:
            result = predictor.predict(str(img_path))
            
            print(f"📊 {img_path.name}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"     {cls}: {prob:.2%}")
            print()
            
        except Exception as e:
            print(f"❌ Error predicting {img_path.name}: {e}\n")
    
    print("=" * 60)
    print("✅ Testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_mri_model()
