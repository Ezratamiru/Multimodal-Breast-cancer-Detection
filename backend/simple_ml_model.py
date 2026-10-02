import os
import numpy as np
import json
import random
from PIL import Image
from datetime import datetime

class SimpleMammogramClassifier:
    """
    Simplified mammogram classifier for demonstration purposes
    This uses basic image analysis instead of deep learning
    """
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.class_names = ['Benign', 'Malignant']
        self.is_trained = False
        
    def analyze_image_features(self, image_path):
        """Extract simple features from image for classification"""
        try:
            # Load and process image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize(self.img_size)
            img_array = np.array(img)
            
            # Extract simple features
            features = {
                'mean_intensity': np.mean(img_array),
                'std_intensity': np.std(img_array),
                'min_intensity': np.min(img_array),
                'max_intensity': np.max(img_array),
                'contrast': np.max(img_array) - np.min(img_array),
                'brightness': np.mean(img_array) / 255.0
            }
            
            return features
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predict a single image using simple heuristics"""
        features = self.analyze_image_features(image_path)
        
        if features is None:
            raise ValueError("Could not analyze image")
        
        # Simple heuristic-based classification
        # This is for demonstration - real models would use trained weights
        
        # Calculate a "suspicion score" based on image characteristics
        suspicion_score = 0.0
        
        # Higher contrast might indicate abnormalities
        if features['contrast'] > 150:
            suspicion_score += 0.3
        
        # Very low or very high brightness might be suspicious
        if features['brightness'] < 0.3 or features['brightness'] > 0.8:
            suspicion_score += 0.2
        
        # High standard deviation might indicate texture irregularities
        if features['std_intensity'] > 50:
            suspicion_score += 0.2
        
        # Add some randomness to simulate model uncertainty
        suspicion_score += random.uniform(-0.1, 0.1)
        
        # Ensure score is between 0 and 1
        suspicion_score = max(0.0, min(1.0, suspicion_score))
        
        # Determine prediction
        predicted_class = int(suspicion_score > 0.5)
        confidence = suspicion_score if predicted_class == 1 else 1 - suspicion_score
        
        result = {
            'prediction': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probability_malignant': float(suspicion_score),
            'probability_benign': float(1 - suspicion_score),
            'features': features
        }
        
        return result
    
    def evaluate_on_samples(self, dataset_path, num_samples=20):
        """Evaluate the model on sample images"""
        results = []
        
        for class_folder in ['0', '1']:
            class_path = os.path.join(dataset_path, 'test', class_folder)
            if not os.path.exists(class_path):
                continue
                
            true_label = 'Benign' if class_folder == '0' else 'Malignant'
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample random images
            sample_images = random.sample(images, min(num_samples // 2, len(images)))
            
            for img_name in sample_images:
                img_path = os.path.join(class_path, img_name)
                try:
                    prediction = self.predict_image(img_path)
                    prediction['true_label'] = true_label
                    prediction['filename'] = img_name
                    prediction['correct'] = prediction['prediction'] == true_label
                    results.append(prediction)
                except Exception as e:
                    print(f"Error predicting {img_name}: {e}")
        
        # Calculate metrics
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / len(results) if results else 0
        
        return {
            'results': results,
            'accuracy': accuracy,
            'total_samples': len(results),
            'correct_predictions': correct_predictions
        }
    
    def save_model(self, filepath='simple_mammogram_classifier.json'):
        """Save the model configuration"""
        model_data = {
            'model_type': 'simple_heuristic',
            'img_size': self.img_size,
            'class_names': self.class_names,
            'is_trained': True,
            'created_at': datetime.now().isoformat(),
            'description': 'Simple heuristic-based mammogram classifier for demonstration'
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='simple_mammogram_classifier.json'):
        """Load the model configuration"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
                self.img_size = tuple(model_data['img_size'])
                self.class_names = model_data['class_names']
                self.is_trained = model_data.get('is_trained', True)
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model file found at {filepath}")

def main():
    """Test the simple classifier"""
    print("🧠 Testing Simple Mammogram Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = SimpleMammogramClassifier()
    
    # Dataset path
    dataset_path = "Mammogram-20250924T132236Z-1-001/Mammogram"
    
    if os.path.exists(dataset_path):
        print("📊 Evaluating on sample images...")
        evaluation = classifier.evaluate_on_samples(dataset_path, num_samples=10)
        
        print(f"\n📈 Results:")
        print(f"   Total samples: {evaluation['total_samples']}")
        print(f"   Correct predictions: {evaluation['correct_predictions']}")
        print(f"   Accuracy: {evaluation['accuracy']:.2%}")
        
        # Show some example predictions
        print(f"\n🔍 Sample Predictions:")
        for i, result in enumerate(evaluation['results'][:5]):
            status = "✅" if result['correct'] else "❌"
            print(f"   {status} {result['filename'][:30]:<30} | True: {result['true_label']:<9} | Pred: {result['prediction']:<9} | Conf: {result['confidence']:.2%}")
        
        # Save model
        classifier.save_model()
        
    else:
        print("❌ Dataset not found. Please ensure the mammogram dataset is available.")

if __name__ == "__main__":
    main()
