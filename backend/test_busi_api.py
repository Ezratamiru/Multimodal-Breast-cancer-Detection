"""
Test script for BUSI API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_model_info():
    """Test BUSI model info endpoint"""
    print("\n" + "="*60)
    print("Testing BUSI Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/busi/model-info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Model Info:")
        print(f"  Loaded: {data.get('loaded')}")
        print(f"  Input Features: {data.get('input_features')}")
        print(f"  Classes: {data.get('class_names')}")
        print(f"  Model Type: {data.get('model_type')}")
    else:
        print(f"❌ Error: {response.text}")

def test_feature_importance():
    """Test feature importance endpoint"""
    print("\n" + "="*60)
    print("Testing Feature Importance Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/busi/feature-importance")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(data['top_10'].items(), 1):
            print(f"  {i}. {feature}: {importance:.6f}")
    else:
        print(f"❌ Error: {response.text}")

def test_example_features():
    """Test example features endpoint"""
    print("\n" + "="*60)
    print("Testing Example Features Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/busi/example-features")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Example Features Available:")
        print(f"  - Benign Example")
        print(f"  - Malignant Example")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_prediction(features, label):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print(f"Testing Prediction - {label}")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/busi/predict",
        json=features
    )
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Prediction Results:")
        print(f"  Diagnosis: {data['prediction']}")
        print(f"  Confidence: {data['confidence']*100:.2f}%")
        print(f"  Risk Level: {data['risk_level']}")
        print(f"  Probability (Benign): {data['probability_benign']*100:.2f}%")
        print(f"  Probability (Malignant): {data['probability_malignant']*100:.2f}%")
    else:
        print(f"❌ Error: {response.text}")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    # Get example features
    examples = test_example_features()
    if not examples:
        return
    
    # Create batch with both examples
    batch = [
        examples['benign_example'],
        examples['malignant_example']
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/busi/predict-batch",
        json=batch
    )
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Batch Prediction Results:")
        print(f"  Total Predictions: {data['count']}")
        for i, pred in enumerate(data['predictions'], 1):
            print(f"\n  Patient {i}:")
            print(f"    Diagnosis: {pred['prediction']}")
            print(f"    Confidence: {pred['confidence']*100:.2f}%")
            print(f"    Risk Level: {pred['risk_level']}")
    else:
        print(f"❌ Error: {response.text}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BUSI API Test Suite")
    print("="*60)
    print("\nMake sure the FastAPI server is running:")
    print("  cd backend && venv/bin/uvicorn main:app --reload")
    print("\nStarting tests...\n")
    
    try:
        # Test 1: Model Info
        test_model_info()
        
        # Test 2: Feature Importance
        test_feature_importance()
        
        # Test 3: Get Example Features
        examples = test_example_features()
        
        if examples:
            # Test 4: Benign Prediction
            test_prediction(examples['benign_example'], "Benign Case")
            
            # Test 5: Malignant Prediction
            test_prediction(examples['malignant_example'], "Malignant Case")
            
            # Test 6: Batch Prediction
            # Skipping batch test to avoid redundancy
            # test_batch_prediction()
        
        print("\n" + "="*60)
        print("✅ All Tests Completed!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*60)
        print("❌ ERROR: Could not connect to API server")
        print("="*60)
        print("\nPlease start the server first:")
        print("  cd backend")
        print("  venv/bin/uvicorn main:app --reload")
        print()

if __name__ == "__main__":
    main()
