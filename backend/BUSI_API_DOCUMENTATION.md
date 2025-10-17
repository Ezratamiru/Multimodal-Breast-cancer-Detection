# BUSI Breast Cancer API Documentation

## Overview
REST API endpoints for breast cancer prediction using the BUSI-WHU trained model. The model analyzes 30 clinical features to predict whether a breast tumor is benign or malignant.

---

## Base URL
```
http://localhost:8000
```

---

## Endpoints

### 1. Get Model Information
Retrieve information about the loaded BUSI model.

**Endpoint:** `GET /api/busi/model-info`

**Response:**
```json
{
  "loaded": true,
  "model_path": "busi_breast_classifier.h5",
  "input_features": 30,
  "feature_names": ["radius_mean", "texture_mean", ...],
  "class_names": ["Benign", "Malignant"],
  "model_type": "Deep Neural Network",
  "framework": "TensorFlow/Keras",
  "dataset": "BUSI-WHU Breast Cancer",
  "trained_at": "2025-10-06T12:57:27.683372"
}
```

**Example:**
```bash
curl http://localhost:8000/api/busi/model-info
```

---

### 2. Get Feature Importance
Retrieve feature importance scores from the model.

**Endpoint:** `GET /api/busi/feature-importance`

**Response:**
```json
{
  "feature_importance": {
    "perimeter_mean": 0.0366,
    "area_mean": 0.0364,
    "smoothness_se": 0.0353,
    ...
  },
  "top_10": {
    "perimeter_mean": 0.0366,
    "area_mean": 0.0364,
    ...
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/busi/feature-importance
```

---

### 3. Get Example Features
Retrieve example feature sets for testing (one benign, one malignant).

**Endpoint:** `GET /api/busi/example-features`

**Response:**
```json
{
  "benign_example": {
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    "perimeter_mean": 87.46,
    ...
  },
  "malignant_example": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    ...
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/busi/example-features
```

---

### 4. Predict Single Patient
Make a prediction for a single patient based on clinical features.

**Endpoint:** `POST /api/busi/predict`

**Request Body:**
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.6,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "compactness_worst": 0.6656,
  "concavity_worst": 0.7119,
  "concave_points_worst": 0.2654,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.1189
}
```

**Query Parameters (Optional):**
- `patient_id` (string): Patient identifier

**Response:**
```json
{
  "prediction": "Malignant",
  "prediction_class": 1,
  "probability_benign": 0.0024,
  "probability_malignant": 0.9976,
  "confidence": 0.9976,
  "risk_level": "Very High",
  "patient_id": null,
  "created_at": "2025-10-06T13:00:00.000000"
}
```

**Risk Levels:**
- `Low`: Malignancy probability < 30%
- `Moderate`: Malignancy probability 30-60%
- `High`: Malignancy probability 60-80%
- `Very High`: Malignancy probability > 80%

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/busi/predict \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }'
```

**Example (Python):**
```python
import requests

features = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    # ... all 30 features
}

response = requests.post(
    "http://localhost:8000/api/busi/predict",
    json=features
)

result = response.json()
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

---

### 5. Batch Prediction
Make predictions for multiple patients at once.

**Endpoint:** `POST /api/busi/predict-batch`

**Request Body:**
```json
[
  {
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    ...
  },
  {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    ...
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "Benign",
      "prediction_class": 0,
      "probability_benign": 0.9845,
      "probability_malignant": 0.0155,
      "confidence": 0.9845,
      "risk_level": "Low",
      "created_at": "2025-10-06T13:00:00.000000"
    },
    {
      "prediction": "Malignant",
      "prediction_class": 1,
      "probability_benign": 0.0024,
      "probability_malignant": 0.9976,
      "confidence": 0.9976,
      "risk_level": "Very High",
      "created_at": "2025-10-06T13:00:00.000000"
    }
  ],
  "count": 2
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/busi/predict-batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "radius_mean": 13.54,
      "texture_mean": 14.36,
      ...
    },
    {
      "radius_mean": 17.99,
      "texture_mean": 10.38,
      ...
    }
  ]'
```

---

## Feature Descriptions

### Mean Features
Computed from a digitized image of a fine needle aspirate (FNA) of a breast mass:

1. **radius_mean**: Mean of distances from center to points on the perimeter
2. **texture_mean**: Standard deviation of gray-scale values
3. **perimeter_mean**: Mean size of the core tumor
4. **area_mean**: Mean area of the tumor
5. **smoothness_mean**: Mean of local variation in radius lengths
6. **compactness_mean**: Mean of perimeter² / area - 1.0
7. **concavity_mean**: Mean of severity of concave portions of the contour
8. **concave_points_mean**: Mean number of concave portions of the contour
9. **symmetry_mean**: Mean symmetry of the tumor
10. **fractal_dimension_mean**: Mean "coastline approximation" - 1

### Standard Error (SE) Features
Standard error of the above 10 features (features 11-20)

### Worst Features
"Worst" or largest mean value of the above features (features 21-30)

---

## Error Responses

### 503 Service Unavailable
Model not loaded or unavailable.
```json
{
  "detail": "BUSI model not available. Please check model files."
}
```

### 500 Internal Server Error
Prediction or processing error.
```json
{
  "detail": "Prediction failed: <error message>"
}
```

### 422 Validation Error
Invalid input features.
```json
{
  "detail": [
    {
      "loc": ["body", "radius_mean"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Model Performance

- **Accuracy**: 97.67%
- **Precision**: 96.88%
- **Recall**: 96.88%
- **AUC-ROC**: 99.83%

---

## Integration Examples

### JavaScript/React
```javascript
const predictBreastCancer = async (features) => {
  try {
    const response = await fetch('http://localhost:8000/api/busi/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(features),
    });
    
    const result = await response.json();
    console.log('Prediction:', result.prediction);
    console.log('Confidence:', (result.confidence * 100).toFixed(2) + '%');
    return result;
  } catch (error) {
    console.error('Prediction failed:', error);
  }
};
```

### Python
```python
import requests

def predict_breast_cancer(features):
    url = "http://localhost:8000/api/busi/predict"
    response = requests.post(url, json=features)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        return result
    else:
        print(f"Error: {response.text}")
        return None
```

### cURL
```bash
# Get model info
curl http://localhost:8000/api/busi/model-info

# Get example features
curl http://localhost:8000/api/busi/example-features

# Make prediction (using example features)
curl -X POST http://localhost:8000/api/busi/predict \
  -H "Content-Type: application/json" \
  -d @benign_example.json
```

---

## Testing

Run the test suite:
```bash
cd backend
venv/bin/python test_busi_api.py
```

Make sure the FastAPI server is running:
```bash
cd backend
venv/bin/uvicorn main:app --reload
```

---

## Notes

- All 30 features are required for prediction
- Feature values should be properly scaled (the API handles scaling automatically)
- Predictions return probabilities for both classes
- The model was trained on Wisconsin Breast Cancer Dataset features
- For production use, consider implementing:
  - Authentication/Authorization
  - Rate limiting
  - Request logging
  - Input validation enhancements
  - HTTPS encryption

---

## Support

For issues or questions:
- Check model files exist: `busi_breast_classifier.h5`, `busi_scaler.pkl`
- Review training results: `busi_training_results.json`
- Test inference module: `python busi_model.py`
- Check API logs when running the server

---

**API Version**: 1.0  
**Last Updated**: October 6, 2025
