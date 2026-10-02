import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, Upload, Brain, Activity, CheckCircle, AlertCircle, Play, Download, Eye } from 'lucide-react'

const MLSimulator = () => {
  const [modelInfo, setModelInfo] = useState(null)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [sampleImages, setSampleImages] = useState([])
  const [selectedImage, setSelectedImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState(null)

  useEffect(() => {
    fetchModelInfo()
    fetchTrainingStatus()
    fetchSampleImages()
  }, [])

  const fetchModelInfo = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/ml/model-info')
      const data = await response.json()
      setModelInfo(data)
    } catch (error) {
      console.error('Error fetching model info:', error)
    }
  }

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/ml/training-status')
      const data = await response.json()
      setTrainingStatus(data)
    } catch (error) {
      console.error('Error fetching training status:', error)
    }
  }

  const fetchSampleImages = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/ml/sample-images')
      const data = await response.json()
      setSampleImages(data.images || [])
    } catch (error) {
      console.error('Error fetching sample images:', error)
    }
  }

  const startTraining = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8000/api/ml/train', {
        method: 'POST'
      })
      const data = await response.json()
      setTrainingStatus(data)
      
      // Poll for training updates
      const pollInterval = setInterval(async () => {
        const statusResponse = await fetch('http://localhost:8000/api/ml/training-status')
        const statusData = await statusResponse.json()
        setTrainingStatus(statusData)
        
        if (statusData.status === 'completed' || statusData.status === 'failed') {
          clearInterval(pollInterval)
          fetchModelInfo() // Refresh model info
        }
      }, 1000)
      
    } catch (error) {
      console.error('Error starting training:', error)
    } finally {
      setLoading(false)
    }
  }

  const predictSampleImage = async (imageId) => {
    try {
      setLoading(true)
      const response = await fetch(`http://localhost:8000/api/ml/predict-sample/${imageId}`, {
        method: 'POST'
      })
      const data = await response.json()
      setPrediction(data)
      setSelectedImage(sampleImages.find(img => img.id === imageId))
    } catch (error) {
      console.error('Error predicting image:', error)
    } finally {
      setLoading(false)
    }
  }

  const predictUploadedImage = async () => {
    if (!uploadedFile) return

    try {
      setLoading(true)
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const response = await fetch('http://localhost:8000/api/ml/predict', {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      setPrediction(data)
      setSelectedImage({ image_data: URL.createObjectURL(uploadedFile), filename: uploadedFile.name })
    } catch (error) {
      console.error('Error predicting uploaded image:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setUploadedFile(file)
      setPrediction(null)
      setSelectedImage(null)
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getPredictionIcon = (prediction) => {
    if (prediction === 'Malignant') return <AlertCircle className="h-5 w-5 text-red-400" />
    return <CheckCircle className="h-5 w-5 text-green-400" />
  }

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center space-x-6 mb-8">
          <Link 
            to="/" 
            className="medical-btn medical-btn-arrow group"
          >
            <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform duration-200" />
            Back to Dashboard
          </Link>
          
          <div className="flex items-center space-x-4">
            <div className="p-4 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl">
              <Brain className="h-8 w-8 text-purple-400" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white">
                ML Simulator
              </h1>
              <p className="text-gray-400 text-lg">
                Train and test mammogram classification models
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Model Status */}
          <div className="medical-card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white">Model Status</h2>
              <Activity className="h-5 w-5 text-blue-400" />
            </div>

            {modelInfo && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Model Loaded:</span>
                  <div className="flex items-center space-x-2">
                    {modelInfo.model_loaded ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-green-400" />
                        <span className="text-green-400">Yes</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-4 w-4 text-red-400" />
                        <span className="text-red-400">No</span>
                      </>
                    )}
                  </div>
                </div>

                {modelInfo.model_loaded && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Image Size:</span>
                      <span className="text-white">{modelInfo.image_size?.join('x')}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Classes:</span>
                      <span className="text-white">{modelInfo.class_names?.join(', ')}</span>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Training Section */}
            <div className="mt-6 pt-6 border-t border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Training</h3>
              
              {trainingStatus && (
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-400">Status:</span>
                    <span className={`font-medium ${
                      trainingStatus.status === 'completed' ? 'text-green-400' :
                      trainingStatus.status === 'training' ? 'text-blue-400' :
                      trainingStatus.status === 'failed' ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {trainingStatus.status}
                    </span>
                  </div>
                  
                  {trainingStatus.progress !== undefined && (
                    <div className="mb-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-400">Progress:</span>
                        <span className="text-white">{trainingStatus.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${trainingStatus.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  <p className="text-sm text-gray-400">{trainingStatus.message}</p>
                  
                  {trainingStatus.results && (
                    <div className="mt-4 p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                      <h4 className="text-green-400 font-medium mb-2">Training Results:</h4>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>Accuracy: <span className="text-white">{(trainingStatus.results.accuracy * 100).toFixed(1)}%</span></div>
                        <div>Precision: <span className="text-white">{(trainingStatus.results.precision * 100).toFixed(1)}%</span></div>
                        <div>Recall: <span className="text-white">{(trainingStatus.results.recall * 100).toFixed(1)}%</span></div>
                        <div>F1-Score: <span className="text-white">{(trainingStatus.results.f1_score * 100).toFixed(1)}%</span></div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <button 
                onClick={startTraining}
                disabled={loading || trainingStatus?.status === 'training'}
                className="medical-btn medical-btn-primary w-full"
              >
                <Play className="h-4 w-4" />
                {trainingStatus?.status === 'training' ? 'Training...' : 'Start Training'}
              </button>
            </div>
          </div>

          {/* Image Upload & Prediction */}
          <div className="medical-card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white">Image Prediction</h2>
              <Upload className="h-5 w-5 text-blue-400" />
            </div>

            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Upload Mammogram Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-500/20 file:text-blue-400 hover:file:bg-blue-500/30"
              />
            </div>

            {uploadedFile && (
              <button
                onClick={predictUploadedImage}
                disabled={loading || !modelInfo?.model_loaded}
                className="medical-btn medical-btn-primary w-full mb-6"
              >
                <Brain className="h-4 w-4" />
                {loading ? 'Predicting...' : 'Predict Image'}
              </button>
            )}

            {/* Selected Image Display */}
            {selectedImage && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Selected Image</h3>
                <div className="bg-gray-800 rounded-lg p-4">
                  <img 
                    src={selectedImage.image_data} 
                    alt={selectedImage.filename}
                    className="w-full h-48 object-cover rounded-lg mb-3"
                  />
                  <p className="text-sm text-gray-400">{selectedImage.filename}</p>
                </div>
              </div>
            )}

            {/* Prediction Results */}
            {prediction && (
              <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-3">Prediction Results</h3>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Prediction:</span>
                    <div className="flex items-center space-x-2">
                      {getPredictionIcon(prediction.prediction)}
                      <span className="font-medium text-white">{prediction.prediction}</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Confidence:</span>
                    <span className={`font-medium ${getConfidenceColor(prediction.confidence)}`}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="text-center p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                      <div className="text-green-400 font-medium">Benign</div>
                      <div className="text-white text-lg">{(prediction.probability_benign * 100).toFixed(1)}%</div>
                    </div>
                    <div className="text-center p-3 bg-red-500/10 rounded-lg border border-red-500/20">
                      <div className="text-red-400 font-medium">Malignant</div>
                      <div className="text-white text-lg">{(prediction.probability_malignant * 100).toFixed(1)}%</div>
                    </div>
                  </div>

                  {prediction.true_label && (
                    <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                      <div className="flex items-center justify-between">
                        <span className="text-blue-400">True Label:</span>
                        <span className="text-white font-medium">{prediction.true_label}</span>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-blue-400">Correct:</span>
                        <span className={`font-medium ${
                          prediction.prediction === prediction.true_label ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {prediction.prediction === prediction.true_label ? 'Yes' : 'No'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Sample Images */}
          <div className="medical-card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white">Sample Images</h2>
              <Eye className="h-5 w-5 text-blue-400" />
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {sampleImages.map((image) => (
                <div key={image.id} className="p-3 bg-gray-800/50 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors">
                  <div className="flex items-center space-x-3">
                    <img 
                      src={image.image_data} 
                      alt={image.filename}
                      className="w-16 h-16 object-cover rounded-lg"
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">{image.filename}</p>
                      <p className="text-xs text-gray-400">True: {image.true_label}</p>
                    </div>
                    <button
                      onClick={() => predictSampleImage(image.id)}
                      disabled={loading || !modelInfo?.model_loaded}
                      className="medical-btn medical-btn-secondary text-xs px-3 py-1"
                    >
                      Test
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {sampleImages.length === 0 && (
              <div className="text-center py-8">
                <AlertCircle className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">No sample images available</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MLSimulator
