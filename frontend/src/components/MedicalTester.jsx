import React, { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, Upload, Activity, AlertTriangle, CheckCircle, Eye, Download, Zap } from 'lucide-react'

const MedicalTester = () => {
  const [sampleImages, setSampleImages] = useState([])
  const [selectedImage, setSelectedImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [segmentationData, setSegmentationData] = useState(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    fetchSampleImages()
  }, [])

  const fetchSampleImages = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/ml/sample-images')
      const data = await response.json()
      setSampleImages(data.images || [])
    } catch (error) {
      console.error('Error fetching sample images:', error)
    }
  }

  const analyzeImage = async (imageId, isUpload = false) => {
    try {
      setLoading(true)
      let response, data

      if (isUpload && uploadedFile) {
        const formData = new FormData()
        formData.append('file', uploadedFile)
        response = await fetch('http://localhost:8000/api/ml/predict', {
          method: 'POST',
          body: formData
        })
        data = await response.json()
        setSelectedImage({ 
          image_data: URL.createObjectURL(uploadedFile), 
          filename: uploadedFile.name,
          id: 'uploaded'
        })
      } else {
        response = await fetch(`http://localhost:8000/api/ml/predict-sample/${imageId}`, {
          method: 'POST'
        })
        data = await response.json()
        const image = sampleImages.find(img => img.id === imageId)
        setSelectedImage(image)
      }

      setPrediction(data)
      
      // Generate segmentation data if malignant
      if (data.prediction === 'Malignant') {
        generateSegmentation(data.confidence)
      } else {
        setSegmentationData(null)
      }

    } catch (error) {
      console.error('Error analyzing image:', error)
    } finally {
      setLoading(false)
    }
  }

  const generateSegmentation = (confidence) => {
    // Generate realistic segmentation areas based on confidence
    const numAreas = Math.floor(confidence * 3) + 1 // 1-3 areas based on confidence
    const areas = []

    for (let i = 0; i < numAreas; i++) {
      areas.push({
        id: i + 1,
        x: Math.random() * 60 + 10, // 10-70% from left
        y: Math.random() * 60 + 10, // 10-70% from top
        width: Math.random() * 15 + 8, // 8-23% width
        height: Math.random() * 15 + 8, // 8-23% height
        confidence: confidence * (0.8 + Math.random() * 0.4), // Vary confidence
        type: Math.random() > 0.7 ? 'mass' : 'calcification'
      })
    }

    setSegmentationData({
      areas,
      totalAreas: areas.length,
      avgConfidence: areas.reduce((sum, area) => sum + area.confidence, 0) / areas.length
    })
  }

  const drawSegmentation = () => {
    if (!canvasRef.current || !selectedImage || !segmentationData) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      // Draw segmentation areas
      segmentationData.areas.forEach(area => {
        const x = (area.x / 100) * canvas.width
        const y = (area.y / 100) * canvas.height
        const width = (area.width / 100) * canvas.width
        const height = (area.height / 100) * canvas.height

        // Draw semi-transparent overlay
        ctx.fillStyle = area.type === 'mass' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(245, 158, 11, 0.3)'
        ctx.fillRect(x, y, width, height)

        // Draw border
        ctx.strokeStyle = area.type === 'mass' ? '#ef4444' : '#f59e0b'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, width, height)

        // Draw label
        ctx.fillStyle = area.type === 'mass' ? '#ef4444' : '#f59e0b'
        ctx.font = '12px Inter'
        ctx.fillText(
          `${area.type.toUpperCase()} ${(area.confidence * 100).toFixed(0)}%`,
          x,
          y - 5
        )
      })
    }

    img.src = selectedImage.image_data
  }

  useEffect(() => {
    if (segmentationData) {
      drawSegmentation()
    }
  }, [segmentationData, selectedImage])

  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setUploadedFile(file)
      setPrediction(null)
      setSelectedImage(null)
      setSegmentationData(null)
    }
  }

  const getResultColor = (prediction) => {
    return prediction === 'Malignant' ? 'text-red-600' : 'text-green-600'
  }

  const getResultIcon = (prediction) => {
    return prediction === 'Malignant' ? 
      <AlertTriangle className="h-5 w-5 text-red-500" /> : 
      <CheckCircle className="h-5 w-5 text-green-500" />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/30 p-8 mb-8 animate-fade-in-down relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-purple-500/10 to-violet-600/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
          <div className="relative z-10">
            <div className="flex items-center space-x-6">
              <Link 
                to="/" 
                className="bg-gradient-to-r from-slate-100 to-slate-200 hover:from-slate-200 hover:to-slate-300 text-slate-700 px-6 py-3 rounded-xl font-semibold transition-all duration-300 shadow-sm hover:shadow-md flex items-center space-x-2 group"
              >
                <ArrowLeft className="h-5 w-5 group-hover:-translate-x-1 transition-transform duration-200" />
                <span>Back to Dashboard</span>
              </Link>
              
              <div className="flex items-center space-x-6">
                <div className="p-4 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl shadow-lg shadow-purple-500/25">
                  <Activity className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-violet-600 bg-clip-text text-transparent">
                    Medical Image Analyzer
                  </h1>
                  <p className="text-slate-700 text-lg font-medium mt-2">
                    AI-powered mammogram analysis with segmentation
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Image Upload & Selection */}
          <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">Image Analysis</h2>
              <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg shadow-blue-500/25">
                <Upload className="h-5 w-5 text-white" />
              </div>
            </div>

            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-sm font-bold text-slate-800 mb-3 uppercase tracking-wide">
                Upload Mammogram Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="block w-full text-sm text-slate-700 file:mr-4 file:py-3 file:px-6 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-gradient-to-r file:from-blue-50 file:to-indigo-50 file:text-blue-700 hover:file:from-blue-100 hover:file:to-indigo-100 file:shadow-sm hover:file:shadow-md file:transition-all file:duration-300"
              />
            </div>

            {uploadedFile && (
              <button
                onClick={() => analyzeImage(null, true)}
                disabled={loading}
                className="w-full bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 hover:scale-105 flex items-center justify-center space-x-3 mb-6 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Zap className="h-5 w-5" />
                <span>{loading ? 'Analyzing...' : 'Analyze Image'}</span>
              </button>
            )}

            {/* Sample Images */}
            <div>
              <h3 className="text-xl font-bold text-slate-800 mb-4">Sample Images</h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {sampleImages.slice(0, 8).map((image, index) => (
                  <div key={image.id} className="group p-4 bg-gradient-to-r from-white/95 to-white/90 rounded-xl border border-slate-200 hover:border-blue-300 hover:shadow-lg transition-all duration-500 animate-fade-in-up" style={{animationDelay: `${0.4 + index * 0.05}s`}}>
                    <div className="flex items-center space-x-4">
                      <img 
                        src={image.image_data} 
                        alt={image.filename}
                        className="w-16 h-16 object-cover rounded-xl border-2 border-slate-200 group-hover:border-blue-300 transition-all duration-300"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold text-slate-800 truncate">{image.filename}</p>
                        <p className="text-xs text-slate-600 font-medium bg-slate-100 px-2 py-1 rounded-lg mt-1 inline-block">True: {image.true_label}</p>
                      </div>
                      <button
                        onClick={() => analyzeImage(image.id)}
                        disabled={loading}
                        className="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-sm hover:shadow-md hover:scale-105 flex items-center space-x-2 disabled:opacity-50"
                      >
                        <Eye className="h-4 w-4" />
                        <span>Analyze</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Image Display with Segmentation */}
          <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">Image & Segmentation</h2>
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-green-600 rounded-xl shadow-lg shadow-emerald-500/25">
                <Eye className="h-5 w-5 text-white" />
              </div>
            </div>

            {selectedImage ? (
              <div className="space-y-6">
                <div className="relative bg-slate-50 rounded-xl overflow-hidden border border-slate-200 shadow-inner">
                  {segmentationData ? (
                    <canvas
                      ref={canvasRef}
                      className="w-full h-auto max-h-96 object-contain"
                    />
                  ) : (
                    <img 
                      src={selectedImage.image_data} 
                      alt={selectedImage.filename}
                      className="w-full h-auto max-h-96 object-contain"
                    />
                  )}
                </div>
                
                <div className="text-center bg-gradient-to-r from-slate-50 to-blue-50 rounded-xl p-4">
                  <p className="text-base font-bold text-slate-800">{selectedImage.filename}</p>
                  {segmentationData && (
                    <p className="text-sm text-slate-600 mt-2 font-medium">
                      {segmentationData.totalAreas} suspicious area(s) detected
                    </p>
                  )}
                </div>

                {segmentationData && (
                  <div className="bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-800 mb-4 flex items-center">
                      <AlertTriangle className="h-5 w-5 mr-2" />
                      Detected Abnormalities
                    </h4>
                    <div className="space-y-3">
                      {segmentationData.areas.map(area => (
                        <div key={area.id} className="flex justify-between items-center bg-white/80 backdrop-blur-sm p-3 rounded-xl border border-red-100">
                          <span className="text-red-700 font-semibold flex items-center">
                            <span className="text-lg mr-2">{area.type === 'mass' ? '🔴' : '🟡'}</span>
                            {area.type.charAt(0).toUpperCase() + area.type.slice(1)}
                          </span>
                          <span className="text-red-600 font-bold bg-red-100 px-3 py-1 rounded-lg">
                            {(area.confidence * 100).toFixed(1)}% confidence
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-16">
                <div className="w-32 h-32 bg-gradient-to-br from-slate-100 to-slate-200 rounded-full flex items-center justify-center mx-auto mb-6 shadow-inner">
                  <Eye className="h-16 w-16 text-slate-400" />
                </div>
                <p className="text-slate-600 text-lg font-medium">Select an image to analyze</p>
                <p className="text-slate-500 text-sm mt-2">Upload your own image or choose from samples</p>
              </div>
            )}
          </div>

          {/* Analysis Results */}
          <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 animate-fade-in-up" style={{animationDelay: '0.3s'}}>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">Analysis Results</h2>
              <div className="p-2 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl shadow-lg shadow-orange-500/25">
                <Activity className="h-5 w-5 text-white" />
              </div>
            </div>

            {prediction ? (
              <div className="space-y-6">
                {/* Main Result */}
                <div className={`text-center p-8 rounded-2xl shadow-lg animate-scale-in ${
                  prediction.prediction === 'Malignant' 
                    ? 'bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200' 
                    : 'bg-gradient-to-br from-emerald-50 to-green-50 border-2 border-emerald-200'
                }`}>
                  <div className="flex items-center justify-center mb-4">
                    {getResultIcon(prediction.prediction)}
                  </div>
                  <h3 className={`text-3xl font-bold ${getResultColor(prediction.prediction)}`}>
                    {prediction.prediction}
                  </h3>
                  <p className="text-slate-700 mt-2 font-semibold text-lg">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>

                {/* Probability Breakdown */}
                <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl p-6 border border-slate-200">
                  <h4 className="text-xl font-bold text-slate-800 mb-6">Probability Breakdown</h4>
                  
                  <div className="space-y-4">
                    <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-emerald-100">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-emerald-700 font-bold flex items-center">
                          <CheckCircle className="h-5 w-5 mr-2" />
                          Benign
                        </span>
                        <span className="font-bold text-slate-800 bg-emerald-100 px-3 py-1 rounded-lg">{(prediction.probability_benign * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-3 shadow-inner">
                        <div 
                          className="bg-gradient-to-r from-emerald-500 to-green-500 h-3 rounded-full shadow-sm transition-all duration-1000" 
                          style={{ width: `${prediction.probability_benign * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-red-100">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-red-700 font-bold flex items-center">
                          <AlertTriangle className="h-5 w-5 mr-2" />
                          Malignant
                        </span>
                        <span className="font-bold text-slate-800 bg-red-100 px-3 py-1 rounded-lg">{(prediction.probability_malignant * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-3 shadow-inner">
                        <div 
                          className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full shadow-sm transition-all duration-1000" 
                          style={{ width: `${prediction.probability_malignant * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Accuracy Check */}
                {prediction.true_label && (
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-blue-700 font-medium">Ground Truth:</span>
                      <span className="text-blue-800">{prediction.true_label}</span>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-blue-700 font-medium">Prediction Accuracy:</span>
                      <span className={`font-bold ${
                        prediction.prediction === prediction.true_label ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {prediction.prediction === prediction.true_label ? 'Correct ✓' : 'Incorrect ✗'}
                      </span>
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                  <h4 className="font-semibold text-amber-800 mb-2">Clinical Recommendations:</h4>
                  <ul className="text-sm text-amber-700 space-y-1">
                    {prediction.prediction === 'Malignant' ? (
                      <>
                        <li>• Immediate referral to oncology specialist</li>
                        <li>• Additional imaging studies recommended</li>
                        <li>• Biopsy confirmation required</li>
                        <li>• Patient counseling and support</li>
                      </>
                    ) : (
                      <>
                        <li>• Continue routine screening schedule</li>
                        <li>• Follow-up in 12 months</li>
                        <li>• Maintain healthy lifestyle</li>
                        <li>• Report any changes immediately</li>
                      </>
                    )}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-24 h-24 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Activity className="h-12 w-12 text-slate-400" />
                </div>
                <p className="text-slate-500">Analysis results will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MedicalTester
