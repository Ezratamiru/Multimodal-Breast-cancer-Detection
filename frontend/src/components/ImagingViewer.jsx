import React, { useState } from 'react'
import { ZoomIn, Download, Info, Eye, Maximize2, AlertTriangle } from 'lucide-react'

const ImagingViewer = ({ imagingType, imagingData }) => {
  const [selectedImage, setSelectedImage] = useState(null)

  // Real sample images from datasets
  const getPlaceholderImage = (type) => {
    const placeholders = {
      mammogram: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iIzMzMzMzMyIvPgogIDx0ZXh0IHg9IjIwMCIgeT0iMTUwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5NYW1tb2dyYW0gU2FtcGxlPC90ZXh0Pgo8L3N2Zz4K',
      ultrasound: 'http://localhost:8000/static/samples/ultrasound_sample.jpg',
      mri: 'http://localhost:8000/static/samples/mri_sample.png'
    }
    return placeholders[type] || placeholders.mammogram
  }

  if (!imagingData || imagingData.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl border border-white/20 max-w-md mx-auto p-8">
          <div className="text-slate-400 mb-6">
            <Info className="h-16 w-16 mx-auto" />
          </div>
          <h3 className="text-2xl font-bold bg-gradient-to-r from-slate-700 to-slate-600 bg-clip-text text-transparent mb-3">
            No {imagingType} results available
          </h3>
          <p className="text-slate-600 mb-4 leading-relaxed">
            Imaging studies are not available for this modality.
          </p>
          <div className="text-sm text-slate-500 bg-slate-50 px-3 py-2 rounded-lg">Please choose another tab or add a new study.</div>
        </div>
      </div>
    )
  }

  const currentImage = selectedImage || imagingData[0]

  return (
    <div className="space-y-8">
      {/* Image Selection */}
      {imagingData.length > 1 && (
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-lg border border-white/20 p-6">
          <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center">
            <Eye className="h-6 w-6 mr-3 text-blue-600" />
            Available Studies ({imagingData.length})
          </h3>
          <div className="flex space-x-4 overflow-x-auto pb-2">
            {imagingData.map((image, index) => (
              <button
                key={image.id}
                onClick={() => setSelectedImage(image)}
                className={`group flex-shrink-0 p-4 border-2 rounded-2xl transition-all duration-300 ${
                  currentImage.id === image.id
                    ? 'border-blue-500 bg-blue-50 shadow-xl scale-105'
                    : 'border-slate-200 hover:border-blue-300 hover:bg-white hover:shadow-lg hover:scale-102'
                }`}
              >
                <div className="w-28 h-24 bg-gradient-to-br from-slate-100 to-slate-200 rounded-xl flex items-center justify-center mb-3 group-hover:from-blue-50 group-hover:to-indigo-50 transition-all duration-300">
                  <span className="text-sm font-bold text-slate-700 group-hover:text-blue-700">
                    {imagingType.charAt(0).toUpperCase() + imagingType.slice(1)} {index + 1}
                  </span>
                </div>
                <div className="text-xs text-slate-600 text-center font-medium">
                  {new Date(image.created_at).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Image Viewer */}
        <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl border border-white/20 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent flex items-center">
              <Maximize2 className="h-6 w-6 mr-3 text-blue-600" />
              {imagingType.charAt(0).toUpperCase() + imagingType.slice(1)} Analysis
            </h3>
            <div className="flex space-x-3">
              <button className="bg-gradient-to-r from-slate-100 to-slate-200 hover:from-slate-200 hover:to-slate-300 text-slate-700 px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-sm hover:shadow-md flex items-center space-x-2">
                <ZoomIn className="h-4 w-4" />
                <span>Zoom</span>
              </button>
              <button className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-4 py-2 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 flex items-center space-x-2">
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
            </div>
          </div>

          <div className="relative bg-slate-50 rounded-xl overflow-hidden border border-slate-200">
            <img
              src={currentImage.image_data || currentImage.image_url || getPlaceholderImage(imagingType)}
              alt={`${imagingType} scan`}
              className="w-full h-auto"
            />
            
            {/* Enhanced Segmentation Overlay for Mammogram */}
            {imagingType === 'mammogram' && currentImage.segmentation_data && (
              <>
                <div
                  className="absolute border-4 border-red-500 bg-red-500/20 rounded-lg animate-pulse"
                  style={{
                    left: `${currentImage.segmentation_data.coordinates.x}px`,
                    top: `${currentImage.segmentation_data.coordinates.y}px`,
                    width: `${currentImage.segmentation_data.coordinates.width}px`,
                    height: `${currentImage.segmentation_data.coordinates.height}px`,
                    boxShadow: '0 0 20px rgba(239, 68, 68, 0.6)'
                  }}
                />
                <div
                  className="absolute bg-white/95 backdrop-blur-sm border-2 border-red-500 rounded-xl p-3 shadow-xl"
                  style={{
                    left: `${currentImage.segmentation_data.coordinates.x + currentImage.segmentation_data.coordinates.width + 10}px`,
                    top: `${currentImage.segmentation_data.coordinates.y}px`,
                  }}
                >
                  <div className="space-y-2">
                    <div className="flex items-center font-bold text-red-700">
                      <AlertTriangle className="h-4 w-4 mr-2" />
                      Detected Mass
                    </div>
                    <div className="text-sm space-y-1 text-slate-800">
                      <div><span className="font-semibold text-slate-700">Size:</span> {currentImage.segmentation_data.mass_size}</div>
                      <div><span className="font-semibold text-slate-700">Shape:</span> {currentImage.segmentation_data.shape}</div>
                      <div><span className="font-semibold text-slate-700">Density:</span> {currentImage.segmentation_data.density}</div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Enhanced Findings and Analysis */}
        <div className="space-y-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 backdrop-blur-xl rounded-2xl shadow-lg border border-blue-200/50 p-6">
            <h4 className="text-xl font-bold text-slate-800 mb-4 flex items-center">
              <Info className="h-6 w-6 mr-3 text-blue-600" />
              Radiologist Findings
            </h4>
            <p className="text-slate-700 leading-relaxed text-base">{currentImage.findings}</p>
          </div>

          {currentImage.segmentation_data && (
            <div className="bg-gradient-to-br from-red-50 to-orange-50 backdrop-blur-xl rounded-2xl shadow-lg border border-red-200/50 p-6">
              <h4 className="text-xl font-bold text-red-800 mb-6 flex items-center">
                <AlertTriangle className="h-6 w-6 mr-3 text-red-600" />
                Mass Characteristics
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-red-100 shadow-sm">
                  <span className="text-red-700 font-bold block text-sm uppercase tracking-wide mb-2">Size</span>
                  <span className="text-xl font-bold text-slate-800">{currentImage.segmentation_data.mass_size}</span>
                </div>
                <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-red-100 shadow-sm">
                  <span className="text-red-700 font-bold block text-sm uppercase tracking-wide mb-2">Shape</span>
                  <span className="text-xl font-bold text-slate-800">{currentImage.segmentation_data.shape}</span>
                </div>
                <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-red-100 shadow-sm">
                  <span className="text-red-700 font-bold block text-sm uppercase tracking-wide mb-2">Density</span>
                  <span className="text-xl font-bold text-slate-800">{currentImage.segmentation_data.density}</span>
                </div>
                <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl border border-red-100 shadow-sm">
                  <span className="text-red-700 font-bold block text-sm uppercase tracking-wide mb-2">Location</span>
                  <span className="text-xl font-bold text-slate-800">Upper outer quadrant</span>
                </div>
              </div>
            </div>
          )}

          <div className={`backdrop-blur-xl rounded-2xl shadow-lg p-6 ${
            imagingType === 'mammogram' && currentImage.segmentation_data 
              ? 'bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200/50' 
              : 'bg-gradient-to-br from-emerald-50 to-green-50 border border-emerald-200/50'
          }`}>
            <h4 className={`text-xl font-bold mb-4 flex items-center ${
              imagingType === 'mammogram' && currentImage.segmentation_data 
                ? 'text-amber-800' 
                : 'text-emerald-800'
            }`}>
              <Info className={`h-6 w-6 mr-3 ${
                imagingType === 'mammogram' && currentImage.segmentation_data 
                  ? 'text-amber-600' 
                  : 'text-emerald-600'
              }`} />
              Clinical Recommendation
            </h4>
            <p className="text-base leading-relaxed text-slate-700 font-medium">
              {imagingType === 'mammogram' && currentImage.segmentation_data
                ? 'Immediate biopsy recommended due to suspicious mass characteristics. Schedule follow-up within 48 hours.'
                : 'Continue routine screening as per established guidelines. Next screening in 12 months.'
              }
            </p>
          </div>

          <div className="bg-gradient-to-br from-slate-50 to-blue-50 backdrop-blur-xl rounded-2xl shadow-lg border border-slate-200/50 p-6">
            <h4 className="text-xl font-bold text-slate-800 mb-6">Study Information</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/60 backdrop-blur-sm p-4 rounded-xl border border-slate-100">
                <span className="text-slate-600 font-semibold text-sm uppercase tracking-wide block mb-1">Scan Date</span>
                <span className="text-slate-800 font-bold text-base">{new Date(currentImage.created_at).toLocaleDateString()}</span>
              </div>
              <div className="bg-white/60 backdrop-blur-sm p-4 rounded-xl border border-slate-100">
                <span className="text-slate-600 font-semibold text-sm uppercase tracking-wide block mb-1">Study ID</span>
                <span className="text-slate-800 font-bold text-base">{currentImage.id}</span>
              </div>
              <div className="bg-white/60 backdrop-blur-sm p-4 rounded-xl border border-slate-100">
                <span className="text-slate-600 font-semibold text-sm uppercase tracking-wide block mb-1">Modality</span>
                <span className="text-slate-800 font-bold text-base capitalize">{imagingType}</span>
              </div>
              <div className="bg-white/60 backdrop-blur-sm p-4 rounded-xl border border-slate-100">
                <span className="text-slate-600 font-semibold text-sm uppercase tracking-wide block mb-1">Status</span>
                <span className="text-emerald-700 font-bold text-base">Reviewed</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImagingViewer
