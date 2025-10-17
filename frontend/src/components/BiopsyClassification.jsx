import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, FileText, AlertCircle, CheckCircle, Microscope, Calculator, TrendingUp } from 'lucide-react'
import { patientService } from '../services/api'

const BiopsyClassification = () => {
  const { patientId } = useParams()
  const [patient, setPatient] = useState(null)
  const [existingBiopsy, setExistingBiopsy] = useState(null)
  const [formData, setFormData] = useState({
    tumor_size: '',
    lymph_nodes_affected: '',
    grade: '',
    hormone_receptor_status: '',
    her2_status: ''
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const patientData = await patientService.getPatient(patientId)
        setPatient(patientData)
        
        // Try to fetch existing biopsy data
        try {
          const biopsyData = await patientService.getPatientBiopsy(patientId)
          setExistingBiopsy(biopsyData)
        } catch (biopsyError) {
          // No existing biopsy data, which is fine
          console.log('No existing biopsy data found')
        }
      } catch (err) {
        setError('Failed to fetch patient data')
        console.error('Error fetching data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [patientId])

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)

    try {
      const biopsyData = {
        ...formData,
        tumor_size: parseFloat(formData.tumor_size),
        lymph_nodes_affected: parseInt(formData.lymph_nodes_affected),
        grade: parseInt(formData.grade)
      }

      const result = await patientService.createBiopsyClassification(patientId, biopsyData)
      setResult(result)
    } catch (err) {
      setError('Failed to process biopsy classification')
      console.error('Error submitting biopsy data:', err)
    } finally {
      setSubmitting(false)
    }
  }

  const getStageColor = (stage) => {
    switch (stage) {
      case 'Stage 1': return 'text-green-600 bg-green-100'
      case 'Stage 2': return 'text-yellow-600 bg-yellow-100'
      case 'Stage 3': return 'text-orange-600 bg-orange-100'
      case 'Stage 4': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="medical-spinner mx-auto"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-500 to-violet-600 opacity-20 animate-ping"></div>
            </div>
          </div>
          <p className="mt-8 text-xl font-bold bg-gradient-to-r from-purple-600 to-violet-600 bg-clip-text text-transparent">Loading biopsy data...</p>
          <p className="text-sm text-slate-600 mt-3 font-medium">Analyzing pathology results</p>
        </div>
      </div>
    )
  }

  if (error && !patient) {
    return (
      <div className="medical-alert-danger">
        <div className="flex items-center">
          <AlertCircle className="h-5 w-5 mr-2" />
          <p className="font-semibold">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <div className="flex items-center space-x-6 mb-8 animate-fade-in-down">
          <Link 
            to={`/patient/${patientId}`}
            className="bg-gradient-to-r from-slate-100 to-slate-200 hover:from-slate-200 hover:to-slate-300 text-slate-700 px-6 py-3 rounded-xl font-semibold transition-all duration-300 shadow-sm hover:shadow-md flex items-center space-x-2 group"
          >
            <ArrowLeft className="h-5 w-5 group-hover:-translate-x-1 transition-transform duration-200" />
            <span>Back to Profile</span>
          </Link>
          <div className="flex items-center space-x-4">
            <div className="p-4 bg-gradient-to-br from-purple-500 to-violet-600 rounded-2xl shadow-lg shadow-purple-500/25">
              <Microscope className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-violet-600 bg-clip-text text-transparent">
                Biopsy Classification
              </h1>
              <p className="text-lg text-slate-700 mt-1 font-medium">
                Patient: {patient?.name} • Advanced Cancer Staging
              </p>
            </div>
          </div>
        </div>

        <div className="medical-grid-2">
          {/* Existing Biopsy Results */}
          {existingBiopsy && (
            <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-8 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
              <div className="medical-card-header">
                <h2 className="text-xl font-bold text-gray-900 flex items-center">
                  <CheckCircle className="h-6 w-6 text-success-600 mr-3" />
                  Current Pathology Results
                </h2>
                <div className="text-sm text-gray-500">
                  Last updated: {new Date(existingBiopsy.created_at).toLocaleDateString()}
                </div>
              </div>
              
              <div className="space-y-6">
                <div className={`p-6 rounded-2xl border-2 ${getStageColor(existingBiopsy.stage)} shadow-lg`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <TrendingUp className="h-6 w-6" />
                      <span className="text-2xl font-bold">{existingBiopsy.stage}</span>
                    </div>
                    <div className="medical-badge-stage-2">
                      Confirmed
                    </div>
                  </div>
                  <p className="text-sm leading-relaxed opacity-90">
                    {existingBiopsy.stage_explanation}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white/50 p-4 rounded-xl">
                    <span className="text-gray-600 font-semibold block mb-1">Tumor Size</span>
                    <span className="text-xl font-bold text-gray-900">{existingBiopsy.tumor_size} cm</span>
                  </div>
                  <div className="bg-white/50 p-4 rounded-xl">
                    <span className="text-gray-600 font-semibold block mb-1">Lymph Nodes</span>
                    <span className="text-xl font-bold text-gray-900">{existingBiopsy.lymph_nodes_affected}</span>
                  </div>
                  <div className="bg-white/50 p-4 rounded-xl">
                    <span className="text-gray-600 font-semibold block mb-1">Grade</span>
                    <span className="text-xl font-bold text-gray-900">{existingBiopsy.grade}</span>
                  </div>
                  <div className="bg-white/50 p-4 rounded-xl">
                    <span className="text-gray-600 font-semibold block mb-1">Hormone Status</span>
                    <span className="text-lg font-bold text-gray-900">{existingBiopsy.hormone_receptor_status}</span>
                  </div>
                  <div className="bg-white/50 p-4 rounded-xl col-span-2">
                    <span className="text-gray-600 font-semibold block mb-1">HER2 Status</span>
                    <span className="text-lg font-bold text-gray-900">{existingBiopsy.her2_status}</span>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* New Biopsy Classification Form */}
          <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-8 animate-fade-in-up" style={{animationDelay: existingBiopsy ? '0.2s' : '0.1s'}}>
            <div className="medical-card-header">
              <h2 className="text-xl font-bold text-gray-900 flex items-center">
                <Calculator className="h-6 w-6 text-medical-600 mr-3" />
                {existingBiopsy ? 'Update Pathology Data' : 'New Biopsy Analysis'}
              </h2>
            </div>

            {error && (
              <div className="medical-alert-danger mb-6">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 mr-2" />
                  <p className="font-semibold">{error}</p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="medical-grid-2">
                <div>
                  <label className="medical-label" htmlFor="tumor_size">
                    Tumor Size (cm)
                  </label>
                  <input
                    type="number"
                    id="tumor_size"
                    name="tumor_size"
                    step="0.1"
                    min="0"
                    value={formData.tumor_size}
                    onChange={handleInputChange}
                    required
                    placeholder="e.g., 2.1"
                    className="medical-input"
                  />
                </div>

                <div>
                  <label className="medical-label" htmlFor="lymph_nodes_affected">
                    Lymph Nodes Affected
                  </label>
                  <input
                    type="number"
                    id="lymph_nodes_affected"
                    name="lymph_nodes_affected"
                    min="0"
                    value={formData.lymph_nodes_affected}
                    onChange={handleInputChange}
                    required
                    placeholder="e.g., 2"
                    className="medical-input"
                  />
                </div>
              </div>

              <div>
                <label className="medical-label" htmlFor="grade">
                  Tumor Grade
                </label>
                <select
                  id="grade"
                  name="grade"
                  value={formData.grade}
                  onChange={handleInputChange}
                  required
                  className="medical-select"
                >
                  <option value="">Select Tumor Grade</option>
                  <option value="1">Grade 1 (Low - Well differentiated)</option>
                  <option value="2">Grade 2 (Intermediate - Moderately differentiated)</option>
                  <option value="3">Grade 3 (High - Poorly differentiated)</option>
                </select>
              </div>

              <div>
                <label className="medical-label" htmlFor="hormone_receptor_status">
                  Hormone Receptor Status
                </label>
                <select
                  id="hormone_receptor_status"
                  name="hormone_receptor_status"
                  value={formData.hormone_receptor_status}
                  onChange={handleInputChange}
                  required
                  className="medical-select"
                >
                  <option value="">Select Hormone Status</option>
                  <option value="ER+/PR+">ER+/PR+ (Estrogen & Progesterone Positive)</option>
                  <option value="ER+/PR-">ER+/PR- (Estrogen Positive Only)</option>
                  <option value="ER-/PR+">ER-/PR+ (Progesterone Positive Only)</option>
                  <option value="ER-/PR-">ER-/PR- (Hormone Negative)</option>
                </select>
              </div>

              <div>
                <label className="medical-label" htmlFor="her2_status">
                  HER2 Status
                </label>
                <select
                  id="her2_status"
                  name="her2_status"
                  value={formData.her2_status}
                  onChange={handleInputChange}
                  required
                  className="medical-select"
                >
                  <option value="">Select HER2 Status</option>
                  <option value="Positive">Positive (Overexpressed)</option>
                  <option value="Negative">Negative (Normal levels)</option>
                  <option value="Equivocal">Equivocal (Requires further testing)</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="medical-btn-primary w-full text-lg py-4"
              >
                {submitting ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-2"></div>
                    Processing Analysis...
                  </>
                ) : (
                  <>
                    <Calculator className="h-5 w-5" />
                    Calculate Cancer Stage
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Classification Result */}
        {result && (
          <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-8 mt-8 animate-scale-in">
            <div className="medical-card-header">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center">
                <CheckCircle className="h-6 w-6 text-success-600 mr-3" />
                Analysis Complete
              </h2>
              <div className="text-sm text-gray-500">
                Generated: {new Date().toLocaleString()}
              </div>
            </div>

            <div className="space-y-6">
              <div className={`p-8 rounded-2xl border-2 ${getStageColor(result.stage)} shadow-xl`}>
                <div className="text-center">
                  <div className="flex items-center justify-center mb-4">
                    <TrendingUp className="h-8 w-8 mr-3" />
                    <h3 className="text-4xl font-bold">Result: {result.stage}</h3>
                  </div>
                  <p className="text-xl leading-relaxed opacity-90 max-w-2xl mx-auto">
                    {result.stage_explanation}
                  </p>
                </div>
              </div>

              <div className="medical-card bg-gradient-to-br from-gray-50 to-blue-50">
                <h4 className="text-lg font-bold text-gray-900 mb-4">Pathology Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-white p-4 rounded-xl shadow-sm">
                    <span className="text-gray-600 font-semibold block mb-1">Tumor Size</span>
                    <span className="text-2xl font-bold text-gray-900">{result.tumor_size} cm</span>
                  </div>
                  <div className="bg-white p-4 rounded-xl shadow-sm">
                    <span className="text-gray-600 font-semibold block mb-1">Lymph Nodes</span>
                    <span className="text-2xl font-bold text-gray-900">{result.lymph_nodes_affected}</span>
                  </div>
                  <div className="bg-white p-4 rounded-xl shadow-sm">
                    <span className="text-gray-600 font-semibold block mb-1">Grade</span>
                    <span className="text-2xl font-bold text-gray-900">{result.grade}</span>
                  </div>
                  <div className="bg-white p-4 rounded-xl shadow-sm">
                    <span className="text-gray-600 font-semibold block mb-1">Hormone Status</span>
                    <span className="text-lg font-bold text-gray-900">{result.hormone_receptor_status}</span>
                  </div>
                  <div className="bg-white p-4 rounded-xl shadow-sm col-span-2">
                    <span className="text-gray-600 font-semibold block mb-1">HER2 Status</span>
                    <span className="text-lg font-bold text-gray-900">{result.her2_status}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default BiopsyClassification
