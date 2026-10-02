import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, FileText, Image as ImageIcon, User, Calendar, Activity, Stethoscope } from 'lucide-react'
import { patientService } from '../services/api'
import ImagingViewer from './ImagingViewer'

const PatientProfile = () => {
  const { patientId } = useParams()
  const [patient, setPatient] = useState(null)
  const [imagingResults, setImagingResults] = useState([])
  const [activeTab, setActiveTab] = useState('mammogram')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        const [patientData, imagingData] = await Promise.all([
          patientService.getPatient(patientId),
          patientService.getPatientImaging(patientId)
        ])
        
        setPatient(patientData)
        setImagingResults(imagingData)
      } catch (err) {
        setError('Failed to fetch patient data')
        console.error('Error fetching patient data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchPatientData()
  }, [patientId])

  const getImagingByType = (type) => {
    return imagingResults.filter(result => result.imaging_type === type)
  }

  const tabs = [
    { id: 'mammogram', label: 'Mammogram', icon: ImageIcon, color: 'text-medical-600' },
    { id: 'ultrasound', label: 'Ultrasound', icon: Activity, color: 'text-purple-600' },
    { id: 'mri', label: 'MRI', icon: Stethoscope, color: 'text-indigo-600' }
  ]

  if (loading) {
    return (
      <div className="medical-loading">
        <div className="text-center">
          <div className="medical-spinner"></div>
          <p className="mt-6 text-lg font-medium text-gray-600">Loading patient profile...</p>
          <p className="text-sm text-gray-500 mt-2">Retrieving medical records</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="medical-alert-danger">
        <p className="font-semibold">{error}</p>
      </div>
    )
  }

  if (!patient) {
    return (
      <div className="medical-alert-warning">
        <p className="font-semibold">Patient not found</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-500 via-pink-500 to-orange-400">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <div className="bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl border-2 border-yellow-300/50 p-8">
          <div className="flex items-center space-x-6">
            <Link 
              to="/dashboard" 
              className="bg-gradient-to-r from-purple-100 to-pink-100 hover:from-purple-200 hover:to-pink-200 text-purple-700 px-6 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105 flex items-center space-x-2 group"
            >
              <ArrowLeft className="h-5 w-5 group-hover:-translate-x-1 transition-transform duration-200" />
              <span>Back to Dashboard</span>
            </Link>
            
            <div className="flex items-center space-x-6">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500 text-white flex items-center justify-center font-bold text-2xl shadow-xl shadow-pink-500/40">
                {patient.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-orange-600 bg-clip-text text-transparent">
                  {patient.name}
                </h1>
                <p className="text-lg text-slate-700 mt-2 font-semibold">
                  Patient ID: {patient.id} • Age: {patient.age} • {patient.gender}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Patient Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-gradient-to-br from-white via-purple-50 to-pink-50 backdrop-blur-xl rounded-2xl shadow-2xl border-2 border-purple-200/60 p-8">
            <div className="flex items-center mb-6">
              <div className="p-3 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500 rounded-xl shadow-lg shadow-pink-500/25 mr-4">
                <User className="h-6 w-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                👤 Patient Information
              </h2>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100 shadow-sm">
                <span className="text-slate-700 font-semibold">Patient ID</span>
                <span className="font-bold text-slate-800 bg-white px-3 py-1 rounded-lg">{patient.id}</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-emerald-50 to-green-50 rounded-xl border border-emerald-100 shadow-sm">
                <span className="text-slate-700 font-semibold">Age</span>
                <span className="font-bold text-slate-800 bg-white px-3 py-1 rounded-lg">{patient.age} years</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-50 to-violet-50 rounded-xl border border-purple-100 shadow-sm">
                <span className="text-slate-700 font-semibold">Gender</span>
                <span className="font-bold text-slate-800 bg-white px-3 py-1 rounded-lg">{patient.gender}</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-100 shadow-sm">
                <span className="text-slate-700 font-semibold">Registration Date</span>
                <span className="font-bold text-slate-800 bg-white px-3 py-1 rounded-lg">
                  {new Date(patient.created_at).toLocaleDateString()}
                </span>
              </div>
              {patient.latest_diagnosis && (
                <div className="p-4 bg-gradient-to-r from-red-50 to-orange-50 rounded-xl border border-red-200 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-red-700 font-bold">Latest Diagnosis</span>
                    <span className="bg-red-100 text-red-800 px-4 py-2 rounded-xl font-bold border border-red-200">
                      {patient.latest_diagnosis}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="bg-gradient-to-br from-white via-blue-50 to-indigo-50 backdrop-blur-xl rounded-2xl shadow-2xl border-2 border-blue-200/60 p-8">
            <div className="flex items-center mb-6">
              <div className="p-3 bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 rounded-xl shadow-lg shadow-blue-500/25 mr-4">
                <Calendar className="h-6 w-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                ⚡ Quick Actions
              </h2>
            </div>
            
            <div className="space-y-4">
              <Link
                to={`/patient/${patientId}/biopsy`}
                className="group w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105 flex items-center justify-center space-x-3"
              >
                <FileText className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Biopsy Classification</span>
              </Link>
              
              <button className="group w-full bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:scale-105 flex items-center justify-center space-x-3">
                <Calendar className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Schedule Follow-up</span>
              </button>
              
              <button className="group w-full bg-gradient-to-r from-purple-500 to-violet-500 hover:from-purple-600 hover:to-violet-600 text-white px-6 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 hover:scale-105 flex items-center justify-center space-x-3">
                <Activity className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span>View Medical History</span>
              </button>
            </div>
          </div>
        </div>

        {/* Breast Cancer Detection Section */}
        <div className="bg-gradient-to-br from-white via-green-50 to-emerald-50 backdrop-blur-xl rounded-2xl shadow-2xl border-2 border-green-200/60 p-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-orange-600 bg-clip-text text-transparent">
                🔬 Multi-Modal Breast Cancer Detection
              </h2>
              <p className="text-slate-700 mt-2 text-lg font-medium">
                Comprehensive imaging analysis and diagnostic results
              </p>
            </div>
            <div className="text-sm font-semibold text-slate-700 bg-gradient-to-r from-blue-50 to-indigo-50 px-4 py-2 rounded-xl border border-blue-200">
              {imagingResults.length} imaging studies available
            </div>
          </div>

          {/* Enhanced Tabs */}
          <div className="flex space-x-2 mb-8 bg-slate-50 p-2 rounded-2xl">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`flex items-center space-x-3 px-6 py-4 rounded-xl font-semibold transition-all duration-300 ${
                  activeTab === tab.id 
                    ? 'bg-white shadow-lg text-blue-700 border-2 border-blue-200' 
                    : 'text-slate-600 hover:text-slate-800 hover:bg-white/50'
                }`}
                onClick={() => setActiveTab(tab.id)}
              >
                <tab.icon className={`h-5 w-5 ${activeTab === tab.id ? 'text-blue-600' : tab.color}`} />
                <span>{tab.label}</span>
                <span className={`px-3 py-1 text-xs rounded-full font-bold ${
                  activeTab === tab.id 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'bg-slate-200 text-slate-700'
                }`}>
                  {getImagingByType(tab.id).length}
                </span>
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="bg-gradient-to-br from-slate-50/50 to-blue-50/50 rounded-2xl p-6 backdrop-blur-sm">
            <ImagingViewer 
              imagingType={activeTab}
              imagingData={getImagingByType(activeTab)}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default PatientProfile
