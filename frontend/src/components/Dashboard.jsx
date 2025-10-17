import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Users, Calendar, Activity, ChevronRight, TrendingUp, AlertCircle, CheckCircle2, Brain } from 'lucide-react'
import { patientService } from '../services/api'

const Dashboard = () => {
  const [patients, setPatients] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const [adding, setAdding] = useState(false)
  const [addForm, setAddForm] = useState({ name: '', age: '', gender: 'Female', latest_diagnosis: '' })

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const data = await patientService.getPatients()
        setPatients(data)
      } catch (err) {
        setError('Failed to load patients')
      } finally {
        setLoading(false)
      }
    }

    fetchPatients()
  }, [])

  const openAddPatient = () => {
    setAddForm({ name: '', age: '', gender: 'Female', latest_diagnosis: '' })
    setShowAddModal(true)
  }

  const handleAddChange = (e) => {
    const { name, value } = e.target
    setAddForm(prev => ({ ...prev, [name]: value }))
  }

  const submitAddPatient = async (e) => {
    e?.preventDefault()
    if (!addForm.name || !addForm.age) return
    setAdding(true)
    try {
      const payload = {
        name: addForm.name,
        age: parseInt(addForm.age, 10),
        gender: addForm.gender,
        latest_diagnosis: addForm.latest_diagnosis || undefined,
      }
      const created = await patientService.createPatient(payload)
      setPatients(prev => [created, ...prev])
      setShowAddModal(false)
    } catch (err) {
      console.error('Add patient failed', err)
    } finally {
      setAdding(false)
    }
  }

  const getStatusBadgeClass = (diagnosis) => {
    if (!diagnosis) return 'medical-badge bg-gray-100 text-gray-800'
    if (diagnosis.includes('Stage 1')) return 'medical-badge-stage-1'
    if (diagnosis.includes('Stage 2')) return 'medical-badge-stage-2'
    if (diagnosis.includes('Stage 3')) return 'medical-badge-stage-3'
    if (diagnosis.includes('Stage 4')) return 'medical-badge-stage-4'
    if (diagnosis.includes('Benign')) return 'medical-badge-benign'
    return 'medical-badge bg-gray-100 text-gray-800'
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="medical-spinner mx-auto"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 opacity-20 animate-ping"></div>
            </div>
          </div>
          <p className="mt-8 text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Loading patient data...</p>
          <p className="text-sm text-slate-600 mt-3 font-medium">Securing medical information</p>
        </div>

        {/* Add Patient Modal */}
        {showAddModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl w-full max-w-lg p-8 border border-white/20">
              <h3 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-6">Add New Patient</h3>
              <form onSubmit={submitAddPatient} className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">Full Name</label>
                  <input name="name" value={addForm.name} onChange={handleAddChange} required className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white/80 backdrop-blur-sm" placeholder="e.g., Jane Doe" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">Age</label>
                    <input name="age" type="number" min="0" value={addForm.age} onChange={handleAddChange} required className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white/80 backdrop-blur-sm" placeholder="e.g., 48" />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">Gender</label>
                    <select name="gender" value={addForm.gender} onChange={handleAddChange} className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white/80 backdrop-blur-sm">
                      <option>Female</option>
                      <option>Male</option>
                      <option>Other</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">Latest Diagnosis (optional)</label>
                  <input name="latest_diagnosis" value={addForm.latest_diagnosis} onChange={handleAddChange} className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white/80 backdrop-blur-sm" placeholder="e.g., Benign Fibroadenoma" />
                </div>
                <div className="flex justify-end space-x-4 pt-4">
                  <button type="button" onClick={() => setShowAddModal(false)} className="px-6 py-3 rounded-xl border-2 border-slate-300 text-slate-700 hover:bg-slate-100 font-semibold transition-all duration-300">Cancel</button>
                  <button disabled={adding} type="submit" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 disabled:opacity-50">
                    {adding ? 'Saving...' : 'Save Patient'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    )
  }

  if (error) {
    return (
      <div className="medical-alert-danger">
        <div className="flex items-center">
          <AlertCircle className="h-5 w-5 mr-2" />
          <p className="font-semibold">{error}</p>
        </div>
      </div>
    )
  }

  const activeCases = patients.filter(p => p.latest_diagnosis && !p.latest_diagnosis.includes('Benign')).length
  const benignCases = patients.filter(p => p.latest_diagnosis && p.latest_diagnosis.includes('Benign')).length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Modern Header with Animation */}
        <div className="mb-8 animate-fade-in">
          <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/30 p-8 hover:shadow-3xl transition-all duration-500 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-blue-500/10 to-indigo-600/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
            <div className="relative z-10">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-5xl font-black bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-3 animate-gradient">
                    Patient Dashboard
                  </h1>
                  <p className="text-slate-700 text-lg font-medium">
                    Advanced AI-powered breast cancer detection and monitoring system
                  </p>
                </div>
                <div className="flex items-center space-x-6">
                  <div className="flex items-center space-x-3 bg-gradient-to-r from-emerald-50 to-green-50 px-5 py-3 rounded-2xl border-2 border-emerald-200 shadow-lg shadow-emerald-500/10 hover:shadow-emerald-500/20 transition-all duration-300">
                    <div className="relative">
                      <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse shadow-lg shadow-emerald-500/50"></div>
                      <div className="absolute inset-0 w-3 h-3 bg-emerald-500 rounded-full animate-ping opacity-75"></div>
                    </div>
                    <span className="text-sm font-bold text-emerald-700">System Online</span>
                  </div>
                  <div className="text-sm text-slate-600 bg-white/70 px-4 py-3 rounded-xl font-semibold border border-slate-200 shadow-sm">
                    {new Date().toLocaleDateString('en-US', { 
                      weekday: 'long', 
                      year: 'numeric', 
                      month: 'long', 
                      day: 'numeric' 
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Modern Stats Grid with Staggered Animation */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Total Patients */}
          <div className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-600 mb-2 uppercase tracking-wide">Total Patients</p>
                <p className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-blue-700 bg-clip-text text-transparent">{patients.length}</p>
                <div className="flex items-center mt-3">
                  <TrendingUp className="h-4 w-4 text-emerald-600 mr-2" />
                  <span className="text-sm text-emerald-600 font-semibold">Active Registry</span>
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl shadow-lg shadow-blue-500/25 group-hover:shadow-blue-500/40 transition-all duration-300">
                <Users className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>

          {/* Active Cases */}
          <div className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-600 mb-2 uppercase tracking-wide">Active Cases</p>
                <p className="text-4xl font-bold bg-gradient-to-r from-orange-500 to-red-500 bg-clip-text text-transparent">{activeCases}</p>
                <div className="flex items-center mt-3">
                  <AlertCircle className="h-4 w-4 text-orange-600 mr-2" />
                  <span className="text-sm text-orange-600 font-semibold">Under Monitoring</span>
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-orange-500 to-red-500 rounded-2xl shadow-lg shadow-orange-500/25 group-hover:shadow-orange-500/40 transition-all duration-300">
                <Activity className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>

          {/* Benign Cases */}
          <div className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up" style={{animationDelay: '0.3s'}}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-600 mb-2 uppercase tracking-wide">Benign Cases</p>
                <p className="text-4xl font-bold bg-gradient-to-r from-emerald-500 to-green-500 bg-clip-text text-transparent">{benignCases}</p>
                <div className="flex items-center mt-3">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 mr-2" />
                  <span className="text-sm text-emerald-600 font-semibold">Stable Condition</span>
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-emerald-500 to-green-500 rounded-2xl shadow-lg shadow-emerald-500/25 group-hover:shadow-emerald-500/40 transition-all duration-300">
                <CheckCircle2 className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up" style={{animationDelay: '0.4s'}}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-600 mb-2 uppercase tracking-wide">System Status</p>
                <p className="text-4xl font-bold bg-gradient-to-r from-emerald-500 to-green-500 bg-clip-text text-transparent">100%</p>
                <div className="flex items-center mt-3">
                  <div className="w-3 h-3 bg-emerald-500 rounded-full mr-2 animate-pulse shadow-lg shadow-emerald-500/50"></div>
                  <span className="text-sm text-emerald-600 font-semibold">Fully Operational</span>
                </div>
              </div>
              <div className="p-4 bg-gradient-to-br from-emerald-500 to-green-500 rounded-2xl shadow-lg shadow-emerald-500/25 group-hover:shadow-emerald-500/40 transition-all duration-300">
                <Activity className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>
        </div>

        {/* Modern Quick Actions with Enhanced Animations */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-8 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up hover:border-blue-300" style={{animationDelay: '0.5s'}}>
            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-blue-500/25 group-hover:shadow-blue-500/40 group-hover:rotate-6 transition-all duration-500">
                <Users className="h-10 w-10 text-white" />
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-3">Add New Patient</h3>
              <p className="text-slate-600 text-sm mb-6 leading-relaxed">Register a new patient for comprehensive monitoring and analysis</p>
              <button onClick={openAddPatient} className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-4 rounded-xl font-semibold w-full transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105">
                Add Patient
              </button>
            </div>
          </div>

          <div className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-8 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up hover:border-emerald-300" style={{animationDelay: '0.6s'}}>
            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-emerald-500 to-green-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-emerald-500/25 group-hover:shadow-emerald-500/40 group-hover:rotate-6 transition-all duration-500">
                <Activity className="h-10 w-10 text-white" />
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-3">System Analytics</h3>
              <p className="text-slate-600 text-sm mb-6 leading-relaxed">View detailed system performance metrics and insights</p>
              <button className="bg-gradient-to-r from-slate-100 to-slate-200 hover:from-slate-200 hover:to-slate-300 text-slate-700 px-8 py-4 rounded-xl font-semibold w-full transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105">
                View Analytics
              </button>
            </div>
          </div>

          <Link to="/medical-tester" className="group bg-white/90 backdrop-blur-xl rounded-3xl shadow-xl border border-white/30 p-8 hover:shadow-2xl hover:scale-105 transition-all duration-500 animate-fade-in-up hover:border-purple-300 block" style={{animationDelay: '0.7s'}}>
            <div className="text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-violet-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-purple-500/25 group-hover:shadow-purple-500/40 group-hover:rotate-6 transition-all duration-500">
                <Brain className="h-10 w-10 text-white" />
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-3">AI Medical Tester</h3>
              <p className="text-slate-600 text-sm mb-6 leading-relaxed">Advanced AI-powered medical image analysis and detection</p>
              <div className="bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700 text-white px-8 py-4 rounded-xl font-semibold w-full transition-all duration-300 shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 hover:scale-105 text-center">
                Open Tester
              </div>
            </div>
          </Link>
        </div>

        {/* Modern Patient List with Enhanced Design */}
        <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/30 p-8 animate-fade-in-up" style={{animationDelay: '0.8s'}}>
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">Patient Registry</h2>
              <p className="text-slate-600 mt-2 text-lg">Comprehensive patient management and monitoring system</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm font-semibold text-slate-700 bg-gradient-to-r from-blue-50 to-indigo-50 px-4 py-2 rounded-xl border border-blue-200">
                {patients.length} patients registered
              </div>
              <div className="text-sm text-slate-500 bg-white/50 px-3 py-2 rounded-lg">
                Updated: {new Date().toLocaleDateString()}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            {patients.map((patient, index) => (
              <div key={patient.id} className="group flex items-center justify-between p-6 bg-gradient-to-r from-white/95 to-white/90 backdrop-blur-sm rounded-2xl border border-slate-200 hover:border-blue-300 hover:shadow-2xl hover:scale-[1.02] transition-all duration-500 animate-fade-in-up" style={{animationDelay: `${0.9 + index * 0.05}s`}}>
                <div className="flex items-center gap-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white flex items-center justify-center font-bold text-xl shadow-lg shadow-blue-500/25 group-hover:shadow-blue-500/40 transition-all duration-300">
                    {patient.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div>
                    <Link to={`/patient/${patient.id}`} className="text-xl font-bold text-slate-800 hover:text-blue-600 transition-colors group-hover:text-blue-600">
                      {patient.name}
                    </Link>
                    <div className="text-sm text-slate-700 mt-2 flex gap-6">
                      <span className="bg-slate-100 px-2 py-1 rounded-lg text-slate-800">ID: {patient.id}</span>
                      <span className="bg-slate-100 px-2 py-1 rounded-lg text-slate-800">Age: {patient.age}</span>
                      <span className="bg-slate-100 px-2 py-1 rounded-lg text-slate-800">{patient.gender}</span>
                      <span className="bg-slate-100 px-2 py-1 rounded-lg text-slate-800">Registered: {new Date(patient.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  {patient.latest_diagnosis && (
                    <span className={getStatusBadgeClass(patient.latest_diagnosis)}>
                      {patient.latest_diagnosis}
                    </span>
                  )}
                  <Link to={`/patient/${patient.id}`} className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105">
                    View Details
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Add Patient Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl w-full max-w-lg p-8 border border-white/20">
            <h3 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-6">Add New Patient</h3>
            <form onSubmit={submitAddPatient} className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-slate-900 mb-2 uppercase tracking-wide">Full Name</label>
                <input name="name" value={addForm.name} onChange={handleAddChange} required className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white text-slate-900" placeholder="e.g., Jane Doe" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold text-slate-900 mb-2 uppercase tracking-wide">Age</label>
                  <input name="age" type="number" min="0" value={addForm.age} onChange={handleAddChange} required className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white text-slate-900" placeholder="e.g., 48" />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-slate-900 mb-2 uppercase tracking-wide">Gender</label>
                  <select name="gender" value={addForm.gender} onChange={handleAddChange} className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white text-slate-900">
                    <option>Female</option>
                    <option>Male</option>
                    <option>Other</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-semibold text-slate-900 mb-2 uppercase tracking-wide">Latest Diagnosis (optional)</label>
                <input name="latest_diagnosis" value={addForm.latest_diagnosis} onChange={handleAddChange} className="w-full border-2 border-slate-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-300 bg-white text-slate-900" placeholder="e.g., Benign Fibroadenoma" />
              </div>
              <div className="flex justify-end space-x-4 pt-4">
                <button type="button" onClick={() => setShowAddModal(false)} className="px-6 py-3 rounded-xl border-2 border-slate-300 text-slate-700 hover:bg-slate-100 font-semibold transition-all duration-300">Cancel</button>
                <button disabled={adding} type="submit" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 disabled:opacity-50">
                  {adding ? 'Saving...' : 'Save Patient'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
