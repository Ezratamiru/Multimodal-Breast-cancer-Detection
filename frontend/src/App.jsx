import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'
import Header from './components/Header'
import Auth from './components/Auth'
import DashboardNew from './components/DashboardNew'
import PatientProfile from './components/PatientProfile'
import BiopsyClassification from './components/BiopsyClassification'
import MedicalTester from './components/MedicalTester'
import MultiModalTester from './components/MultiModalTester'
import ParticleSystem from './components/ParticleSystem'

function App() {
  // Temporarily disable auth check to test UI
  // const { isAuthenticated } = useAuth()

  return (
    <div className="min-h-screen">
      <ParticleSystem />
      
      <Routes>
        {/* Public Route - Auth (for testing) */}
        <Route path="/auth" element={<Auth />} />
        
        {/* All Routes (temporarily open without auth) */}
        <Route path="/*" element={
          <>
            <Header />
            <main>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" />} />
                <Route path="/dashboard" element={<DashboardNew />} />
                <Route path="/patient/:patientId" element={<PatientProfile />} />
                <Route path="/patient/:patientId/biopsy" element={<BiopsyClassification />} />
                <Route path="/medical-tester" element={<MedicalTester />} />
                <Route path="/multi-modal" element={<MultiModalTester />} />
              </Routes>
            </main>
          </>
        } />
      </Routes>
    </div>
  )
}

export default App
