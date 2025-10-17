import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Multi-Modal Analysis Services
export const modalityService = {
  // X-ray Mammogram
  predictMammogram: async (imageFile) => {
    const formData = new FormData()
    formData.append('file', imageFile)
    const response = await api.post('/api/mammogram/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  // Ultrasound / Density Assessment
  predictDensity: async (imageFile) => {
    const formData = new FormData()
    formData.append('file', imageFile)
    const response = await api.post('/api/density/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  getDensityModelInfo: async () => {
    const response = await api.get('/api/density/model-info')
    return response.data
  },

  getBiradsCategories: async () => {
    const response = await api.get('/api/density/birads-categories')
    return response.data
  },

  // MRI Analysis
  predictMRI: async (features) => {
    const response = await api.post('/api/mri/predict', { features })
    return response.data
  },

  getMRIModelInfo: async () => {
    const response = await api.get('/api/mri/model-info')
    return response.data
  },

  // Biopsy Classification
  predictBiopsy: async (features) => {
    const response = await api.post('/api/busi/predict', features)
    return response.data
  },

  getBiopsyModelInfo: async () => {
    const response = await api.get('/api/busi/model-info')
    return response.data
  },

  getFeatureImportance: async () => {
    const response = await api.get('/api/busi/feature-importance')
    return response.data
  },

  getExampleFeatures: async () => {
    const response = await api.get('/api/busi/example-features')
    return response.data
  },

  // Get all modalities summary
  getModalitiesSummary: async () => {
    const response = await api.get('/api/modalities/summary')
    return response.data
  }
}

// Recommendation Services
export const recommendationService = {
  getNextTests: async (currentModality, findings) => {
    const response = await api.post('/api/recommendations/next-tests', findings, {
      params: { current_modality: currentModality }
    })
    return response.data
  },

  getModalityInfo: async (modality) => {
    const response = await api.get(`/api/recommendations/modality-info/${modality}`)
    return response.data
  },

  getCompleteWorkflow: async () => {
    const response = await api.get('/api/recommendations/complete-workflow')
    return response.data
  }
}

// Test History Services
export const historyService = {
  getUserHistory: async (userId, modality = null) => {
    const params = modality ? { modality } : {}
    const response = await api.get(`/api/users/${userId}/history`, { params })
    return response.data
  },

  saveTestResult: async (testData) => {
    const response = await api.post('/api/tests/save', testData)
    return response.data
  },

  getTestResult: async (testId) => {
    const response = await api.get(`/api/tests/${testId}`)
    return response.data
  }
}

// User Services
export const userService = {
  getUserStats: async (userId) => {
    const response = await api.get(`/api/users/${userId}/stats`)
    return response.data
  },

  getUserProfile: async (userId) => {
    const response = await api.get(`/api/users/${userId}`)
    return response.data
  }
}

// Original Patient Services (keeping for compatibility)
export const patientService = {
  getPatients: async () => {
    const response = await api.get('/api/patients')
    return response.data
  },

  getPatient: async (patientId) => {
    const response = await api.get(`/api/patients/${patientId}`)
    return response.data
  },

  getPatientImaging: async (patientId) => {
    const response = await api.get(`/api/patients/${patientId}/imaging`)
    return response.data
  },

  getPatientImagingByType: async (patientId, imagingType) => {
    const response = await api.get(`/api/patients/${patientId}/imaging/${imagingType}`)
    return response.data
  },

  getPatientBiopsy: async (patientId) => {
    const response = await api.get(`/api/patients/${patientId}/biopsy`)
    return response.data
  },

  createBiopsyClassification: async (patientId, biopsyData) => {
    const response = await api.post(`/api/patients/${patientId}/biopsy`, biopsyData)
    return response.data
  },

  createPatient: async (payload) => {
    const response = await api.post('/api/patients', payload)
    return response.data
  }
}

export default api
