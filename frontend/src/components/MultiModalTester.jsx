import React, { useState, useEffect } from 'react';
import { Upload, Activity, Microscope, Scan, Zap, ChevronRight, AlertCircle, CheckCircle, Info } from 'lucide-react';
import { modalityService, recommendationService, historyService } from '../services/api';
import { useAuth } from '../context/AuthContext';

const MultiModalTester = () => {
  const [activeModality, setActiveModality] = useState('ultrasound');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { user } = useAuth();

  // MRI Features State (76 features)
  const [mriFeatures, setMriFeatures] = useState(Array(76).fill(0));
  
  // Biopsy Features State (30 features)
  const [biopsyFeatures, setBiopsyFeatures] = useState({
    radius_mean: 0,
    texture_mean: 0,
    perimeter_mean: 0,
    area_mean: 0,
    smoothness_mean: 0,
    compactness_mean: 0,
    concavity_mean: 0,
    concave_points_mean: 0,
    symmetry_mean: 0,
    fractal_dimension_mean: 0,
    radius_se: 0,
    texture_se: 0,
    perimeter_se: 0,
    area_se: 0,
    smoothness_se: 0,
    compactness_se: 0,
    concavity_se: 0,
    concave_points_se: 0,
    symmetry_se: 0,
    fractal_dimension_se: 0,
    radius_worst: 0,
    texture_worst: 0,
    perimeter_worst: 0,
    area_worst: 0,
    smoothness_worst: 0,
    compactness_worst: 0,
    concavity_worst: 0,
    concave_points_worst: 0,
    symmetry_worst: 0,
    fractal_dimension_worst: 0
  });

  const modalities = [
    {
      id: 'ultrasound',
      name: 'Ultrasound Density',
      icon: Activity,
      color: 'from-blue-500 to-cyan-500',
      description: 'Breast density assessment with BI-RADS classification',
      inputType: 'image'
    },
    {
      id: 'mri',
      name: 'MRI Analysis',
      icon: Scan,
      color: 'from-purple-500 to-pink-500',
      description: 'DCE-MRI feature-based risk classification',
      inputType: 'features'
    },
    {
      id: 'biopsy',
      name: 'Biopsy Classification',
      icon: Microscope,
      color: 'from-emerald-500 to-teal-500',
      description: 'Fine Needle Aspiration cytology analysis',
      inputType: 'features'
    }
  ];

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setRecommendations(null);
      setError(null);
    }
  };

  const loadExampleFeatures = async (modality) => {
    try {
      if (modality === 'biopsy') {
        const examples = await modalityService.getExampleFeatures();
        if (examples.benign_examples?.length > 0) {
          setBiopsyFeatures(examples.benign_examples[0]);
        }
      }
    } catch (err) {
      console.error('Failed to load example features:', err);
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      let result;
      
      if (activeModality === 'ultrasound') {
        if (!selectedFile) {
          setError('Please upload an ultrasound image');
          setLoading(false);
          return;
        }
        result = await modalityService.predictDensity(selectedFile);
      } else if (activeModality === 'mri') {
        result = await modalityService.predictMRI(mriFeatures);
      } else if (activeModality === 'biopsy') {
        result = await modalityService.predictBiopsy(biopsyFeatures);
      }

      setResults(result);

      // Get recommendations
      try {
        const recs = await recommendationService.getNextTests(activeModality, result);
        setRecommendations(recs);
      } catch (err) {
        console.error('Failed to get recommendations:', err);
      }

      // Save to history if user is logged in
      if (user) {
        try {
          await historyService.saveTestResult({
            user_id: user.user_id,
            modality: activeModality,
            test_date: new Date().toISOString(),
            findings: result,
            recommendations: recommendations
          });
        } catch (err) {
          console.error('Failed to save to history:', err);
        }
      }

    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8 animate-fade-in-down">
        <div className="relative bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-white/50">
          <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-br from-blue-400/20 to-indigo-400/20 rounded-full blur-3xl pointer-events-none" />
          
          <div className="relative">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl shadow-lg">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Multi-Modal AI Diagnostics
                </h1>
                <p className="text-slate-600 mt-1">
                  Advanced breast cancer screening with intelligent recommendations
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modality Selector */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modalities.map((modality, idx) => {
            const Icon = modality.icon;
            return (
              <button
                key={modality.id}
                onClick={() => {
                  setActiveModality(modality.id);
                  setResults(null);
                  setRecommendations(null);
                  setError(null);
                  setSelectedFile(null);
                  setPreviewUrl(null);
                }}
                className={`relative p-6 rounded-2xl border-2 transition-all duration-300 animate-fade-in-up ${
                  activeModality === modality.id
                    ? 'border-blue-500 bg-white shadow-xl shadow-blue-500/20 scale-105'
                    : 'border-white/50 bg-white/60 hover:bg-white/80 hover:border-blue-300 hover:shadow-lg'
                }`}
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${modality.color} mb-4`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-2">{modality.name}</h3>
                <p className="text-sm text-slate-600">{modality.description}</p>
                
                {activeModality === modality.id && (
                  <div className="absolute top-4 right-4">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Section */}
        <div className="space-y-6">
          <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-white/50 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
            <h2 className="text-2xl font-bold text-slate-800 mb-6">Input Data</h2>

            {activeModality === 'ultrasound' && (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-blue-300 rounded-2xl p-8 text-center hover:border-blue-500 transition-all cursor-pointer bg-blue-50/50">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="ultrasound-upload"
                  />
                  <label htmlFor="ultrasound-upload" className="cursor-pointer">
                    <Upload className="w-12 h-12 text-blue-500 mx-auto mb-4" />
                    <p className="text-slate-700 font-semibold mb-2">Upload Ultrasound Image</p>
                    <p className="text-sm text-slate-500">PNG, JPG up to 10MB</p>
                  </label>
                </div>

                {previewUrl && (
                  <div className="rounded-2xl overflow-hidden border border-slate-200 animate-scale-in">
                    <img src={previewUrl} alt="Preview" className="w-full h-64 object-cover" />
                  </div>
                )}
              </div>
            )}

            {activeModality === 'mri' && (
              <div className="space-y-4">
                <div className="p-4 bg-purple-50 border border-purple-200 rounded-xl">
                  <Info className="w-5 h-5 text-purple-600 inline mr-2" />
                  <span className="text-sm text-purple-800">
                    MRI requires 76 pre-extracted features from DCE-MRI scans
                  </span>
                </div>
                
                <div className="max-h-96 overflow-y-auto space-y-3 pr-2">
                  {[...Array(10)].map((_, i) => (
                    <input
                      key={i}
                      type="number"
                      step="0.01"
                      value={mriFeatures[i]}
                      onChange={(e) => {
                        const newFeatures = [...mriFeatures];
                        newFeatures[i] = parseFloat(e.target.value) || 0;
                        setMriFeatures(newFeatures);
                      }}
                      className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/20 transition-all outline-none"
                      placeholder={`Feature ${i + 1}`}
                    />
                  ))}
                  <p className="text-sm text-slate-500 text-center py-2">
                    Showing 10 of 76 features...
                  </p>
                </div>
              </div>
            )}

            {activeModality === 'biopsy' && (
              <div className="space-y-4">
                <button
                  onClick={() => loadExampleFeatures('biopsy')}
                  className="w-full py-3 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-xl hover:shadow-lg transition-all"
                >
                  Load Example Data
                </button>

                <div className="max-h-96 overflow-y-auto space-y-3 pr-2">
                  {Object.entries(biopsyFeatures).slice(0, 10).map(([key, value]) => (
                    <div key={key}>
                      <label className="text-sm font-medium text-slate-700 mb-1 block capitalize">
                        {key.replace(/_/g, ' ')}
                      </label>
                      <input
                        type="number"
                        step="0.01"
                        value={value}
                        onChange={(e) => setBiopsyFeatures({
                          ...biopsyFeatures,
                          [key]: parseFloat(e.target.value) || 0
                        })}
                        className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/20 transition-all outline-none"
                      />
                    </div>
                  ))}
                  <p className="text-sm text-slate-500 text-center py-2">
                    Showing 10 of 30 features...
                  </p>
                </div>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={loading || (activeModality === 'ultrasound' && !selectedFile)}
              className="w-full mt-6 py-4 px-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Analyze
                </>
              )}
            </button>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3 animate-scale-in">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          {results && (
            <>
              <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-white/50 animate-scale-in">
                <h2 className="text-2xl font-bold text-slate-800 mb-6">Results</h2>

                {/* Ultrasound Results */}
                {activeModality === 'ultrasound' && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl border border-blue-200">
                      <p className="text-sm text-slate-600 mb-2">Breast Density</p>
                      <p className="text-4xl font-bold text-blue-600">
                        {results.density_percentage?.toFixed(1)}%
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-white rounded-xl border border-slate-200">
                        <p className="text-sm text-slate-600 mb-1">BI-RADS</p>
                        <p className="text-2xl font-bold text-slate-800">{results.birads_category}</p>
                      </div>
                      <div className="p-4 bg-white rounded-xl border border-slate-200">
                        <p className="text-sm text-slate-600 mb-1">Risk Level</p>
                        <p className={`text-xl font-bold ${
                          results.risk_assessment === 'Low' ? 'text-green-600' :
                          results.risk_assessment === 'Moderate' ? 'text-yellow-600' :
                          'text-red-600'
                        }`}>
                          {results.risk_assessment}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* MRI Results */}
                {activeModality === 'mri' && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl border border-purple-200">
                      <p className="text-sm text-slate-600 mb-2">Classification</p>
                      <p className="text-3xl font-bold text-purple-600">{results.prediction}</p>
                    </div>
                    
                    <div className="p-4 bg-white rounded-xl border border-slate-200">
                      <p className="text-sm text-slate-600 mb-1">Risk Level</p>
                      <p className="text-xl font-bold text-slate-800">{results.risk_level}</p>
                    </div>
                    
                    <div className="p-4 bg-white rounded-xl border border-slate-200">
                      <p className="text-sm text-slate-600 mb-2">Confidence</p>
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                          style={{ width: `${(results.confidence * 100).toFixed(0)}%` }}
                        />
                      </div>
                      <p className="text-right text-sm text-slate-600 mt-1">
                        {(results.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                )}

                {/* Biopsy Results */}
                {activeModality === 'biopsy' && (
                  <div className="space-y-4">
                    <div className="p-6 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-2xl border border-emerald-200">
                      <p className="text-sm text-slate-600 mb-2">Diagnosis</p>
                      <p className={`text-3xl font-bold ${
                        results.prediction === 'Benign' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {results.prediction}
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-white rounded-xl border border-slate-200">
                        <p className="text-sm text-slate-600 mb-1">Stage</p>
                        <p className="text-2xl font-bold text-slate-800">{results.stage}</p>
                      </div>
                      <div className="p-4 bg-white rounded-xl border border-slate-200">
                        <p className="text-sm text-slate-600 mb-1">Confidence</p>
                        <p className="text-xl font-bold text-emerald-600">
                          {(results.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Recommendations */}
              {recommendations && (
                <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-white/50 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
                  <h2 className="text-2xl font-bold text-slate-800 mb-6">Recommendations</h2>

                  <div className={`p-4 rounded-xl mb-4 ${
                    recommendations.urgency === 'urgent' ? 'bg-red-50 border border-red-200' :
                    recommendations.urgency === 'prompt' ? 'bg-yellow-50 border border-yellow-200' :
                    'bg-green-50 border border-green-200'
                  }`}>
                    <p className="text-sm font-semibold uppercase tracking-wide mb-1">
                      {recommendations.urgency}
                    </p>
                    <p className="text-slate-700">{recommendations.clinical_pathway}</p>
                  </div>

                  {recommendations.recommended_tests?.map((test, idx) => (
                    <div key={idx} className="p-4 bg-white rounded-xl border border-slate-200 mb-3 hover:shadow-md transition-all">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p className="font-semibold text-slate-800 mb-1">{test.modality}</p>
                          <p className="text-sm text-slate-600">{test.reason}</p>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          test.priority === 'urgent' ? 'bg-red-100 text-red-700' :
                          test.priority === 'high' ? 'bg-orange-100 text-orange-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          {test.priority}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {!results && !loading && (
            <div className="bg-white/60 backdrop-blur-xl rounded-3xl p-12 shadow-xl border border-white/50 text-center">
              <Activity className="w-16 h-16 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500 text-lg">
                Upload data and click Analyze to see results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MultiModalTester;
