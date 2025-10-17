import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Users, Activity, Microscope, Scan, TrendingUp, ChevronRight, UserPlus, X, AlertCircle, Clock, CheckCircle2 } from 'lucide-react';
import { patientService, userService, historyService } from '../services/api';
import { useAuth } from '../context/AuthContext';

const DashboardNew = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [userStats, setUserStats] = useState(null);
  const [recentTests, setRecentTests] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [addForm, setAddForm] = useState({ name: '', age: '', gender: 'Female' });
  const [adding, setAdding] = useState(false);
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    fetchDashboardData();
  }, [user]);

  const fetchDashboardData = async () => {
    try {
      // Fetch patients
      const patientsData = await patientService.getPatients();
      setPatients(patientsData);

      // Fetch user stats if logged in
      if (user) {
        const stats = await userService.getUserStats(user.user_id);
        setUserStats(stats);

        // Fetch recent tests
        const history = await historyService.getUserHistory(user.user_id);
        setRecentTests(history.tests?.slice(0, 5) || []);
      }
    } catch (err) {
      console.error('Failed to load dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddPatient = async (e) => {
    e.preventDefault();
    if (!addForm.name || !addForm.age) return;
    
    setAdding(true);
    try {
      const payload = {
        name: addForm.name,
        age: parseInt(addForm.age, 10),
        gender: addForm.gender,
        latest_diagnosis: 'Pending Assessment'
      };
      const created = await patientService.createPatient(payload);
      setPatients([...patients, created]);
      setShowAddModal(false);
      setAddForm({ name: '', age: '', gender: 'Female' });
    } catch (err) {
      console.error('Failed to add patient:', err);
    } finally {
      setAdding(false);
    }
  };

  const stats = [
    {
      label: 'Total Patients',
      value: patients.length,
      icon: Users,
      color: 'from-blue-500 to-cyan-500',
      trend: '+12%'
    },
    {
      label: 'Tests Completed',
      value: userStats?.total_tests || 0,
      icon: Activity,
      color: 'from-purple-500 to-pink-500',
      trend: '+8%'
    },
    {
      label: 'Pending Reviews',
      value: userStats?.pending_recommendations || 0,
      icon: Clock,
      color: 'from-orange-500 to-red-500',
      trend: '-3%'
    },
    {
      label: 'AI Accuracy',
      value: '97.6%',
      icon: TrendingUp,
      color: 'from-emerald-500 to-teal-500',
      trend: '+2%'
    }
  ];

  const quickActions = [
    {
      title: 'Multi-Modal Analysis',
      description: 'Ultrasound, MRI, and Biopsy testing',
      icon: Activity,
      color: 'from-blue-500 to-cyan-500',
      link: '/multi-modal'
    },
    {
      title: 'X-ray Mammogram',
      description: 'Mammogram mass detection',
      icon: Scan,
      color: 'from-purple-500 to-pink-500',
      link: '/medical-tester'
    },
    {
      title: 'Test History',
      description: 'View all past diagnostic tests',
      icon: Clock,
      color: 'from-emerald-500 to-teal-500',
      link: '/history'
    },
    {
      title: 'Add New Patient',
      description: 'Register new patient profile',
      icon: UserPlus,
      color: 'from-orange-500 to-red-500',
      action: () => setShowAddModal(true)
    }
  ];

  const getModalityIcon = (modality) => {
    switch (modality) {
      case 'ultrasound': return Activity;
      case 'mri': return Scan;
      case 'biopsy': return Microscope;
      default: return Activity;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-500 via-pink-500 to-orange-400 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4" />
          <div className="absolute w-12 h-12 border-4 border-white/20 border-t-white/60 rounded-full animate-ping mx-auto" />
          <p className="text-white font-semibold text-lg">Loading Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-500 via-pink-500 to-orange-400 p-6">
      {/* Welcome Header */}
      <div className="max-w-7xl mx-auto mb-8 animate-fade-in-down">
        <div className="relative bg-white/95 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border-2 border-yellow-300/50 overflow-hidden">
          <div className="absolute -top-20 -right-20 w-64 h-64 bg-gradient-to-br from-yellow-400/30 to-orange-400/30 rounded-full blur-3xl" />
          <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-gradient-to-br from-pink-400/30 to-purple-400/30 rounded-full blur-3xl" />
          
          <div className="relative">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-orange-600 bg-clip-text text-transparent mb-2">
              {user ? `Welcome back, ${user.full_name}` : 'Medical Diagnostics Dashboard'}
            </h1>
            <p className="text-slate-700 font-medium">
              🩺 AI-powered multi-modal breast cancer diagnostic system
            </p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, idx) => {
            const Icon = stat.icon;
            return (
              <div
                key={idx}
                className="bg-gradient-to-br from-white via-purple-50 to-pink-50 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border-2 border-purple-200/60 hover:scale-105 hover:shadow-pink-300/50 hover:shadow-2xl transition-all duration-300 animate-fade-in-up"
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${stat.color}`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <span className="text-sm font-semibold text-green-600">{stat.trend}</span>
                </div>
                <p className="text-3xl font-bold text-slate-800 mb-1">{stat.value}</p>
                <p className="text-sm text-slate-600">{stat.label}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="max-w-7xl mx-auto mb-8">
        <h2 className="text-2xl font-bold text-white drop-shadow-lg mb-6 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
          ⚡ Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {quickActions.map((action, idx) => {
            const Icon = action.icon;
            const content = (
              <div
                className="group bg-gradient-to-br from-white via-blue-50 to-indigo-50 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border-2 border-blue-200/60 hover:shadow-purple-400/50 hover:shadow-2xl hover:scale-105 hover:border-purple-300 transition-all duration-300 cursor-pointer animate-fade-in-up"
                style={{ animationDelay: `${(idx + 4) * 0.1}s` }}
              >
                <div className={`p-3 rounded-xl bg-gradient-to-br ${action.color} mb-4 group-hover:rotate-6 transition-transform duration-300`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-slate-800 mb-2">{action.title}</h3>
                <p className="text-sm text-slate-600 mb-4">{action.description}</p>
                <div className="flex items-center text-blue-600 font-semibold text-sm">
                  Get Started <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            );

            return action.link ? (
              <Link key={idx} to={action.link}>
                {content}
              </Link>
            ) : (
              <div key={idx} onClick={action.action}>
                {content}
              </div>
            );
          })}
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Tests */}
        {user && recentTests.length > 0 && (
          <div className="bg-gradient-to-br from-white via-green-50 to-emerald-50 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border-2 border-green-200/60 animate-fade-in-up" style={{ animationDelay: '0.8s' }}>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-6">📋 Recent Tests</h2>
            <div className="space-y-4">
              {recentTests.map((test, idx) => {
                const Icon = getModalityIcon(test.modality);
                return (
                  <div key={idx} className="p-4 bg-white rounded-xl border border-slate-200 hover:shadow-md transition-all">
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <Icon className="w-5 h-5 text-blue-600" />
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-slate-800 capitalize">{test.modality}</p>
                        <p className="text-sm text-slate-600">{new Date(test.test_date).toLocaleDateString()}</p>
                      </div>
                      <ChevronRight className="w-5 h-5 text-slate-400" />
                    </div>
                  </div>
                );
              })}
            </div>
            <Link to="/history" className="block mt-4 text-center text-blue-600 font-semibold hover:text-blue-700">
              View All History →
            </Link>
          </div>
        )}

        {/* Patients List */}
        <div className="bg-gradient-to-br from-white via-orange-50 to-yellow-50 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border-2 border-orange-200/60 animate-fade-in-up" style={{ animationDelay: '0.9s' }}>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">👥 Patients</h2>
            <button
              onClick={() => setShowAddModal(true)}
              className="px-5 py-3 bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 text-white rounded-xl hover:shadow-xl hover:shadow-pink-500/50 hover:scale-105 transition-all flex items-center gap-2 font-semibold"
            >
              <UserPlus className="w-5 h-5" />
              Add Patient
            </button>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {patients.map((patient, idx) => (
              <Link
                key={patient.id}
                to={`/patient/${patient.id}`}
                className="block p-4 bg-white rounded-xl border border-slate-200 hover:shadow-md transition-all animate-fade-in-up"
                style={{ animationDelay: `${(idx + 10) * 0.05}s` }}
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 via-pink-500 to-orange-500 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                    {patient.name.charAt(0)}
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-slate-800">{patient.name}</p>
                    <p className="text-sm text-slate-600">{patient.age} years • {patient.gender}</p>
                  </div>
                  <ChevronRight className="w-5 h-5 text-slate-400" />
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Add Patient Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-white/95 backdrop-blur-xl rounded-3xl p-8 max-w-md w-full shadow-2xl border-2 border-purple-300/50 animate-scale-in">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">✨ Add New Patient</h2>
              <button
                onClick={() => setShowAddModal(false)}
                className="p-2 hover:bg-slate-100 rounded-xl transition-all"
              >
                <X className="w-5 h-5 text-slate-600" />
              </button>
            </div>

            <form onSubmit={handleAddPatient} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Full Name</label>
                <input
                  type="text"
                  value={addForm.name}
                  onChange={(e) => setAddForm({ ...addForm, name: e.target.value })}
                  className="w-full px-4 py-3 rounded-xl border-2 border-purple-200 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/30 transition-all outline-none bg-white"
                  placeholder="Enter patient name"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Age</label>
                <input
                  type="number"
                  value={addForm.age}
                  onChange={(e) => setAddForm({ ...addForm, age: e.target.value })}
                  className="w-full px-4 py-3 rounded-xl border-2 border-purple-200 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/30 transition-all outline-none bg-white"
                  placeholder="Enter age"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Gender</label>
                <select
                  value={addForm.gender}
                  onChange={(e) => setAddForm({ ...addForm, gender: e.target.value })}
                  className="w-full px-4 py-3 rounded-xl border-2 border-purple-200 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/30 transition-all outline-none bg-white"
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-xl hover:bg-slate-100 hover:border-slate-400 transition-all font-semibold"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={adding}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 text-white rounded-xl hover:shadow-xl hover:shadow-pink-500/50 hover:scale-105 transition-all font-semibold disabled:opacity-50"
                >
                  {adding ? '✨ Adding...' : '✨ Add Patient'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default DashboardNew;
