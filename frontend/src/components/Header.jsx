import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Heart, Home, Activity, Brain, User, LogOut, ChevronDown, Clock } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

const Header = () => {
  const { user, logout } = useAuth()
  const [showUserMenu, setShowUserMenu] = useState(false)
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/auth')
  }

  return (
    <header className="bg-white/90 backdrop-blur-xl border-b-2 border-pink-200/50 shadow-xl shadow-purple-300/30 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex items-center justify-between">
          <Link to="/dashboard" className="flex items-center space-x-4 group">
            <div className="relative">
              <div className="w-14 h-14 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500 rounded-2xl flex items-center justify-center shadow-xl shadow-pink-500/40 group-hover:shadow-pink-500/60 transition-all duration-300 group-hover:scale-110 group-hover:rotate-6">
                <Heart className="h-8 w-8 text-white" fill="white" />
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full animate-pulse shadow-lg shadow-orange-500/50"></div>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-orange-600 bg-clip-text text-transparent">
                MedicalDiagnostics
              </h1>
              <p className="text-sm font-semibold text-slate-700 tracking-wide">
                🩺 AI Multi-Modal Diagnostic Platform
              </p>
            </div>
          </Link>
          
          <nav className="flex items-center space-x-3">
            <Link 
              to="/dashboard" 
              className="flex items-center space-x-3 px-6 py-3 rounded-xl text-slate-700 hover:text-purple-600 hover:bg-purple-50 transition-all duration-300 font-semibold border-2 border-transparent hover:border-purple-300 hover:shadow-lg"
            >
              <Home className="h-5 w-5" />
              <span>Dashboard</span>
            </Link>
            
            <Link 
              to="/multi-modal" 
              className="flex items-center space-x-3 px-6 py-3 rounded-xl text-slate-700 hover:text-pink-600 hover:bg-pink-50 transition-all duration-300 font-semibold border-2 border-transparent hover:border-pink-300 hover:shadow-lg"
            >
              <Activity className="h-5 w-5" />
              <span>Multi-Modal Test</span>
            </Link>
            
            <Link 
              to="/medical-tester" 
              className="flex items-center space-x-3 px-6 py-3 rounded-xl text-slate-700 hover:text-orange-600 hover:bg-orange-50 transition-all duration-300 font-semibold border-2 border-transparent hover:border-orange-300 hover:shadow-lg"
            >
              <Brain className="h-5 w-5" />
              <span>X-ray Test</span>
            </Link>

            {user ? (
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-200 hover:shadow-lg hover:shadow-purple-300/50 transition-all"
                >
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-500 rounded-full flex items-center justify-center text-white font-bold text-sm shadow-lg">
                    {user.full_name?.charAt(0) || 'U'}
                  </div>
                  <span className="font-semibold text-slate-700">{user.full_name}</span>
                  <ChevronDown className={`h-4 w-4 text-slate-600 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
                </button>

                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-64 bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden animate-fade-in">
                    <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 border-b border-slate-200">
                      <p className="font-semibold text-slate-800">{user.full_name}</p>
                      <p className="text-sm text-slate-600">{user.email}</p>
                    </div>
                    
                    <div className="p-2">
                      <Link
                        to="/history"
                        className="flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-purple-50 transition-all"
                        onClick={() => setShowUserMenu(false)}
                      >
                        <Clock className="w-5 h-5 text-purple-600" />
                        <span className="font-medium text-slate-700">Test History</span>
                      </Link>
                      
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-red-50 transition-all text-left"
                      >
                        <LogOut className="w-5 h-5 text-red-600" />
                        <span className="font-medium text-red-600">Logout</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <Link
                to="/auth"
                className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 via-pink-600 to-orange-500 text-white font-semibold hover:shadow-xl hover:shadow-pink-500/50 hover:scale-105 transition-all"
              >
                <User className="h-5 w-5" />
                <span>Sign In</span>
              </Link>
            )}
            
            <div className="flex items-center space-x-3 px-6 py-3 rounded-xl bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-300 shadow-lg">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50"></div>
              <span className="text-sm text-orange-700 font-bold tracking-wide">🔒 SECURE</span>
            </div>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header
