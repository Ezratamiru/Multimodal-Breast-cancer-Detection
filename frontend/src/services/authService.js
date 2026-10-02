import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

class AuthService {
  constructor() {
    this.user = null;
    this.loadUserFromStorage();
  }

  loadUserFromStorage() {
    const userData = localStorage.getItem('user');
    if (userData) {
      this.user = JSON.parse(userData);
    }
  }

  saveUserToStorage(user) {
    localStorage.setItem('user', JSON.stringify(user));
    this.user = user;
  }

  clearUserFromStorage() {
    localStorage.removeItem('user');
    this.user = null;
  }

  async register(email, password, fullName, role = 'patient') {
    try {
      const response = await axios.post(`${API_URL}/users/register`, {
        email,
        password,
        full_name: fullName,
        role
      });
      
      if (response.data.success) {
        return { success: true, user: response.data.user };
      }
      return { success: false, error: 'Registration failed' };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Registration failed'
      };
    }
  }

  async login(email, password) {
    try {
      const response = await axios.post(`${API_URL}/users/login`, {
        email,
        password
      });
      
      if (response.data.success) {
        this.saveUserToStorage(response.data.user);
        return { success: true, user: response.data.user };
      }
      return { success: false, error: 'Login failed' };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Invalid credentials'
      };
    }
  }

  logout() {
    this.clearUserFromStorage();
  }

  getCurrentUser() {
    return this.user;
  }

  isAuthenticated() {
    return this.user !== null;
  }

  async getUserProfile(userId) {
    try {
      const response = await axios.get(`${API_URL}/users/${userId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch user profile:', error);
      return null;
    }
  }

  async getUserStats(userId) {
    try {
      const response = await axios.get(`${API_URL}/users/${userId}/stats`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch user stats:', error);
      return null;
    }
  }
}

export default new AuthService();
