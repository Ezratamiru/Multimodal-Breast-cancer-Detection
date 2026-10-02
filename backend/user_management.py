"""
User Management System for Medical Diagnostics
Handles user authentication, profiles, and test history
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime
import json
import os
import hashlib
import secrets

class User(BaseModel):
    user_id: str
    email: EmailStr
    full_name: str
    role: str = "patient"  # patient, doctor, admin
    created_at: datetime
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "patient"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TestResult(BaseModel):
    test_id: str
    user_id: str
    modality: str  # xray_mammogram, ultrasound, mri, biopsy
    test_date: datetime
    findings: Dict
    recommendations: Optional[Dict] = None
    notes: Optional[str] = None

class UserManager:
    """Manages users and their test history"""
    
    def __init__(self, data_dir="user_data"):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.json")
        self.history_file = os.path.join(data_dir, "test_history.json")
        self._ensure_data_dir()
        self._load_data()
    
    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump([], f)
    
    def _load_data(self):
        """Load users and history from files"""
        with open(self.users_file, 'r') as f:
            self.users = json.load(f)
        with open(self.history_file, 'r') as f:
            self.history = json.load(f)
    
    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _save_history(self):
        """Save history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return password_hash.hex(), salt
    
    def create_user(self, user_data: UserCreate) -> Dict:
        """Create a new user"""
        # Check if user exists
        if user_data.email in self.users:
            raise ValueError("User already exists")
        
        # Hash password
        password_hash, salt = self._hash_password(user_data.password)
        
        # Create user ID
        user_id = secrets.token_urlsafe(16)
        
        # Create user
        user = {
            "user_id": user_id,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "role": user_data.role,
            "password_hash": password_hash,
            "salt": salt,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        self.users[user_data.email] = user
        self._save_users()
        
        # Return user without sensitive data
        return {
            "user_id": user_id,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "role": user_data.role,
            "created_at": user["created_at"]
        }
    
    def authenticate_user(self, login_data: UserLogin) -> Optional[Dict]:
        """Authenticate user and return user data"""
        if login_data.email not in self.users:
            return None
        
        user = self.users[login_data.email]
        password_hash, _ = self._hash_password(login_data.password, user["salt"])
        
        if password_hash == user["password_hash"]:
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self._save_users()
            
            # Return user without sensitive data
            return {
                "user_id": user["user_id"],
                "email": user["email"],
                "full_name": user["full_name"],
                "role": user["role"],
                "last_login": user["last_login"]
            }
        
        return None
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        for user in self.users.values():
            if user["user_id"] == user_id:
                return {
                    "user_id": user["user_id"],
                    "email": user["email"],
                    "full_name": user["full_name"],
                    "role": user["role"],
                    "created_at": user["created_at"],
                    "last_login": user["last_login"]
                }
        return None
    
    def save_test_result(self, test_result: TestResult) -> str:
        """Save a test result to history"""
        test_id = secrets.token_urlsafe(16)
        
        result_data = {
            "test_id": test_id,
            "user_id": test_result.user_id,
            "modality": test_result.modality,
            "test_date": test_result.test_date.isoformat(),
            "findings": test_result.findings,
            "recommendations": test_result.recommendations,
            "notes": test_result.notes
        }
        
        self.history.append(result_data)
        self._save_history()
        
        return test_id
    
    def get_user_history(self, user_id: str, modality: Optional[str] = None) -> List[Dict]:
        """Get test history for a user"""
        user_tests = [test for test in self.history if test["user_id"] == user_id]
        
        if modality:
            user_tests = [test for test in user_tests if test["modality"] == modality]
        
        # Sort by date (newest first)
        user_tests.sort(key=lambda x: x["test_date"], reverse=True)
        
        return user_tests
    
    def get_test_result(self, test_id: str) -> Optional[Dict]:
        """Get a specific test result"""
        for test in self.history:
            if test["test_id"] == test_id:
                return test
        return None
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user"""
        user_tests = self.get_user_history(user_id)
        
        return {
            "total_tests": len(user_tests),
            "tests_by_modality": {
                "xray_mammogram": len([t for t in user_tests if t["modality"] == "xray_mammogram"]),
                "ultrasound": len([t for t in user_tests if t["modality"] == "ultrasound"]),
                "mri": len([t for t in user_tests if t["modality"] == "mri"]),
                "biopsy": len([t for t in user_tests if t["modality"] == "biopsy"])
            },
            "last_test_date": user_tests[0]["test_date"] if user_tests else None,
            "pending_recommendations": sum(1 for t in user_tests if t.get("recommendations"))
        }


# Initialize global user manager
user_manager = UserManager()


if __name__ == "__main__":
    # Test the system
    manager = UserManager("test_user_data")
    
    # Create test user
    try:
        user = manager.create_user(UserCreate(
            email="test@example.com",
            password="secure_password",
            full_name="Test User",
            role="patient"
        ))
        print(f"✅ Created user: {user['full_name']}")
    except ValueError as e:
        print(f"⚠️ User exists: {e}")
    
    # Test authentication
    auth_result = manager.authenticate_user(UserLogin(
        email="test@example.com",
        password="secure_password"
    ))
    
    if auth_result:
        print(f"✅ Authentication successful: {auth_result['full_name']}")
        
        # Save a test result
        test_result = TestResult(
            test_id="",
            user_id=auth_result['user_id'],
            modality="xray_mammogram",
            test_date=datetime.now(),
            findings={"prediction": "Benign", "confidence": 0.85},
            recommendations={"next_test": "ultrasound"},
            notes="Routine screening"
        )
        
        test_id = manager.save_test_result(test_result)
        print(f"✅ Saved test result: {test_id}")
        
        # Get history
        history = manager.get_user_history(auth_result['user_id'])
        print(f"✅ User history: {len(history)} tests")
        
        # Get stats
        stats = manager.get_user_stats(auth_result['user_id'])
        print(f"✅ User stats: {stats}")
    else:
        print("❌ Authentication failed")
