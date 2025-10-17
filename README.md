# Medical Diagnostics - Breast Cancer Detection System

A comprehensive medical diagnostics application for multi-modal breast cancer detection featuring patient management, imaging analysis, and biopsy classification.

## Features

### 1. Patient Dashboard & Data Integration
- Central patient dashboard with comprehensive patient information
- Patient status summary including latest diagnosis and activities
- Quick access to all patient modules

### 2. Multi-Modal Detection
- **Mammogram Analysis**: Interactive image viewer with mass segmentation
- **Ultrasound Imaging**: Ultrasound scan analysis and findings
- **MRI Scanning**: MRI image analysis and interpretation
- Tabbed interface for easy navigation between imaging modalities

### 3. Mammogram Image Segmentation
- Interactive image viewer with zoom and download capabilities
- Automated mass detection with boundary highlighting
- Detailed mass characteristics display (size, shape, density)
- Real-time overlay with segmentation data

### 4. Biopsy Data Classification
- Automated cancer staging based on clinical parameters
- Input fields for tumor size, lymph node involvement, grade, and receptor status
- Real-time stage calculation with detailed explanations
- Historical biopsy data tracking

## Technology Stack

### Frontend
- **React 18** with Vite for fast development
- **React Router** for navigation
- **Axios** for API communication
- **Lucide React** for icons
- **CSS3** with modern styling

### Backend
- **FastAPI** for high-performance API
- **Pydantic** for data validation
- **Uvicorn** ASGI server
- **CORS** middleware for frontend integration

## Project Structure

```
MedicalDiagnostics/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── static/
│       └── images/          # Medical image storage
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API services
│   │   ├── App.jsx         # Main application
│   │   └── main.jsx        # Entry point
│   ├── package.json        # Node.js dependencies
│   └── vite.config.js      # Vite configuration
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

## API Endpoints

### Patient Management
- `GET /api/patients` - Get all patients
- `GET /api/patients/{patient_id}` - Get specific patient
- `GET /api/patients/{patient_id}/imaging` - Get patient imaging results
- `GET /api/patients/{patient_id}/imaging/{type}` - Get imaging by type

### Biopsy Classification
- `GET /api/patients/{patient_id}/biopsy` - Get patient biopsy data
- `POST /api/patients/{patient_id}/biopsy` - Create biopsy classification

## Usage Guide

### 1. Patient Dashboard
- View all patients in the system
- Click on any patient to access their detailed profile
- Monitor patient statistics and active cases

### 2. Patient Profile
- Access comprehensive patient information
- Navigate between different imaging modalities using tabs
- View detailed imaging analysis and findings

### 3. Imaging Analysis
- **Mammogram**: View images with automated mass detection overlays
- **Ultrasound**: Analyze ultrasound findings and recommendations
- **MRI**: Review MRI scans and enhancement patterns

### 4. Biopsy Classification
- Input clinical parameters (tumor size, lymph nodes, grade, receptors)
- Automatic cancer stage calculation
- View detailed stage explanations and treatment implications

## Sample Data

The application comes with pre-loaded sample data including:
- 2 sample patients with different diagnostic scenarios
- Multiple imaging results for each modality
- Sample biopsy data with various cancer stages

## Development

### Adding New Features
1. Backend: Add new endpoints in `main.py`
2. Frontend: Create new components in `src/components/`
3. Update API service in `src/services/api.js`

### Customization
- Modify styling in `src/index.css`
- Update sample data in `backend/main.py`
- Add new imaging modalities by extending the tabs system

## Security Considerations

- CORS is configured for development (localhost:5173, localhost:3000)
- Input validation using Pydantic models
- Error handling for API failures
- Secure file handling for medical images

## Future Enhancements

- Real medical image integration
- Advanced AI/ML models for detection
- User authentication and authorization
- DICOM image support
- Integration with hospital systems
- Advanced reporting and analytics

## License

This project is for educational and demonstration purposes. Ensure compliance with medical data regulations (HIPAA, GDPR) before using with real patient data.

## Support

For issues and questions, please refer to the project documentation or create an issue in the repository.
