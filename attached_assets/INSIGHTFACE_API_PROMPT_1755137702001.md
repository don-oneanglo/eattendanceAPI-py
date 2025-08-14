
# InsightFace API Backend - Replit Development Prompt

## Project Overview
Create a Python FastAPI backend service to replace the existing face_recognition library with InsightFace for improved face recognition in an attendance management system. This API will integrate with an existing Node.js server and MySQL database.

## Core Requirements

### 1. Technology Stack
- **Framework**: FastAPI (Python 3.8+)
- **Face Recognition**: InsightFace library with ONNX models
- **Database**: MySQL (existing connection details provided)
- **Model**: Use buffalo_l or antelopev2 ONNX models
- **Port**: 8000 (with CORS enabled for Node.js frontend integration)

### 2. Database Integration
Connect to existing MySQL database with these credentials:
```python
DATABASE_CONFIG = {
    'host': 'srv1521.hstgr.io',
    'user': 'u311154254_TADB',
    'password': 'Anglo!123456',
    'database': 'u311154254_TestAttendance',
    'charset': 'utf8mb4'
}
```

### 3. Database Schema Compatibility
The API must work with these existing tables:

#### Student Table
```sql
CREATE TABLE Student (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    StudentCode nchar(10),
    StudentNickname nvarchar(100),
    StudentName nvarchar(200),
    StudentImage image,
    EmailAddress nvarchar(100),
    Campus nvarchar(50),
    Form nvarchar(100),
    CreatedDate DATETIME DEFAULT GETDATE()
)
```

#### Teacher Table
```sql
CREATE TABLE Teacher (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    TeacherCode nchar(10),
    TeacherNickname nvarchar(100),
    TeacherName nvarchar(200),
    TeacherImage image,
    EmailAddress nvarchar(100),
    Campus nvarchar(50),
    Department nvarchar(100),
    CreatedDate DATETIME DEFAULT GETDATE()
)
```

#### FaceData Table (for embeddings storage)
```sql
CREATE TABLE FaceData (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    PersonType VARCHAR(20) CHECK (PersonType IN ('student', 'teacher')),
    PersonCode nchar(10) NOT NULL,
    ImageData VARBINARY(MAX),
    FaceDescriptor NVARCHAR(MAX), -- Store InsightFace embeddings as JSON
    OriginalName VARCHAR(255),
    ContentType VARCHAR(100),
    CreatedDate DATETIME DEFAULT GETDATE()
)
```

#### AttendanceRecords Table
```sql
CREATE TABLE AttendanceRecords (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    SessionId INT NOT NULL,
    StudentCode nchar(10) NOT NULL,
    Status VARCHAR(20) CHECK (Status IN ('Present', 'Absent', 'Late')),
    AttendanceDate DATETIME DEFAULT GETDATE(),
    CreatedDate DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (SessionId) REFERENCES Sessions(Id)
)
```

## Required API Endpoints

### 1. Health Check
```
GET /api/health
Response: {
    "status": "healthy",
    "message": "All systems operational",
    "enrolled_faces": 0,
    "model_loaded": true,
    "similarity_threshold": 0.6
}
```

### 2. Face Enrollment
```
POST /api/face/enroll
Body: {
    "person_type": "student|teacher",
    "person_code": "S001",
    "person_name": "John Doe",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "original_name": "photo.jpg"
}
Response: {
    "success": true,
    "message": "Face enrolled successfully",
    "person_code": "S001",
    "embedding_id": 123
}
```

### 3. Face Recognition
```
POST /api/face/recognize
Body: {
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "person_type": "student|teacher", // optional filter
    "similarity_threshold": 0.6, // optional, default 0.6
    "source_image_name": "attendance_scan.jpg" // optional for logging
}
Response: {
    "success": true,
    "match": {
        "person_type": "student",
        "person_code": "S001",
        "person_name": "John Doe",
        "confidence": 0.89,
        "similarity_score": 0.92
    },
    "processing_time_ms": 245
}
```

### 4. List Enrolled Faces
```
GET /api/face/list/{person_type}
GET /api/face/list/student
GET /api/face/list/teacher
GET /api/face/list/all
Response: {
    "success": true,
    "faces": [
        {
            "id": 1,
            "person_type": "student",
            "person_code": "S001",
            "person_name": "John Doe",
            "enrollment_date": "2024-01-15T10:30:00",
            "has_image": true
        }
    ],
    "total_count": 25
}
```

### 5. Delete Face Enrollment
```
DELETE /api/face/delete
Body: {
    "person_type": "student",
    "person_code": "S001"
}
Response: {
    "success": true,
    "message": "Face enrollment deleted successfully"
}
```

### 6. Batch Operations
```
POST /api/face/batch-enroll
Body: {
    "enrollments": [
        {
            "person_type": "student",
            "person_code": "S001",
            "person_name": "John Doe",
            "image_base64": "..."
        }
    ]
}
Response: {
    "success": true,
    "results": [
        {"person_code": "S001", "success": true, "message": "Enrolled"},
        {"person_code": "S002", "success": false, "message": "No face detected"}
    ],
    "summary": {
        "total": 2,
        "successful": 1,
        "failed": 1
    }
}
```

### 7. Face Quality Assessment
```
POST /api/face/quality-check
Body: {
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
}
Response: {
    "success": true,
    "quality": {
        "score": 0.85,
        "face_detected": true,
        "face_count": 1,
        "face_size": "adequate",
        "brightness": "good",
        "blur_level": "low",
        "recommendation": "suitable_for_enrollment"
    }
}
```

## Technical Specifications

### 1. InsightFace Configuration
- Use ONNX runtime for model inference
- Support both buffalo_l and antelopev2 models
- Generate 512-dimensional embeddings
- Implement face detection with confidence threshold > 0.7
- Select largest face when multiple faces detected

### 2. Performance Requirements
- Face detection: < 500ms per image
- Face recognition: < 300ms against 1000 stored faces
- Memory usage: < 2GB during operation
- Support concurrent requests (up to 10 simultaneous)
- Auto-download models on first run

### 3. Image Processing
- Accept base64 encoded images (JPEG, PNG, WebP)
- Maximum image size: 10MB
- Automatic image resizing for optimal processing
- Face alignment for better accuracy
- Quality validation before enrollment

### 4. Database Operations
- Connection pooling for MySQL
- Embedding storage as JSON in FaceDescriptor field
- Efficient similarity search using vector operations
- Backup and restore functionality for embeddings
- Migration support from existing face_recognition format

### 5. Security & Validation
- Input validation for all endpoints
- Rate limiting (100 requests per minute per IP)
- Image format validation
- SQL injection prevention
- Error handling with detailed logging

## Integration Requirements

### 1. Node.js Server Compatibility
The API must be compatible with these existing Node.js endpoints:
- `/api/face/register` - maps to `/api/face/enroll`
- `/api/face/recognize` - direct mapping
- `/api/face-descriptors/:personType` - maps to `/api/face/list/:person_type`
- `/api/admin/face/:id` (DELETE) - maps to `/api/face/delete`

### 2. Frontend Compatibility
Support existing frontend calls from:
- `attendance-script.js` - real-time face recognition
- `admin-script.js` - face enrollment and management
- `script.js` - teacher login via face recognition

### 3. Bulk Import Support
Support CSV bulk operations for:
- Student face enrollment from photo directories
- Teacher face enrollment from photo directories
- Batch processing with progress reporting

## Environment Configuration

### 1. Environment Variables
```
INSIGHTFACE_MODEL=buffalo_l
SIMILARITY_THRESHOLD=0.6
MAX_CONCURRENT_REQUESTS=10
DATABASE_POOL_SIZE=5
LOG_LEVEL=INFO
ENABLE_FACE_QUALITY_CHECK=true
AUTO_DOWNLOAD_MODELS=true
```

### 2. Model Management
- Automatic model download on startup
- Model caching in `/models` directory
- Support for model switching without restart
- Model integrity verification

## Logging & Monitoring

### 1. Recognition Logs
Store recognition attempts in database:
```sql
CREATE TABLE FaceRecognitionLogs (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    PersonCode nchar(10),
    PersonType VARCHAR(20),
    Confidence DECIMAL(5,4),
    ProcessingTimeMs INT,
    SourceImageName VARCHAR(255),
    RecognitionDate DATETIME DEFAULT GETDATE(),
    Success BIT
)
```

### 2. Performance Metrics
- Recognition accuracy tracking
- Processing time monitoring
- Memory usage tracking
- Error rate monitoring

## Error Handling

### 1. Common Error Responses
```json
{
    "success": false,
    "error_code": "NO_FACE_DETECTED",
    "message": "No face detected in the provided image",
    "details": {
        "suggestion": "Ensure the image contains a clear, frontal face"
    }
}
```

### 2. Error Codes
- `NO_FACE_DETECTED` - No face found in image
- `MULTIPLE_FACES` - Multiple faces detected (use largest)
- `LOW_QUALITY_IMAGE` - Image quality too poor
- `DUPLICATE_ENROLLMENT` - Person already enrolled
- `PERSON_NOT_FOUND` - Person not in database
- `MODEL_NOT_LOADED` - InsightFace model not available
- `DATABASE_ERROR` - Database connection/query error

## Deployment Configuration

### 1. Requirements File
```txt
fastapi==0.104.1
uvicorn==0.24.0
insightface==0.7.3
onnxruntime==1.16.3
opencv-python-headless==4.8.1.78
Pillow==10.1.0
numpy==1.24.4
mysql-connector-python==8.2.0
python-multipart==0.0.6
pydantic==2.5.0
```

### 2. Startup Configuration
- Bind to `0.0.0.0:8000`
- Enable auto-reload for development
- Production-ready error handling
- Graceful shutdown handling

### 3. File Structure
```
/
├── main.py                 # FastAPI application
├── models/
│   ├── __init__.py
│   ├── database.py         # Database models
│   └── insightface_model.py # InsightFace wrapper
├── services/
│   ├── __init__.py
│   ├── face_service.py     # Face processing logic
│   └── database_service.py # Database operations
├── utils/
│   ├── __init__.py
│   ├── image_utils.py      # Image processing utilities
│   └── validation.py      # Input validation
├── config.py               # Configuration management
├── requirements.txt
└── README.md
```

## Testing Requirements

### 1. Unit Tests
- Face enrollment with various image qualities
- Recognition accuracy with different similarity thresholds
- Database operations (CRUD)
- Error handling scenarios

### 2. Integration Tests
- End-to-end face enrollment and recognition
- Bulk operations testing
- Performance benchmarks
- Node.js integration testing

### 3. Load Testing
- Concurrent recognition requests
- Large database performance (1000+ faces)
- Memory usage under load
- Response time consistency

## Migration Strategy

### 1. Data Migration
- Convert existing face_recognition encodings to InsightFace embeddings
- Preserve all existing face data and metadata
- Validation of migration accuracy
- Rollback capability

### 2. Fallback Support
- Maintain compatibility with face_recognition during transition
- Gradual migration of face data
- A/B testing capability
- Performance comparison tools

Create a complete, production-ready FastAPI service that can be deployed on Replit and seamlessly integrated with the existing Node.js attendance system. The API should maintain all current functionality while providing improved face recognition accuracy and performance.
