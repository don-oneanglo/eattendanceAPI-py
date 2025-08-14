# InsightFace Attendance API Documentation

## Overview
A complete Python FastAPI backend for face recognition using the InsightFace library, designed to replace face-api.js with server-side processing. The API includes face enrollment, verification, identification, and management capabilities with comprehensive endpoint support.

## Base URL
```
http://0.0.0.0:5000
```

## Authentication
Currently, no authentication is required for API endpoints.

## API Endpoints

### Health Check
- **GET** `/api/health`
- **GET** `/` (Basic info)

### Face Management
- **POST** `/api/face/enroll` - Enroll a new face
- **POST** `/api/face/recognize` - Recognize a face
- **GET** `/api/face/list/{person_type}` - List enrolled faces
- **DELETE** `/api/face/delete` - Delete a face enrollment
- **POST** `/api/face/quality-check` - Assess image quality
- **POST** `/api/face/batch-enroll` - Batch enroll multiple faces

### Legacy Compatibility
- **POST** `/enroll` - Legacy face enrollment
- **POST** `/identify` - Legacy face identification
- **GET** `/faces` - Legacy list faces
- **DELETE** `/faces/{name}` - Legacy delete face

## Request/Response Formats

### Face Enrollment
**POST** `/api/face/enroll`

Request body:
```json
{
  "person_type": "student",
  "person_code": "ST001",
  "person_name": "John Doe",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "original_name": "john_photo.jpg"
}
```

Response:
```json
{
  "success": true,
  "message": "Face enrolled successfully",
  "person_code": "ST001",
  "embedding_id": 1234
}
```

### Face Recognition
**POST** `/api/face/recognize`

Request body:
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "person_type": "student",
  "similarity_threshold": 0.6,
  "source_image_name": "test_image.jpg"
}
```

Response:
```json
{
  "success": true,
  "match": {
    "person_type": "student",
    "person_code": "ST001",
    "person_name": "John Doe",
    "confidence": 0.85,
    "similarity_score": 0.85
  },
  "processing_time_ms": 250
}
```

### List Enrolled Faces
**GET** `/api/face/list/{person_type}`

- `person_type`: "student", "teacher", or "all"

Response:
```json
{
  "success": true,
  "faces": [
    {
      "id": 1234,
      "person_type": "student",
      "person_code": "ST001",
      "person_name": "John Doe",
      "enrollment_date": "2024-01-15T10:30:00",
      "has_image": true
    }
  ],
  "total_count": 1
}
```

### Face Quality Check
**POST** `/api/face/quality-check`

Request body:
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

Response:
```json
{
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

### Delete Face
**DELETE** `/api/face/delete`

Request body:
```json
{
  "person_type": "student",
  "person_code": "ST001"
}
```

Response:
```json
{
  "success": true,
  "message": "Face enrollment deleted successfully"
}
```

### Batch Enrollment
**POST** `/api/face/batch-enroll`

Request body:
```json
{
  "enrollments": [
    {
      "person_type": "student",
      "person_code": "ST001",
      "person_name": "John Doe",
      "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
      "original_name": "john.jpg"
    },
    {
      "person_type": "teacher",
      "person_code": "TC001",
      "person_name": "Jane Smith",
      "image_base64": "data:image/png;base64,iVBORw0KGgo...",
      "original_name": "jane.png"
    }
  ]
}
```

Response:
```json
{
  "success": true,
  "results": [
    {
      "person_code": "ST001",
      "success": true,
      "message": "Enrolled"
    },
    {
      "person_code": "TC001",
      "success": true,
      "message": "Enrolled"
    }
  ],
  "summary": {
    "total": 2,
    "successful": 2,
    "failed": 0
  }
}
```

## Error Responses

All errors follow this format:
```json
{
  "success": false,
  "error_code": "INVALID_IMAGE",
  "message": "No faces detected in the image",
  "details": {
    "additional_info": "value"
  }
}
```

Common error codes:
- `INVALID_IMAGE` - Image format or quality issues
- `NO_FACE_DETECTED` - No face found in image
- `MULTIPLE_FACES` - Multiple faces detected (enrollment only)
- `PERSON_ALREADY_EXISTS` - Person already enrolled
- `PERSON_NOT_FOUND` - Person not found for deletion
- `ENGINE_NOT_INITIALIZED` - Face recognition engine not ready

## Image Requirements

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Base64 encoded with data URI format

### Quality Guidelines
- **Resolution**: Minimum 200x200 pixels
- **Face Size**: At least 80x80 pixels
- **Quality**: Clear, well-lit images
- **Orientation**: Frontal face view preferred
- **Background**: Any background acceptable
- **Multiple Faces**: Only one face per enrollment image

### Base64 Format
Images must be provided as base64 data URIs:
```
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...
```

## Configuration

### Environment Variables
- `SIMILARITY_THRESHOLD`: Default similarity threshold (0.6)
- `FACE_CONFIDENCE_THRESHOLD`: Minimum face detection confidence (0.7)
- `MAX_IMAGE_SIZE_MB`: Maximum image size in MB (10)
- `PROCESSING_TIMEOUT_SECONDS`: Processing timeout (30)

### Model Information
- **Model**: InsightFace buffalo_l
- **Embedding Size**: 512 dimensions
- **Similarity Metric**: Cosine similarity
- **Detection Engine**: ONNX Runtime with CPU provider

## Performance

### Typical Response Times
- Face Enrollment: 200-500ms
- Face Recognition: 150-300ms
- Quality Check: 100-200ms
- List Faces: <50ms

### Throughput
- Concurrent requests supported: 10
- Rate limiting: 100 requests/minute (configurable)

## Migration from face-api.js

This API provides full compatibility for migrating from face-api.js:

1. **Detection**: Use `/api/face/quality-check` instead of face detection
2. **Recognition**: Use `/api/face/recognize` instead of face matching
3. **Descriptors**: Handled automatically with 512-dim embeddings
4. **Database**: File-based storage with JSON format

### Key Differences
- Server-side processing (more accurate, faster)
- No browser compatibility concerns
- Centralized model management
- Better security and performance
- RESTful API instead of JavaScript library

## Testing

Use the automatic API documentation at:
```
http://0.0.0.0:5000/docs
```

This provides an interactive interface for testing all endpoints with example requests and responses.