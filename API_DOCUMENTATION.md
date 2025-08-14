# Face Recognition API Documentation

## Overview

This is a complete Python FastAPI backend for face recognition using the InsightFace library. The API provides face enrollment, verification, and identification capabilities using the high-accuracy buffalo_l model. All data is stored exclusively in MySQL database with no file-based storage.

**Base URL**: `http://localhost:5000` (or your deployed URL)  
**Database**: MySQL (srv1521.hstgr.io)  
**Model**: InsightFace buffalo_l  
**Storage**: MySQL database only  

## Authentication

Currently, no authentication is required for API endpoints. All endpoints are publicly accessible.

## System Status

### GET `/api/health`

Check the health status of the API and database connectivity.

**Response Example:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "enrolled_faces": 0,
  "database_face_count": 0,
  "model_loaded": true,
  "similarity_threshold": 0.6,
  "database_connected": true,
  "database_config": {
    "host": "srv1521.hstgr.io",
    "database": "u311154254_TestAttendance",
    "user": "u311154254_TADB"
  }
}
```

### GET `/`

Get basic API information and available endpoints.

**Response Example:**
```json
{
  "message": "InsightFace Attendance API",
  "status": "running",
  "version": "2.0.0",
  "engine_initialized": true,
  "endpoints": {
    "health": "/api/health",
    "enroll": "/api/face/enroll",
    "recognize": "/api/face/recognize",
    "list": "/api/face/list/{person_type}",
    "delete": "/api/face/delete",
    "quality_check": "/api/face/quality-check",
    "batch_enroll": "/api/face/batch-enroll",
    "docs": "/docs"
  }
}
```

## Face Management

### POST `/api/face/enroll`

Enroll a new face in the system. Stores face embedding and image data in MySQL database.

**Request Body:**
```json
{
  "person_type": "student",
  "person_code": "STU001",
  "person_name": "John Doe",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "original_name": "john_photo.jpg"
}
```

**Request Fields:**
- `person_type` (string): Type of person ("student" or "teacher")
- `person_code` (string): Unique identifier for the person
- `person_name` (string): Full name of the person
- `image_base64` (string): Base64 encoded image with data URL prefix
- `original_name` (string, optional): Original filename of the image

**Response Example:**
```json
{
  "success": true,
  "message": "Face enrolled successfully in database",
  "person_code": "STU001",
  "person_name": "John Doe",
  "embedding_id": 123,
  "processing_time_ms": 245,
  "confidence": 0.95
}
```

### POST `/api/face/recognize`

Recognize a face from the enrolled database.

**Request Body:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "person_type": "student",
  "similarity_threshold": 0.6,
  "source_image_name": "camera_capture.jpg"
}
```

**Request Fields:**
- `image_base64` (string): Base64 encoded image with data URL prefix
- `person_type` (string, optional): Filter by person type ("student" or "teacher")
- `similarity_threshold` (float, optional): Custom similarity threshold (default: 0.6)
- `source_image_name` (string, optional): Name/source of the image for logging

**Response Example (Match Found):**
```json
{
  "success": true,
  "match": {
    "person_type": "student",
    "person_code": "STU001",
    "person_name": "John Doe",
    "confidence": 0.87,
    "similarity_score": 0.87
  },
  "processing_time_ms": 156
}
```

**Response Example (No Match):**
```json
{
  "success": true,
  "match": null,
  "processing_time_ms": 142
}
```

### GET `/api/face/list/{person_type}`

List enrolled faces by person type.

**Parameters:**
- `person_type` (path): Person type to filter ("student", "teacher", or "all")

**Response Example:**
```json
{
  "success": true,
  "faces": [
    {
      "id": 123,
      "person_type": "student",
      "person_code": "STU001",
      "person_name": "John Doe",
      "created_date": "2025-01-14T10:30:00",
      "has_image": true
    }
  ],
  "total_count": 1
}
```

### DELETE `/api/face/delete`

Delete a face enrollment from the database.

**Request Body:**
```json
{
  "person_type": "student",
  "person_code": "STU001"
}
```

**Response Example:**
```json
{
  "success": true,
  "message": "Face enrollment deleted successfully for student STU001"
}
```

## Face Analysis

### POST `/api/face/quality-check`

Assess the quality of a face image before enrollment.

**Request Body:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response Example:**
```json
{
  "success": true,
  "quality": {
    "score": 0.9,
    "face_detected": true,
    "face_count": 1,
    "face_size": "adequate",
    "brightness": "good",
    "blur_level": "low",
    "recommendation": "suitable_for_enrollment"
  }
}
```

**Quality Recommendations:**
- `suitable_for_enrollment`: High quality, recommended for enrollment
- `acceptable_with_caution`: Medium quality, may work but not optimal
- `not_suitable`: Poor quality, should not be used for enrollment

## Batch Operations

### POST `/api/face/batch-enroll`

Enroll multiple faces in a single request.

**Request Body:**
```json
{
  "enrollments": [
    {
      "person_type": "student",
      "person_code": "STU001",
      "person_name": "John Doe",
      "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
      "original_name": "john.jpg"
    },
    {
      "person_type": "student",
      "person_code": "STU002",
      "person_name": "Jane Smith",
      "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
      "original_name": "jane.jpg"
    }
  ]
}
```

**Response Example:**
```json
{
  "success": true,
  "results": [
    {
      "person_code": "STU001",
      "person_name": "John Doe",
      "success": true,
      "message": "Enrolled successfully",
      "database_id": 123
    },
    {
      "person_code": "STU002",
      "person_name": "Jane Smith",
      "success": false,
      "message": "No faces detected in the image"
    }
  ],
  "summary": {
    "total": 2,
    "successful": 1,
    "failed": 1
  }
}
```

## Legacy Endpoints

For backward compatibility with older frontends:

### POST `/enroll`

Legacy face enrollment endpoint using form data.

**Request (Form Data):**
- `name` (string): Person name/code
- `file` (file): Image file (JPG/PNG)

### POST `/identify`

Legacy face identification endpoint using form data.

**Request (Form Data):**
- `file` (file): Image file to identify

## Error Responses

All endpoints return consistent error responses:

**400 Bad Request:**
```json
{
  "detail": "No faces detected in the image"
}
```

**404 Not Found:**
```json
{
  "detail": "No enrollment found for student STU001"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error: Database connection failed"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Face recognition engine not initialized"
}
```

## Image Requirements

- **Formats**: JPG, JPEG, PNG
- **Encoding**: Base64 with data URL prefix (e.g., `data:image/jpeg;base64,`)
- **Face Count**: Exactly one face per image
- **Quality**: Clear, well-lit images work best
- **Size**: No strict size limits, but reasonable file sizes recommended

## Database Schema

The API uses the following MySQL database tables:

### FaceData Table
- `Id` (INT, Primary Key): Unique face record ID
- `PersonType` (VARCHAR): "student" or "teacher"
- `PersonCode` (VARCHAR): Unique person identifier
- `PersonName` (VARCHAR): Full name of the person
- `ImageData` (LONGBLOB): Binary image data
- `FaceDescriptor` (JSON): Face embedding as JSON array
- `OriginalName` (VARCHAR): Original filename
- `ContentType` (VARCHAR): Image MIME type
- `CreatedDate` (DATETIME): Enrollment timestamp
- `HasImage` (BOOLEAN): Whether image data is stored

### FaceRecognitionLogs Table
- `Id` (INT, Primary Key): Log entry ID
- `PersonCode` (VARCHAR): Recognized person code (null if no match)
- `PersonType` (VARCHAR): Person type filter used
- `Confidence` (FLOAT): Recognition confidence score
- `ProcessingTimeMs` (INT): Processing time in milliseconds
- `SourceImageName` (VARCHAR): Source image identifier
- `Success` (BOOLEAN): Whether recognition was successful
- `CreatedDate` (DATETIME): Recognition attempt timestamp

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production use.

## Performance Notes

- Face embedding extraction: ~200-500ms per image
- Face recognition: ~100-300ms per query
- Database operations: ~10-50ms per query
- Batch operations: Processed sequentially, not in parallel

## Development

- **Interactive Documentation**: Available at `/docs` (Swagger UI)
- **Health Monitoring**: Use `/api/health` for system status
- **Logging**: All operations are logged with timestamps and confidence scores