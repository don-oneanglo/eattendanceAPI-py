# Overview

This is a Face Recognition API built with Python FastAPI and the InsightFace library. The system provides face enrollment, verification, and identification capabilities using the high-accuracy buffalo_l model. It's designed as a production-ready backend service that can register faces, verify if two images contain the same person, and identify individuals from a database of enrolled faces. The API uses cosine similarity matching with configurable thresholds and maintains persistent storage of face embeddings.

## Recent Updates (January 2025)
- **SQL Server Integration**: Configured API to connect to user's SQL Server database (srv1521.hstgr.io)
- **Database Schema Support**: Full support for FaceData, Student, Teacher, and AttendanceRecords tables
- **Dual Storage System**: API works with both file-based storage and SQL Server database
- **Recognition Logging**: Automatic logging of face recognition attempts with confidence scores
- **Production Ready**: Configured for deployment with proper SQL Server drivers (pyodbc)

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Framework
- **FastAPI**: Chosen for its automatic API documentation, built-in validation, and high performance
- **ASGI Server**: Uses Uvicorn for production-grade async request handling
- **CORS Middleware**: Configured to allow cross-origin requests for frontend integration

## Face Recognition Engine
- **InsightFace Library**: Utilizes the buffalo_l model for state-of-the-art face recognition accuracy
- **Model Management**: Automatic model downloading and caching in local `/models` directory
- **Embedding Extraction**: Generates 512-dimensional face embeddings for comparison
- **Similarity Matching**: Uses cosine similarity with configurable threshold (default 0.6)

## Data Storage
- **Dual Storage System**: Both file-based JSON and SQL Server database integration
- **SQL Server Integration**: Configured for user's database (srv1521.hstgr.io)
- **Database Schema**: Supports FaceData, Student, Teacher, and AttendanceRecords tables
- **Embedding Storage**: Face embeddings stored as JSON in FaceDescriptor field
- **Image Storage**: Binary image data stored in ImageData field (VARBINARY)
- **Recognition Logging**: Automatic creation of FaceRecognitionLogs table for tracking
- **Fallback Support**: File-based storage available when database unavailable

## Image Processing
- **PIL/Pillow**: Handles image format conversion and basic processing
- **OpenCV**: Used by InsightFace for computer vision operations
- **NumPy**: Array operations for embedding calculations and comparisons
- **Multi-format Support**: Accepts various image formats through multipart form uploads

## API Design
- **RESTful Endpoints**: Standard HTTP methods for different operations
- **Form Data Upload**: File upload handling for image submissions
- **Error Handling**: Custom exception classes with proper HTTP status codes
- **Validation**: Built-in FastAPI validation for request parameters

## Modular Architecture
- **Separation of Concerns**: Face engine logic separated from API layer
- **Engine Abstraction**: FaceEngine class encapsulates all face recognition operations
- **Startup Initialization**: Lazy loading of models during application startup
- **Configurable Parameters**: Threshold and model settings can be adjusted

# External Dependencies

## Core Libraries
- **InsightFace (0.7.3)**: Primary face recognition library providing the buffalo_l model
- **ONNX Runtime (1.17.1)**: Neural network inference engine for model execution
- **FastAPI (0.111.0)**: Web framework for API endpoints and request handling
- **Uvicorn (0.29.0)**: ASGI server for running the FastAPI application

## Image Processing
- **Pillow (10.2.0)**: Image manipulation and format conversion
- **NumPy (1.26.4)**: Numerical computing for array operations and embeddings
- **OpenCV**: Integrated through InsightFace for computer vision tasks

## Utilities
- **Python Multipart (0.0.9)**: File upload handling for image submissions

## Model Dependencies
- **Buffalo_l Model**: Downloaded automatically from InsightFace model zoo
- **ONNX Models**: Pre-trained neural network models for face detection and recognition

## System Requirements
- **Python 3.10+**: Required runtime environment
- **Local File System**: Storage for models, database, and temporary files
- **Internet Connection**: Required for initial model download only