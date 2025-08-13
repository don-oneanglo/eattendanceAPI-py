# Face Recognition API

A complete Python FastAPI backend for face recognition using the InsightFace library. Supports face enrollment, verification, and identification with persistent storage.

## Features

- **Face Enrollment**: Register new faces with unique names
- **Face Verification**: Compare two images to verify if they contain the same person
- **Face Identification**: Search enrolled faces to identify a person
- **Face Management**: List and delete enrolled faces
- **High Accuracy**: Uses InsightFace buffalo_l model for superior recognition performance
- **Persistent Storage**: JSON-based database with automatic model downloading
- **Production Ready**: Comprehensive error handling and validation

## Installation & Setup

### Prerequisites

- Python 3.10 or later
- pip package manager

### Install Dependencies

```bash
pip install insightface==0.7.3 onnxruntime==1.17.1 fastapi==0.111.0 uvicorn==0.29.0 pillow==10.2.0 numpy==1.26.4 python-multipart==0.0.9
