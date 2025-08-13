"""
FastAPI backend for face recognition using InsightFace.
Provides endpoints for enrollment, verification, and identification.
"""

import os
import json
import uuid
from typing import List, Optional
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np

from face_engine import FaceEngine, FaceEngineError

# Initialize face engine
face_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global face_engine
    try:
        face_engine = FaceEngine()
        print("Face recognition engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize face engine: {e}")
        raise
    yield
    # Cleanup code can go here if needed
    print("Application shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="InsightFace-powered face recognition system for enrollment, verification, and identification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image(file: UploadFile) -> Image.Image:
    """Validate and load uploaded image file."""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG or PNG)"
        )
    
    if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG and PNG images are supported"
        )
    
    try:
        image = Image.open(BytesIO(file.file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )


@app.post("/enroll")
async def enroll_face(
    name: str = Form(..., description="Name of the person to enroll"),
    file: UploadFile = File(..., description="Image file containing the face to enroll")
):
    """
    Enroll a new face in the system.
    
    - **name**: Unique identifier for the person
    - **file**: Image file (JPG/PNG) containing exactly one face
    
    Returns enrollment details including face ID and bounding box.
    """
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    # Validate inputs
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name cannot be empty"
        )
    
    name = name.strip()
    
    # Check if name already exists
    if face_engine.is_name_enrolled(name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Person with name '{name}' is already enrolled"
        )
    
    # Validate and load image
    image = validate_image(file)
    
    try:
        result = face_engine.enroll_face(name, image)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": f"Successfully enrolled face for '{name}'",
                "face_id": result["face_id"],
                "name": result["name"],
                "bounding_box": result["bounding_box"],
                "confidence": result["confidence"]
            }
        )
    except FaceEngineError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/verify")
async def verify_faces(
    file1: UploadFile = File(..., description="First image for comparison"),
    file2: UploadFile = File(..., description="Second image for comparison")
):
    """
    Verify if two images contain the same person.
    
    - **file1**: First image file (JPG/PNG)
    - **file2**: Second image file (JPG/PNG)
    
    Returns similarity score and match result.
    """
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    # Validate and load images
    image1 = validate_image(file1)
    image2 = validate_image(file2)
    
    try:
        result = face_engine.verify_faces(image1, image2)
        return {
            "is_match": result["is_match"],
            "similarity_score": result["similarity_score"],
            "threshold": result["threshold"],
            "face1_bbox": result["face1_bbox"],
            "face2_bbox": result["face2_bbox"],
            "confidence1": result["confidence1"],
            "confidence2": result["confidence2"]
        }
    except FaceEngineError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/identify")
async def identify_face(
    file: UploadFile = File(..., description="Image file to identify")
):
    """
    Identify a face by searching enrolled faces database.
    
    - **file**: Image file (JPG/PNG) containing the face to identify
    
    Returns top 3 matches with similarity scores.
    """
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    # Validate and load image
    image = validate_image(file)
    
    try:
        result = face_engine.identify_face(image)
        return {
            "matches": result["matches"],
            "bounding_box": result["bounding_box"],
            "confidence": result["confidence"],
            "total_enrolled": result["total_enrolled"]
        }
    except FaceEngineError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/faces")
async def list_enrolled_faces():
    """
    Get list of all enrolled faces with metadata.
    
    Returns list of enrolled persons with their details.
    """
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    try:
        faces = face_engine.list_enrolled_faces()
        return {
            "total_count": len(faces),
            "faces": faces
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/faces/{name}")
async def delete_enrolled_face(name: str):
    """
    Delete an enrolled face from the database.
    
    - **name**: Name of the person to remove
    
    Returns confirmation of deletion.
    """
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name cannot be empty"
        )
    
    name = name.strip()
    
    try:
        success = face_engine.delete_face(name)
        if success:
            return {
                "message": f"Successfully deleted face for '{name}'",
                "deleted_name": name
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No enrolled face found for name '{name}'"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Face Recognition API is running",
        "status": "healthy",
        "engine_initialized": face_engine is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check with system status."""
    if not face_engine:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": "Face recognition engine not initialized"
            }
        )
    
    try:
        enrolled_count = len(face_engine.list_enrolled_faces())
        return {
            "status": "healthy",
            "message": "All systems operational",
            "enrolled_faces": enrolled_count,
            "model_loaded": True,
            "similarity_threshold": face_engine.similarity_threshold
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": f"Engine error: {str(e)}"
            }
        )


if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("storage", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info"
    )
