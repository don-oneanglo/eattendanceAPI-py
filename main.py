"""
FastAPI backend for face recognition using InsightFace.
Attendance management system with MySQL database integration.
"""

import os
import json
import base64
import time
import logging
from typing import List, Optional, Dict, Any
from io import BytesIO
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image
import numpy as np
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from face_engine import FaceEngine, FaceEngineError
from services.database_service import DatabaseService
from config import get_settings


# Pydantic models for the new API
class FaceEnrollmentRequest(BaseModel):
    person_type: str = Field(..., pattern="^(student|teacher)$")
    person_code: str = Field(..., min_length=1, max_length=10)
    person_name: str = Field(..., min_length=1, max_length=200)
    image_base64: str = Field(..., description="Base64 encoded image")
    original_name: Optional[str] = None
    
    @validator('image_base64')
    def validate_base64_image(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('Image must be base64 encoded with data URI format')
        return v


class FaceRecognitionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    person_type: Optional[str] = Field(None, pattern="^(student|teacher)$")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0)
    source_image_name: Optional[str] = None
    
    @validator('image_base64')
    def validate_base64_image(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('Image must be base64 encoded with data URI format')
        return v


class FaceDeleteRequest(BaseModel):
    person_type: str = Field(..., pattern="^(student|teacher)$")
    person_code: str = Field(..., min_length=1, max_length=10)


class FaceQualityRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    
    @validator('image_base64')
    def validate_base64_image(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('Image must be base64 encoded with data URI format')
        return v


class BatchEnrollmentItem(BaseModel):
    person_type: str = Field(..., pattern="^(student|teacher)$")
    person_code: str = Field(..., min_length=1, max_length=10)
    person_name: str = Field(..., min_length=1, max_length=200)
    image_base64: str
    original_name: Optional[str] = None


class BatchEnrollmentRequest(BaseModel):
    enrollments: List[BatchEnrollmentItem] = Field(..., min_items=1, max_items=100)


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    try:
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )

# Global services
face_engine = None
db_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global face_engine, db_service
    try:
        settings = get_settings()
        
        # Initialize database service
        try:
            db_service = DatabaseService()
            await db_service.initialize()
            print("SQL Server database service initialized successfully")
        except Exception as e:
            print(f"Warning: Database service failed to initialize: {e}")
            print("Continuing with file-based storage only")
            db_service = None
        
        # Initialize face recognition engine
        face_engine = FaceEngine(
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            model_name=settings.INSIGHTFACE_MODEL
        )
        print("Face recognition engine initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        raise
    yield
    # Cleanup
    if db_service:
        await db_service.close()
    print("Application shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="InsightFace Attendance API",
    description="InsightFace-powered face recognition system for attendance management",
    version="2.0.0",
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


# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint with database connectivity status."""
    if not face_engine:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": "Face recognition engine not initialized",
                "enrolled_faces": 0,
                "model_loaded": False,
                "similarity_threshold": 0.6,
                "database_connected": False
            }
        )
    
    try:
        enrolled_count = len(face_engine.list_enrolled_faces())
        
        # Test database connectivity
        database_connected = False
        database_face_count = 0
        if db_service:
            try:
                database_face_count = await db_service.get_face_count()
                database_connected = True
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
        
        return {
            "status": "healthy",
            "message": "All systems operational",
            "enrolled_faces": enrolled_count,
            "database_face_count": database_face_count,
            "model_loaded": True,
            "similarity_threshold": face_engine.similarity_threshold,
            "database_connected": database_connected,
            "database_config": {
                "host": get_settings().DB_HOST,
                "database": get_settings().DB_NAME,
                "user": get_settings().DB_USER
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": f"Engine error: {str(e)}",
                "enrolled_faces": 0,
                "model_loaded": False,
                "similarity_threshold": 0.6,
                "database_connected": False
            }
        )


@app.post("/api/face/enroll")
async def enroll_face(request: FaceEnrollmentRequest):
    """Enroll a new face in the system."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Decode and validate image
        image = decode_base64_image(request.image_base64)
        
        # Extract face embedding
        embedding, metadata = face_engine.extract_embedding(image)
        
        # Create unique identifier combining person type and code
        unique_name = f"{request.person_type}_{request.person_code}"
        
        # Check if already enrolled
        if face_engine.is_name_enrolled(unique_name):
            # Update existing enrollment
            face_engine.delete_face(unique_name)
        
        # Enroll face in the face engine (file-based storage)
        result = face_engine.enroll_face(unique_name, image)
        
        # Store additional metadata in face database
        face_data = face_engine.faces_db["faces"][unique_name]
        face_data["person_type"] = request.person_type
        face_data["person_code"] = request.person_code
        face_data["person_name"] = request.person_name
        face_data["original_name"] = request.original_name
        face_engine._save_database()
        
        # Also try to store in SQL Server database if available
        database_id = None
        if db_service:
            try:
                # Convert image to bytes for database storage
                image_bytes = db_service.base64_to_bytes(request.image_base64)
                
                database_id = await db_service.enroll_face(
                    person_type=request.person_type,
                    person_code=request.person_code,
                    person_name=request.person_name,
                    image_data=image_bytes,
                    face_descriptor=embedding.tolist(),
                    original_name=request.original_name,
                    content_type="image/jpeg"
                )
                logger.info(f"Face enrolled in SQL Server with ID: {database_id}")
            except Exception as e:
                logger.warning(f"Failed to store in SQL Server database: {e}")
        
        return {
            "success": True,
            "message": "Face enrolled successfully",
            "person_code": request.person_code,
            "embedding_id": database_id or result["face_id"]
        }
        
    except FaceEngineError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/face/recognize")
async def recognize_face(request: FaceRecognitionRequest):
    """Recognize a face from the enrolled database."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Decode and validate image
        image = decode_base64_image(request.image_base64)
        
        # Perform identification with optional person type filter
        result = face_engine.identify_face(image)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        if result["matches"]:
            best_match = result["matches"][0]
            
            # Extract person details from the database entry
            face_data = face_engine.faces_db["faces"].get(best_match["name"])
            if face_data:
                person_type = face_data.get("person_type", "unknown")
                person_code = face_data.get("person_code", best_match["name"])
                person_name = face_data.get("person_name", best_match["name"])
                
                # Filter by person type if specified
                if request.person_type and person_type != request.person_type:
                    return {
                        "success": True,
                        "match": None,
                        "processing_time_ms": processing_time_ms
                    }
                
                # Check similarity threshold
                threshold = request.similarity_threshold or face_engine.similarity_threshold
                if best_match["similarity_score"] >= threshold:
                    # Log successful recognition attempt
                    if db_service:
                        try:
                            await db_service.log_recognition_attempt(
                                person_code=person_code,
                                person_type=person_type,
                                confidence=best_match["similarity_score"],
                                processing_time_ms=processing_time_ms,
                                source_image_name=request.source_image_name,
                                success=True
                            )
                        except Exception as e:
                            logger.warning(f"Failed to log recognition attempt: {e}")
                    
                    return {
                        "success": True,
                        "match": {
                            "person_type": person_type,
                            "person_code": person_code,
                            "person_name": person_name,
                            "confidence": best_match["similarity_score"],
                            "similarity_score": best_match["similarity_score"]
                        },
                        "processing_time_ms": processing_time_ms
                    }
        
        # No match found or below threshold
        # Log the recognition attempt if database is available
        if db_service:
            try:
                await db_service.log_recognition_attempt(
                    person_code=None,
                    person_type=request.person_type,
                    confidence=0.0,
                    processing_time_ms=processing_time_ms,
                    source_image_name=request.source_image_name,
                    success=False
                )
            except Exception as e:
                logger.warning(f"Failed to log recognition attempt: {e}")
        
        return {
            "success": True,
            "match": None,
            "processing_time_ms": processing_time_ms
        }
        
    except FaceEngineError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/face/list/{person_type}")
async def list_enrolled_faces(person_type: str):
    """List enrolled faces by person type."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    try:
        all_faces = face_engine.list_enrolled_faces()
        
        if person_type == "all":
            filtered_faces = all_faces
        else:
            # Filter by person type
            filtered_faces = []
            for face_info in all_faces:
                face_data = face_engine.faces_db["faces"].get(face_info["name"])
                if face_data and face_data.get("person_type") == person_type:
                    face_info["person_type"] = person_type
                    face_info["person_code"] = face_data.get("person_code", face_info["name"])
                    face_info["person_name"] = face_data.get("person_name", face_info["name"])
                    filtered_faces.append(face_info)
        
        # Format response
        formatted_faces = []
        for face in filtered_faces:
            formatted_faces.append({
                "id": hash(face["name"]),  # Simple ID generation
                "person_type": face.get("person_type", "unknown"),
                "person_code": face.get("person_code", face["name"]),
                "person_name": face.get("person_name", face["name"]),
                "enrollment_date": face["enrollment_date"],
                "has_image": True
            })
        
        return {
            "success": True,
            "faces": formatted_faces,
            "total_count": len(formatted_faces)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/api/face/delete")
async def delete_face(request: FaceDeleteRequest):
    """Delete a face enrollment."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    try:
        unique_name = f"{request.person_type}_{request.person_code}"
        
        success = face_engine.delete_face(unique_name)
        
        if success:
            return {
                "success": True,
                "message": "Face enrollment deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No enrollment found for {request.person_type} {request.person_code}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/face/quality-check")
async def assess_face_quality(request: FaceQualityRequest):
    """Assess the quality of a face image."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    try:
        image = decode_base64_image(request.image_base64)
        
        # Perform basic quality assessment
        try:
            embedding, metadata = face_engine.extract_embedding(image)
            
            # Calculate quality metrics
            face_detected = True
            face_count = metadata.get("face_count", 1)
            confidence = metadata.get("confidence", 0.0)
            
            # Determine quality score based on detection confidence
            if confidence >= 0.9:
                score = 0.9
                recommendation = "suitable_for_enrollment"
                face_size = "adequate"
                brightness = "good"
                blur_level = "low"
            elif confidence >= 0.7:
                score = 0.7
                recommendation = "acceptable_with_caution"
                face_size = "adequate"
                brightness = "good"
                blur_level = "medium"
            else:
                score = 0.5
                recommendation = "not_suitable"
                face_size = "small"
                brightness = "poor"
                blur_level = "high"
            
        except FaceEngineError:
            face_detected = False
            face_count = 0
            score = 0.0
            recommendation = "not_suitable"
            face_size = "unknown"
            brightness = "unknown"
            blur_level = "unknown"
        
        return {
            "success": True,
            "quality": {
                "score": score,
                "face_detected": face_detected,
                "face_count": face_count,
                "face_size": face_size,
                "brightness": brightness,
                "blur_level": blur_level,
                "recommendation": recommendation
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/face/batch-enroll")
async def batch_enroll_faces(request: BatchEnrollmentRequest):
    """Batch enroll multiple faces."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    results = []
    successful = 0
    failed = 0
    
    for enrollment in request.enrollments:
        try:
            # Decode and validate image
            image = decode_base64_image(enrollment.image_base64)
            
            # Extract face embedding
            embedding, metadata = face_engine.extract_embedding(image)
            
            # Create unique identifier
            unique_name = f"{enrollment.person_type}_{enrollment.person_code}"
            
            # Check if already enrolled and update
            if face_engine.is_name_enrolled(unique_name):
                face_engine.delete_face(unique_name)
            
            # Enroll face
            result = face_engine.enroll_face(unique_name, image)
            
            # Store additional metadata
            face_data = face_engine.faces_db["faces"][unique_name]
            face_data["person_type"] = enrollment.person_type
            face_data["person_code"] = enrollment.person_code
            face_data["person_name"] = enrollment.person_name
            face_data["original_name"] = enrollment.original_name
            
            results.append({
                "person_code": enrollment.person_code,
                "success": True,
                "message": "Enrolled"
            })
            successful += 1
            
        except Exception as e:
            results.append({
                "person_code": enrollment.person_code,
                "success": False,
                "message": str(e)
            })
            failed += 1
    
    # Save all changes at once
    try:
        face_engine._save_database()
    except Exception as e:
        pass  # Log but don't fail the entire batch
    
    return {
        "success": True,
        "results": results,
        "summary": {
            "total": len(request.enrollments),
            "successful": successful,
            "failed": failed
        }
    }


# Legacy compatibility endpoints (for backward compatibility with old frontend)
@app.post("/enroll")
async def legacy_enroll_face(
    name: str = Form(..., description="Name of the person to enroll"),
    file: UploadFile = File(..., description="Image file containing the face to enroll")
):
    """Legacy endpoint for face enrollment."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    # Validate image file
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


@app.post("/identify")
async def legacy_identify_face(
    file: UploadFile = File(..., description="Image file to identify")
):
    """Legacy endpoint for face identification."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
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


@app.get("/faces")
async def legacy_list_faces():
    """Legacy endpoint for listing enrolled faces."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
    faces = face_engine.list_enrolled_faces()
    return {
        "total_count": len(faces),
        "faces": faces
    }


@app.delete("/faces/{name}")
async def legacy_delete_face(name: str):
    """Legacy endpoint for deleting a face."""
    if not face_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition engine not initialized"
        )
    
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "InsightFace Attendance API",
        "status": "running",
        "version": "2.0.0",
        "engine_initialized": face_engine is not None,
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


if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("storage", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the application on port 5000 for Replit compatibility
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info"
    )
