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
from pydantic import BaseModel, Field, field_validator
from PIL import Image
import numpy as np
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from face_engine import FaceEngine, FaceEngineError
from services.database_service import DatabaseService
from config import get_settings
from models.user_models import (
    UserCreate, UserUpdate, UserResponse, LoginRequest, LoginResponse,
    SessionResponse, LogResponse, LogStatsResponse
)


# Pydantic models for the new API
class FaceEnrollmentRequest(BaseModel):
    person_type: str = Field(..., pattern="^(student|teacher)$")
    person_code: str = Field(..., min_length=1, max_length=10)
    person_name: str = Field(..., min_length=1, max_length=200)
    image_base64: str = Field(..., description="Base64 encoded image")
    original_name: Optional[str] = None

    @field_validator('image_base64')
    @classmethod
    def validate_base64_image(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('Image must be base64 encoded with data URI format')
        return v


class FaceRecognitionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    person_type: Optional[str] = Field(None, pattern="^(student|teacher)$")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0)
    source_image_name: Optional[str] = None

    @field_validator('image_base64')
    @classmethod
    def validate_base64_image(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('Image must be base64 encoded with data URI format')
        return v


class FaceDeleteRequest(BaseModel):
    person_type: str = Field(..., pattern="^(student|teacher)$")
    person_code: str = Field(..., min_length=1, max_length=10)


class FaceQualityRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")

    @field_validator('image_base64')
    @classmethod
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
    enrollments: List[BatchEnrollmentItem] = Field(..., min_length=1, max_length=100)


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

        # Initialize MySQL database service (required)
        db_service = DatabaseService()
        await db_service.initialize()
        print("MySQL database service initialized successfully")

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
        # Get face count from file system
        enrolled_faces = face_engine.get_face_count()

        # Try to get database face count and connection status
        database_face_count = 0
        database_connected = False

        if db_service:
            try:
                database_connected = await db_service.is_connected()
                if database_connected:
                    database_face_count = await db_service.get_face_count()
            except Exception as e:
                logger.warning(f"Database not available for health check: {e}")
                database_connected = False

        return {
            "status": "healthy",
            "message": "All systems operational",
            "enrolled_faces": enrolled_faces,
            "database_face_count": database_face_count,
            "model_loaded": True,
            "similarity_threshold": face_engine.similarity_threshold,
            "database_connected": database_connected
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": f"Engine error: {str(e)}",
                "enrolled_faces": 0,
                "database_face_count": 0,
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

        # Extract face embedding using the face engine
        embedding, metadata = face_engine.extract_embedding(image)

        # Convert image to bytes for MySQL database storage
        image_bytes = db_service.base64_to_bytes(request.image_base64)

        # Store in MySQL database
        database_id = await db_service.enroll_face(
            person_type=request.person_type,
            person_code=request.person_code,
            person_name=request.person_name,
            image_data=image_bytes,
            face_descriptor=embedding.tolist(),
            original_name=request.original_name,
            content_type="image/jpeg"
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Face enrolled in MySQL database with ID: {database_id} (took {processing_time}ms)")

        return {
            "success": True,
            "message": "Face enrolled successfully in database",
            "person_code": request.person_code,
            "person_name": request.person_name,
            "embedding_id": database_id,
            "processing_time_ms": processing_time,
            "confidence": metadata["confidence"]
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

        # Extract face embedding from query image
        query_embedding, metadata = face_engine.extract_embedding(image)

        # Get all face descriptors from MySQL database
        database_embeddings = await db_service.get_face_descriptors(request.person_type)

        # Find best match using face engine
        threshold = request.similarity_threshold or face_engine.similarity_threshold
        best_match = face_engine.find_best_match(
            query_embedding.tolist(),
            database_embeddings,
            threshold
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        if best_match:
            # Log successful recognition attempt
            await db_service.log_recognition_attempt(
                person_code=best_match["person_code"],
                person_type=best_match["person_type"],
                confidence=best_match["confidence"],
                processing_time_ms=processing_time_ms,
                source_image_name=request.source_image_name,
                success=True
            )

            return {
                "success": True,
                "match": {
                    "person_type": best_match["person_type"],
                    "person_code": best_match["person_code"],
                    "person_name": best_match["person_name"],
                    "confidence": best_match["confidence"],
                    "similarity_score": best_match["similarity_score"]
                },
                "processing_time_ms": processing_time_ms
            }

        # No match found or below threshold
        await db_service.log_recognition_attempt(
            person_code=None,
            person_type=request.person_type,
            confidence=0.0,
            processing_time_ms=processing_time_ms,
            source_image_name=request.source_image_name,
            success=False
        )

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
        # Get enrolled faces from MySQL database
        faces_from_db = await db_service.get_enrolled_faces(person_type)

        # Format response
        formatted_faces = []
        for face in faces_from_db:
            formatted_faces.append({
                "id": face["Id"],
                "person_type": face["PersonType"],
                "person_code": face["PersonCode"],
                "person_name": face["PersonName"],
                "created_date": face["CreatedDate"],
                "has_image": face["HasImage"]
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
        # Delete from MySQL database
        success = await db_service.delete_face(request.person_type, request.person_code)

        if success:
            return {
                "success": True,
                "message": f"Face enrollment deleted successfully for {request.person_type} {request.person_code}"
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

            # Convert image to bytes for MySQL database storage
            image_bytes = db_service.base64_to_bytes(enrollment.image_base64)

            # Store in MySQL database
            database_id = await db_service.enroll_face(
                person_type=enrollment.person_type,
                person_code=enrollment.person_code,
                person_name=enrollment.person_name,
                image_data=image_bytes,
                face_descriptor=embedding.tolist(),
                original_name=enrollment.original_name,
                content_type="image/jpeg"
            )

            results.append({
                "person_code": enrollment.person_code,
                "person_name": enrollment.person_name,
                "success": True,
                "message": "Enrolled successfully",
                "database_id": database_id
            })
            successful += 1

        except Exception as e:
            results.append({
                "person_code": enrollment.person_code,
                "person_name": enrollment.person_name,
                "success": False,
                "message": str(e)
            })
            failed += 1

    return {
        "success": True,
        "results": results,
        "summary": {
            "total": len(request.enrollments),
            "successful": successful,
            "failed": failed
        }
    }


# ============ USER MANAGEMENT ENDPOINTS ============

@app.post("/api/admin/login")
async def admin_login(request: LoginRequest):
    """Admin login endpoint."""
    try:
        # Get user by username
        user = await db_service.get_user_by_username(request.username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        if not user['IsActive']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Verify password
        password_valid = await db_service.verify_password(request.password, user['PasswordHash'])
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create session
        # Note: We can't access request.client.host directly here, so we'll use a placeholder
        session_token = await db_service.create_session(
            user_id=user['Id'],
            ip_address=None,  # Add middleware to capture this if needed
            user_agent=None   # Add middleware to capture this if needed
        )
        
        # Update last login
        await db_service.update_last_login(user['Id'])
        
        # Log the login
        await db_service.log_user_action(
            user_id=user['Id'],
            username=user['Username'],
            action="LOGIN",
            description=f"User {user['Username']} logged in"
        )
        
        return {
            "success": True,
            "message": "Login successful",
            "sessionToken": session_token,
            "user": {
                "Id": user['Id'],
                "Username": user['Username'],
                "FullName": user['FullName'],
                "Email": user['Email'],
                "Role": user['Role']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )


@app.get("/api/admin/users")
async def get_all_users():
    """Get all users."""
    try:
        users = await db_service.get_all_users()
        return {
            "success": True,
            "users": users
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving users: {str(e)}"
        )


@app.get("/api/admin/users/{user_id}")
async def get_user(user_id: int):
    """Get a single user by ID."""
    try:
        user = await db_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        return {
            "success": True,
            "user": user
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user: {str(e)}"
        )


@app.post("/api/admin/users")
async def create_user(request: UserCreate):
    """Create a new user."""
    try:
        user_id = await db_service.create_user(
            username=request.username,
            password=request.password,
            full_name=request.fullName,
            email=request.email,
            role=request.role,
            is_active=request.isActive
        )
        
        # Log the action
        await db_service.log_user_action(
            user_id=None,  # Could be from session if we had auth middleware
            username="system",
            action="CREATE_USER",
            table_name="users",
            record_id=user_id,
            new_value=json.dumps({"username": request.username, "role": request.role}),
            description=f"Created new user: {request.username}"
        )
        
        return {
            "success": True,
            "message": "User created successfully",
            "userId": user_id
        }
        
    except Exception as e:
        error_message = str(e)
        if "already exists" in error_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {error_message}"
        )


@app.put("/api/admin/users/{user_id}")
async def update_user(user_id: int, request: UserUpdate):
    """Update a user."""
    try:
        # Get old user data for logging
        old_user = await db_service.get_user_by_id(user_id)
        
        if not old_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        success = await db_service.update_user(
            user_id=user_id,
            username=request.username,
            password=request.password,
            full_name=request.fullName,
            email=request.email,
            role=request.role,
            is_active=request.isActive
        )
        
        if success:
            # Log the action
            await db_service.log_user_action(
                user_id=None,
                username="system",
                action="UPDATE_USER",
                table_name="users",
                record_id=user_id,
                old_value=json.dumps({"username": old_user['Username'], "role": old_user['Role']}),
                new_value=json.dumps(request.dict(exclude_none=True)),
                description=f"Updated user: {old_user['Username']}"
            )
            
            return {
                "success": True,
                "message": "User updated successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e)
        if "already exists" in error_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user: {error_message}"
        )


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user."""
    try:
        # Get user data for logging
        user = await db_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        success = await db_service.delete_user(user_id)
        
        if success:
            # Log the action
            await db_service.log_user_action(
                user_id=None,
                username="system",
                action="DELETE_USER",
                table_name="users",
                record_id=user_id,
                old_value=json.dumps({"username": user['Username'], "role": user['Role']}),
                description=f"Deleted user: {user['Username']}"
            )
            
            return {
                "success": True,
                "message": "User deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )


# ============ USER SESSION ENDPOINTS ============

@app.get("/api/admin/user-sessions")
async def get_all_sessions(limit: int = 100):
    """Get all user sessions."""
    try:
        sessions = await db_service.get_all_sessions(limit)
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sessions: {str(e)}"
        )


@app.get("/api/admin/user-sessions/active")
async def get_active_sessions():
    """Get active user sessions."""
    try:
        sessions = await db_service.get_active_sessions()
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving active sessions: {str(e)}"
        )


@app.post("/api/admin/user-sessions/{session_id}/terminate")
async def terminate_session(session_id: int):
    """Terminate a user session."""
    try:
        success = await db_service.terminate_session(session_id)
        
        if success:
            # Log the action
            await db_service.log_user_action(
                user_id=None,
                username="system",
                action="TERMINATE_SESSION",
                table_name="user_sessions",
                record_id=session_id,
                description=f"Terminated session ID: {session_id}"
            )
            
            return {
                "success": True,
                "message": "Session terminated successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session with ID {session_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error terminating session: {str(e)}"
        )


# ============ AUDIT LOG ENDPOINTS ============

@app.get("/api/admin/logs")
async def get_logs(
    userId: Optional[int] = None,
    action: Optional[str] = None,
    tableName: Optional[str] = None,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    limit: int = 100
):
    """Get user logs with optional filtering."""
    try:
        logs = await db_service.get_logs(
            user_id=userId,
            action=action,
            table_name=tableName,
            start_date=startDate,
            end_date=endDate,
            limit=limit
        )
        
        return {
            "success": True,
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving logs: {str(e)}"
        )


@app.get("/api/admin/logs/stats")
async def get_log_stats():
    """Get log statistics."""
    try:
        stats = await db_service.get_log_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving log stats: {str(e)}"
        )


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
        # Extract face embedding
        embedding, metadata = face_engine.extract_embedding(image)

        # Convert image to bytes
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()

        # Store in MySQL database (legacy endpoint treats name as person_code)
        database_id = await db_service.enroll_face(
            person_type="student",  # Default to student for legacy compatibility
            person_code=name,
            person_name=name,
            image_data=image_bytes,
            face_descriptor=embedding.tolist(),
            original_name=file.filename,
            content_type="image/jpeg"
        )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": f"Successfully enrolled face for '{name}'",
                "face_id": database_id,
                "name": name,
                "bounding_box": metadata["bounding_box"],
                "confidence": metadata["confidence"]
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

    # Check if running locally or on Replit
    is_replit = os.getenv('REPLIT_DB_URL') is not None
    
    print(f"Starting Face Recognition API...")
    print(f"Environment: {'Replit' if is_replit else 'Local'}")
    print(f"Database Host: {get_settings().DB_HOST}")
    print(f"Server will be available at: http://0.0.0.0:5000")

    # Run the application on port 5000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5100,
        reload=False,
        log_level="info"
    )