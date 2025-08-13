"""
Face recognition engine using InsightFace buffalo_l model.
Handles face detection, embedding extraction, and similarity matching.
"""

import os
import json
import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import shutil
from urllib.request import urlretrieve
import gzip

import insightface
from insightface.app import FaceAnalysis
import cv2
from PIL import Image
import onnxruntime


class FaceEngineError(Exception):
    """Custom exception for face engine errors."""
    pass


class FaceEngine:
    """Face recognition engine using InsightFace."""
    
    def __init__(self, similarity_threshold: float = 0.6, model_name: str = "buffalo_l"):
        """
        Initialize face recognition engine.
        
        Args:
            similarity_threshold: Cosine similarity threshold for face matching
            model_name: InsightFace model name to use
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.models_dir = "models"
        self.storage_dir = "storage"
        self.db_file = os.path.join(self.storage_dir, "faces_db.json")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize face analysis app
        self.app = None
        self._initialize_model()
        
        # Load existing database
        self.faces_db = self._load_database()
    
    def _initialize_model(self):
        """Initialize InsightFace model with automatic download."""
        try:
            print(f"Initializing InsightFace model: {self.model_name}")
            
            # Initialize face analysis with the specified model
            self.app = FaceAnalysis(
                name=self.model_name,
                root=self.models_dir,
                providers=['CPUExecutionProvider']  # Use CPU for broader compatibility
            )
            
            # Prepare the model - this will download if not present
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            print(f"Successfully initialized {self.model_name} model")
            
        except Exception as e:
            raise FaceEngineError(f"Failed to initialize face recognition model: {str(e)}")
    
    def _load_database(self) -> Dict[str, Any]:
        """Load faces database from JSON file."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load database file, starting fresh: {e}")
                return {"faces": {}, "metadata": {"created": datetime.now().isoformat()}}
        else:
            # Create initial database structure
            return {"faces": {}, "metadata": {"created": datetime.now().isoformat()}}
    
    def _save_database(self):
        """Save faces database to JSON file."""
        try:
            self.faces_db["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.db_file, 'w') as f:
                json.dump(self.faces_db, f, indent=2, default=str)
        except IOError as e:
            raise FaceEngineError(f"Failed to save database: {str(e)}")
    
    def _image_to_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def _detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect faces in image using InsightFace.
        
        Args:
            image: PIL Image
            
        Returns:
            List of face detection results with embeddings and bounding boxes
        """
        try:
            if self.app is None:
                raise FaceEngineError("Face analysis app is not initialized")
            cv2_image = self._image_to_cv2(image)
            faces = self.app.get(cv2_image)
            
            if not faces:
                raise FaceEngineError("No faces detected in the image")
            
            if len(faces) > 1:
                raise FaceEngineError(
                    f"Multiple faces detected ({len(faces)}). Please provide an image with exactly one face."
                )
            
            # Extract face information
            face = faces[0]
            result = {
                "embedding": face.normed_embedding.tolist(),  # 512-d normalized embedding
                "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                "det_score": float(face.det_score),  # Detection confidence
                "landmark": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None,
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": int(face.gender) if hasattr(face, 'gender') else None
            }
            
            return [result]
            
        except Exception as e:
            if isinstance(e, FaceEngineError):
                raise
            raise FaceEngineError(f"Face detection failed: {str(e)}")
    
    def extract_embedding(self, image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract face embedding from image.
        
        Args:
            image: PIL Image containing a face
            
        Returns:
            Tuple of (embedding array, face metadata)
        """
        faces = self._detect_faces(image)
        face_data = faces[0]  # We already validated there's exactly one face
        
        embedding = np.array(face_data["embedding"])
        metadata = {
            "bounding_box": face_data["bbox"],
            "confidence": face_data["det_score"],
            "landmark": face_data["landmark"],
            "age": face_data["age"],
            "gender": face_data["gender"]
        }
        
        return embedding, metadata
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Embeddings from InsightFace are already normalized
        return float(np.dot(embedding1, embedding2))
    
    def enroll_face(self, name: str, image: Image.Image) -> Dict[str, Any]:
        """
        Enroll a new face in the database.
        
        Args:
            name: Unique identifier for the person
            image: PIL Image containing the face
            
        Returns:
            Enrollment result with face ID and metadata
        """
        if self.is_name_enrolled(name):
            raise FaceEngineError(f"Person with name '{name}' is already enrolled")
        
        # Extract embedding and metadata
        embedding, metadata = self.extract_embedding(image)
        
        # Generate unique face ID
        face_id = str(uuid.uuid4())
        
        # Save to database
        face_data = {
            "face_id": face_id,
            "name": name,
            "embedding": embedding.tolist(),
            "metadata": metadata,
            "enrollment_date": datetime.now().isoformat(),
            "image_size": image.size
        }
        
        self.faces_db["faces"][name] = face_data
        self._save_database()
        
        # Save image file
        image_path = os.path.join(self.storage_dir, f"{face_id}.jpg")
        image.save(image_path, "JPEG", quality=95)
        
        return {
            "face_id": face_id,
            "name": name,
            "bounding_box": metadata["bounding_box"],
            "confidence": metadata["confidence"]
        }
    
    def verify_faces(self, image1: Image.Image, image2: Image.Image) -> Dict[str, Any]:
        """
        Verify if two images contain the same person.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Verification result with similarity score and match status
        """
        # Extract embeddings from both images
        embedding1, metadata1 = self.extract_embedding(image1)
        embedding2, metadata2 = self.extract_embedding(image2)
        
        # Calculate similarity
        similarity = self._cosine_similarity(embedding1, embedding2)
        is_match = similarity >= self.similarity_threshold
        
        return {
            "is_match": is_match,
            "similarity_score": float(similarity),
            "threshold": self.similarity_threshold,
            "face1_bbox": metadata1["bounding_box"],
            "face2_bbox": metadata2["bounding_box"],
            "confidence1": metadata1["confidence"],
            "confidence2": metadata2["confidence"]
        }
    
    def identify_face(self, image: Image.Image) -> Dict[str, Any]:
        """
        Identify a face by searching the enrolled faces database.
        
        Args:
            image: PIL Image containing the face to identify
            
        Returns:
            Identification result with top 3 matches
        """
        if not self.faces_db["faces"]:
            return {
                "matches": [],
                "bounding_box": None,
                "confidence": None,
                "total_enrolled": 0
            }
        
        # Extract embedding from query image
        query_embedding, metadata = self.extract_embedding(image)
        
        # Calculate similarities with all enrolled faces
        similarities = []
        for name, face_data in self.faces_db["faces"].items():
            enrolled_embedding = np.array(face_data["embedding"])
            similarity = self._cosine_similarity(query_embedding, enrolled_embedding)
            
            similarities.append({
                "name": name,
                "similarity_score": float(similarity),
                "is_match": similarity >= self.similarity_threshold,
                "face_id": face_data["face_id"],
                "enrollment_date": face_data["enrollment_date"]
            })
        
        # Sort by similarity score (descending) and get top 3
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_matches = similarities[:3]
        
        return {
            "matches": top_matches,
            "bounding_box": metadata["bounding_box"],
            "confidence": metadata["confidence"],
            "total_enrolled": len(self.faces_db["faces"])
        }
    
    def list_enrolled_faces(self) -> List[Dict[str, Any]]:
        """
        Get list of all enrolled faces.
        
        Returns:
            List of enrolled faces with metadata
        """
        faces_list = []
        for name, face_data in self.faces_db["faces"].items():
            face_info = {
                "name": name,
                "face_id": face_data["face_id"],
                "enrollment_date": face_data["enrollment_date"],
                "bounding_box": face_data["metadata"]["bounding_box"],
                "confidence": face_data["metadata"]["confidence"],
                "image_size": face_data.get("image_size"),
                "age": face_data["metadata"].get("age"),
                "gender": face_data["metadata"].get("gender")
            }
            faces_list.append(face_info)
        
        # Sort by enrollment date (newest first)
        faces_list.sort(key=lambda x: x["enrollment_date"], reverse=True)
        return faces_list
    
    def delete_face(self, name: str) -> bool:
        """
        Delete an enrolled face from the database.
        
        Args:
            name: Name of the person to delete
            
        Returns:
            True if deletion was successful, False if name not found
        """
        if name not in self.faces_db["faces"]:
            return False
        
        # Get face data before deletion
        face_data = self.faces_db["faces"][name]
        face_id = face_data["face_id"]
        
        # Remove from database
        del self.faces_db["faces"][name]
        self._save_database()
        
        # Remove image file if it exists
        image_path = os.path.join(self.storage_dir, f"{face_id}.jpg")
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                print(f"Warning: Could not remove image file {image_path}")
        
        return True
    
    def is_name_enrolled(self, name: str) -> bool:
        """Check if a name is already enrolled."""
        return name in self.faces_db["faces"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_faces = len(self.faces_db["faces"])
        
        ages = [face_data["metadata"].get("age") for face_data in self.faces_db["faces"].values() 
                if face_data["metadata"].get("age") is not None]
        
        genders = [face_data["metadata"].get("gender") for face_data in self.faces_db["faces"].values() 
                   if face_data["metadata"].get("gender") is not None]
        
        return {
            "total_enrolled": total_faces,
            "average_age": sum(ages) / len(ages) if ages else None,
            "gender_distribution": {
                "male": sum(1 for g in genders if g == 1),
                "female": sum(1 for g in genders if g == 0)
            } if genders else None,
            "similarity_threshold": self.similarity_threshold,
            "model_name": self.model_name
        }
