"""
Face recognition engine using InsightFace buffalo_l model.
Handles face detection, embedding extraction, and similarity matching.
MySQL database integration only - no file-based storage.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import insightface
from insightface.app import FaceAnalysis
import cv2
from PIL import Image


class FaceEngineError(Exception):
    """Custom exception for face engine errors."""
    pass


class FaceEngine:
    """Face recognition engine using InsightFace with MySQL database integration only."""
    
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
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize face analysis app
        self.app = None
        self._initialize_model()
    
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
    
    def extract_face_data(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract face data from image for MySQL database storage.
        
        Args:
            image: PIL Image containing the face
            
        Returns:
            Face data including embedding and metadata
        """
        # Extract embedding and metadata
        embedding, metadata = self.extract_embedding(image)
        
        return {
            "embedding": embedding.tolist(),
            "metadata": metadata,
            "extraction_date": datetime.now().isoformat(),
            "image_size": image.size
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
    
    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compare two face embeddings and return similarity score.
        
        Args:
            embedding1: First face embedding as list
            embedding2: Second face embedding as list
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        return self._cosine_similarity(emb1, emb2)
    
    def find_best_match(self, query_embedding: List[float], database_embeddings: List[Dict[str, Any]], 
                       threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Find the best matching face from database embeddings.
        
        Args:
            query_embedding: Face embedding to search for
            database_embeddings: List of database face records with embeddings
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Best match information or None if no match
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        best_match = None
        best_score = 0.0
        
        query_emb = np.array(query_embedding)
        
        for db_face in database_embeddings:
            if not db_face.get('FaceDescriptor'):
                continue
            
            db_embedding = np.array(db_face['FaceDescriptor'])
            similarity = self._cosine_similarity(query_emb, db_embedding)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = {
                    "person_type": db_face['PersonType'],
                    "person_code": db_face['PersonCode'],
                    "person_name": db_face['PersonName'],
                    "similarity_score": similarity,
                    "confidence": similarity,  # For compatibility
                    "database_id": db_face['Id']
                }
        
        return best_match
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "models_directory": self.models_dir,
            "is_initialized": self.app is not None,
            "providers": ['CPUExecutionProvider']
        }