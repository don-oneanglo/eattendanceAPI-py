"""
Face recognition service using InsightFace.
"""

import base64
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from io import BytesIO
import cv2
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import logging

from config import get_settings

logger = logging.getLogger(__name__)


class FaceService:
    """Face recognition service using InsightFace."""
    
    def __init__(self):
        self.settings = get_settings()
        self.app = None
        self.similarity_threshold = self.settings.SIMILARITY_THRESHOLD
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize InsightFace model."""
        try:
            logger.info(f"Initializing InsightFace model: {self.settings.INSIGHTFACE_MODEL}")
            
            self.app = FaceAnalysis(
                name=self.settings.INSIGHTFACE_MODEL,
                root=self.settings.MODELS_DIR,
                providers=['CPUExecutionProvider']
            )
            
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            raise
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image/'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return cv2_image
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError(f"Invalid image format: {str(e)}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of face detection results
        """
        try:
            if self.app is None:
                raise RuntimeError("Face analysis app is not initialized")
            
            faces = self.app.get(image)
            
            if not faces:
                raise ValueError("No faces detected in the image")
            
            # Process detected faces
            face_results = []
            for face in faces:
                if face.det_score < self.settings.FACE_CONFIDENCE_THRESHOLD:
                    continue
                
                result = {
                    "embedding": face.normed_embedding.tolist(),
                    "bbox": face.bbox.tolist(),
                    "det_score": float(face.det_score),
                    "landmark": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None,
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": int(face.gender) if hasattr(face, 'gender') else None
                }
                face_results.append(result)
            
            if not face_results:
                raise ValueError("No faces detected with sufficient confidence")
            
            # Sort by detection score and return the best face
            face_results.sort(key=lambda x: x["det_score"], reverse=True)
            return face_results
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Face detection failed: {e}")
            raise RuntimeError(f"Face detection failed: {str(e)}")
    
    def extract_embedding(self, base64_image: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Extract face embedding from base64 image.
        
        Args:
            base64_image: Base64 encoded image
            
        Returns:
            Tuple of (embedding, face_metadata)
        """
        # Decode image
        cv2_image = self.decode_base64_image(base64_image)
        
        # Detect faces
        faces = self.detect_faces(cv2_image)
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the largest/most confident face")
        
        # Use the best face (already sorted by confidence)
        best_face = faces[0]
        
        embedding = best_face["embedding"]
        metadata = {
            "bounding_box": best_face["bbox"],
            "confidence": best_face["det_score"],
            "landmark": best_face["landmark"],
            "age": best_face["age"],
            "gender": best_face["gender"],
            "face_count": len(faces)
        }
        
        return embedding, metadata
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Embeddings from InsightFace are already normalized
        return float(np.dot(emb1, emb2))
    
    def find_best_match(self, query_embedding: List[float], 
                       database_embeddings: List[Dict[str, Any]], 
                       threshold: float = None) -> Optional[Dict[str, Any]]:
        """
        Find the best matching face from database.
        
        Args:
            query_embedding: Query face embedding
            database_embeddings: List of database face records
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Best match information or None if no match
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        best_match = None
        best_score = 0.0
        
        for db_face in database_embeddings:
            if not db_face['FaceDescriptor']:
                continue
            
            similarity = self.calculate_similarity(query_embedding, db_face['FaceDescriptor'])
            
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
    
    def assess_image_quality(self, base64_image: str) -> Dict[str, Any]:
        """
        Assess image quality for face recognition.
        
        Args:
            base64_image: Base64 encoded image
            
        Returns:
            Quality assessment results
        """
        try:
            cv2_image = self.decode_base64_image(base64_image)
            
            # Basic image properties
            height, width = cv2_image.shape[:2]
            
            # Try to detect faces
            faces = []
            try:
                faces = self.detect_faces(cv2_image)
            except (ValueError, RuntimeError):
                pass
            
            face_detected = len(faces) > 0
            face_count = len(faces)
            
            # Calculate quality metrics
            quality_score = 0.0
            face_size = "unknown"
            brightness = "unknown"
            blur_level = "unknown"
            
            if face_detected:
                best_face = faces[0]
                bbox = best_face["bbox"]
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_area = face_width * face_height
                
                # Face size assessment
                if face_area > 10000:
                    face_size = "large"
                    quality_score += 0.3
                elif face_area > 5000:
                    face_size = "adequate"
                    quality_score += 0.2
                else:
                    face_size = "small"
                    quality_score += 0.1
                
                # Detection confidence as quality indicator
                quality_score += min(best_face["det_score"], 1.0) * 0.5
                
                # Brightness assessment (simplified)
                face_region = cv2_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                if face_region.size > 0:
                    mean_brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
                    if 50 <= mean_brightness <= 200:
                        brightness = "good"
                        quality_score += 0.2
                    elif mean_brightness < 50:
                        brightness = "too_dark"
                    else:
                        brightness = "too_bright"
                
                # Blur assessment (simplified using Laplacian variance)
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var > 500:
                    blur_level = "low"
                    quality_score += 0.1
                elif laplacian_var > 100:
                    blur_level = "medium"
                else:
                    blur_level = "high"
            
            # Determine recommendation
            if quality_score >= 0.7 and face_count == 1:
                recommendation = "suitable_for_enrollment"
            elif quality_score >= 0.5:
                recommendation = "acceptable_with_caution"
            else:
                recommendation = "not_suitable"
            
            return {
                "score": round(quality_score, 2),
                "face_detected": face_detected,
                "face_count": face_count,
                "face_size": face_size,
                "brightness": brightness,
                "blur_level": blur_level,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {
                "score": 0.0,
                "face_detected": False,
                "face_count": 0,
                "face_size": "unknown",
                "brightness": "unknown",
                "blur_level": "unknown",
                "recommendation": "assessment_failed"
            }