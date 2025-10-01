"""
Database service for MySQL operations.
"""

import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import base64
from io import BytesIO

# Import MySQL connector
try:
    import mysql.connector
    from mysql.connector import pooling, Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("mysql-connector-python not available - installing...")

from config import get_database_config, get_settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for MySQL operations."""
    
    def __init__(self):
        self.pool = None
        self.settings = get_settings()
    
    async def initialize(self):
        """Initialize MySQL database connection pool."""
        try:
            if not MYSQL_AVAILABLE:
                raise Exception("mysql-connector-python not available")
                
            config = get_database_config()
            config['pool_name'] = 'face_recognition_pool'
            config['pool_size'] = self.settings.DB_POOL_SIZE
            config['pool_reset_session'] = True
            
            self.pool = mysql.connector.pooling.MySQLConnectionPool(**config)
            logger.info("MySQL database connection pool initialized successfully")
            
            # Test connection
            await self.test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize MySQL database pool: {e}")
            raise
    
    async def test_connection(self):
        """Test MySQL database connection."""
        try:
            if not MYSQL_AVAILABLE:
                logger.warning("mysql-connector-python not available")
                return False
                
            connection = self.pool.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            connection.close()
            logger.info("MySQL database connection test successful")
            return True
        except Exception as e:
            logger.error(f"MySQL database connection test failed: {e}")
            return False
    
    async def close(self):
        """Close database connections."""
        if self.pool:
            # MySQL connector pool doesn't have explicit close method
            # Connections will be closed automatically
            pass
    
    def get_connection(self):
        """Get MySQL database connection from pool."""
        if not self.pool:
            raise Exception("MySQL database pool not initialized")
        return self.pool.get_connection()
    
    async def enroll_face(self, person_type: str, person_code: str, person_name: str, 
                         image_data: bytes, face_descriptor: List[float], 
                         original_name: Optional[str] = None, content_type: str = "image/jpeg") -> int:
        """
        Enroll a face in the SQL Server database.
        
        Args:
            person_type: 'student' or 'teacher'
            person_code: Unique person code
            person_name: Person's full name
            image_data: Binary image data
            face_descriptor: Face embedding as list
            original_name: Original filename
            content_type: MIME type of the image
            
        Returns:
            Database ID of the enrolled face
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # First, update or insert into Student/Teacher table
            await self._update_person_table(cursor, person_type, person_code, person_name, image_data)
            
            # Check if face already exists
            cursor.execute(
                "SELECT Id FROM FaceData WHERE PersonType = %s AND PersonCode = %s",
                (person_type, person_code)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing face
                cursor.execute(
                    """UPDATE FaceData SET 
                       ImageData = %s, 
                       FaceDescriptor = %s, 
                       OriginalName = %s,
                       ContentType = %s,
                       CreatedDate = NOW()
                       WHERE PersonType = %s AND PersonCode = %s""",
                    (image_data, json.dumps(face_descriptor), original_name, 
                     content_type, person_type, person_code)
                )
                face_id = existing[0]
            else:
                # Insert new face
                cursor.execute(
                    """INSERT INTO FaceData (PersonType, PersonCode, ImageData, 
                       FaceDescriptor, OriginalName, ContentType, CreatedDate) 
                       VALUES (%s, %s, %s, %s, %s, %s, NOW())""",
                    (person_type, person_code, image_data, json.dumps(face_descriptor), 
                     original_name, content_type)
                )
                face_id = cursor.lastrowid
            
            connection.commit()
            return int(face_id)
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error enrolling face: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def _update_person_table(self, cursor, person_type: str, person_code: str, 
                                 person_name: str, image_data: bytes):
        """Update person information in Student or Teacher table using MySQL syntax."""
        if person_type == "student":
            # Check if student exists
            cursor.execute("SELECT Id FROM Student WHERE StudentCode = %s", (person_code,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute(
                    """UPDATE Student SET StudentName = %s, StudentImage = %s 
                       WHERE StudentCode = %s""",
                    (person_name, image_data, person_code)
                )
            else:
                cursor.execute(
                    """INSERT INTO Student (StudentCode, StudentName, StudentImage, CreatedDate) 
                       VALUES (%s, %s, %s, NOW())""",
                    (person_code, person_name, image_data)
                )
        
        elif person_type == "teacher":
            # Check if teacher exists
            cursor.execute("SELECT Id FROM Teacher WHERE TeacherCode = %s", (person_code,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute(
                    """UPDATE Teacher SET TeacherName = %s, TeacherImage = %s 
                       WHERE TeacherCode = %s""",
                    (person_name, image_data, person_code)
                )
            else:
                cursor.execute(
                    """INSERT INTO Teacher (TeacherCode, TeacherName, TeacherImage, CreatedDate) 
                       VALUES (%s, %s, %s, NOW())""",
                    (person_code, person_name, image_data)
                )
    
    async def get_all_face_descriptors(self, person_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all face descriptors from SQL Server database.
        
        Args:
            person_type: Filter by person type (optional)
            
        Returns:
            List of face descriptor records
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            if person_type:
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.FaceDescriptor,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName
                       FROM FaceData fd
                       LEFT JOIN Student s ON TRIM(fd.PersonCode) = TRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON TRIM(fd.PersonCode) = TRIM(t.TeacherCode) AND fd.PersonType = 'teacher'
                       WHERE fd.PersonType = %s""",
                    (person_type,)
                )
            else:
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.FaceDescriptor,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName
                       FROM FaceData fd
                       LEFT JOIN Student s ON TRIM(fd.PersonCode) = TRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON TRIM(fd.PersonCode) = TRIM(t.TeacherCode) AND fd.PersonType = 'teacher'"""
                )
            
            results = cursor.fetchall()
            
            # Parse face descriptors from JSON
            for result in results:
                if result['FaceDescriptor']:
                    try:
                        result['FaceDescriptor'] = json.loads(result['FaceDescriptor'])
                    except:
                        result['FaceDescriptor'] = None
                    
            return results
            
        except Error as e:
            logger.error(f"Error getting face descriptors: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_face_descriptors(self, person_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Alias for get_all_face_descriptors for API compatibility.
        Get face descriptors from database, optionally filtered by person type.
        """
        return await self.get_all_face_descriptors(person_type)
    
    async def get_enrolled_faces(self, person_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of enrolled faces with metadata from MySQL."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            if person_type and person_type != "all":
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.CreatedDate,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName,
                       (fd.ImageData IS NOT NULL) as HasImage
                       FROM FaceData fd
                       LEFT JOIN Student s ON TRIM(fd.PersonCode) = TRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON TRIM(fd.PersonCode) = TRIM(t.TeacherCode) AND fd.PersonType = 'teacher'
                       WHERE fd.PersonType = %s
                       ORDER BY fd.CreatedDate DESC""",
                    (person_type,)
                )
            else:
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.CreatedDate,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName,
                       (fd.ImageData IS NOT NULL) as HasImage
                       FROM FaceData fd
                       LEFT JOIN Student s ON TRIM(fd.PersonCode) = TRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON TRIM(fd.PersonCode) = TRIM(t.TeacherCode) AND fd.PersonType = 'teacher'
                       ORDER BY fd.CreatedDate DESC"""
                )
            
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error getting enrolled faces: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def delete_face(self, person_type: str, person_code: str) -> bool:
        """Delete a face enrollment from SQL Server."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute(
                "DELETE FROM FaceData WHERE PersonType = %s AND PersonCode = %s",
                (person_type, person_code)
            )
            
            deleted_rows = cursor.rowcount
            connection.commit()
            
            return deleted_rows > 0
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error deleting face: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def log_recognition_attempt(self, person_code: Optional[str], person_type: Optional[str],
                                    confidence: float, processing_time_ms: int,
                                    source_image_name: Optional[str], success: bool):
        """Log a face recognition attempt to SQL Server."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Create the recognition logs table if it doesn't exist
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS FaceRecognitionLogs (
                        Id INT AUTO_INCREMENT PRIMARY KEY,
                        PersonCode CHAR(10),
                        PersonType VARCHAR(20),
                        Confidence FLOAT,
                        ProcessingTimeMs INT,
                        SourceImageName VARCHAR(255),
                        RecognitionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
                        Success BOOLEAN
                    )
                """)
                connection.commit()
            except:
                pass  # Table may already exist
            
            cursor.execute(
                """INSERT INTO FaceRecognitionLogs 
                   (PersonCode, PersonType, Confidence, ProcessingTimeMs, SourceImageName, RecognitionDate, Success)
                   VALUES (%s, %s, %s, %s, %s, NOW(), %s)""",
                (person_code, person_type, confidence, processing_time_ms, 
                 source_image_name, success)
            )
            
            connection.commit()
            
        except Error as e:
            logger.error(f"Error logging recognition attempt: {e}")
            # Don't raise here as logging shouldn't break the main flow
        finally:
            if connection:
                connection.close()
    
    async def get_face_count(self) -> int:
        """Get total number of enrolled faces from SQL Server."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM FaceData")
            result = cursor.fetchone()
            
            return result[0] if result else 0
            
        except Error as e:
            logger.error(f"Error getting face count: {e}")
            return 0
        finally:
            if connection:
                connection.close()
    
    async def is_connected(self) -> bool:
        """Check if database is connected and accessible."""
        try:
            if not self.pool or not MYSQL_AVAILABLE:
                return False
            return await self.test_connection()
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def base64_to_bytes(self, base64_string: str) -> bytes:
        """Convert base64 string to bytes for database storage."""
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)