"""
Database service for SQL Server operations.
"""

import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import base64
from io import BytesIO

# Note: pyodbc is not available in this environment
# When deploying to a production environment with SQL Server access, 
# uncomment the following line and install pyodbc:
# import pyodbc

from config import get_database_config, get_settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for SQL Server operations."""
    
    def __init__(self):
        self.connection_string = None
        self.settings = get_settings()
    
    async def initialize(self):
        """Initialize database connection."""
        try:
            config = get_database_config()
            
            # Build SQL Server connection string
            self.connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={config['host']};"
                f"DATABASE={config['database']};"
                f"UID={config['user']};"
                f"PWD={config['password']};"
                f"TrustServerCertificate=yes;"
                f"Connection Timeout=30;"
            )
            
            logger.info("Database connection string configured")
            
            # Test connection
            await self.test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    async def test_connection(self):
        """Test database connection."""
        try:
            # Note: This is a placeholder since pyodbc is not available
            # When deploying with proper SQL Server drivers, this will work
            logger.warning("pyodbc not available - database connection test skipped")
            logger.info(f"Database would connect to: {self.connection_string}")
            return True
            
            # Uncomment when pyodbc is available:
            # connection = pyodbc.connect(self.connection_string)
            # cursor = connection.cursor()
            # cursor.execute("SELECT 1")
            # cursor.fetchone()
            # cursor.close()
            # connection.close()
            # logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        # No persistent connections to close
        pass
    
    def get_connection(self):
        """Get database connection."""
        if not self.connection_string:
            raise Exception("Database connection not initialized")
        
        # Note: pyodbc is not available in this environment
        # This will work when deployed with proper SQL Server drivers
        raise Exception("pyodbc not available - install SQL Server drivers for database functionality")
        
        # Uncomment when pyodbc is available:
        # return pyodbc.connect(self.connection_string)
    
    async def enroll_face(self, person_type: str, person_code: str, person_name: str, 
                         image_data: bytes, face_descriptor: List[float], 
                         original_name: str = None, content_type: str = "image/jpeg") -> int:
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
                "SELECT Id FROM FaceData WHERE PersonType = ? AND PersonCode = ?",
                (person_type, person_code)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing face
                cursor.execute(
                    """UPDATE FaceData SET 
                       ImageData = ?, 
                       FaceDescriptor = ?, 
                       OriginalName = ?,
                       ContentType = ?,
                       CreatedDate = GETDATE()
                       WHERE PersonType = ? AND PersonCode = ?""",
                    (image_data, json.dumps(face_descriptor), original_name, 
                     content_type, person_type, person_code)
                )
                face_id = existing[0]
            else:
                # Insert new face
                cursor.execute(
                    """INSERT INTO FaceData (PersonType, PersonCode, ImageData, 
                       FaceDescriptor, OriginalName, ContentType, CreatedDate) 
                       VALUES (?, ?, ?, ?, ?, ?, GETDATE())""",
                    (person_type, person_code, image_data, json.dumps(face_descriptor), 
                     original_name, content_type)
                )
                cursor.execute("SELECT SCOPE_IDENTITY()")
                face_id = cursor.fetchone()[0]
            
            connection.commit()
            return int(face_id)
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Error enrolling face: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def _update_person_table(self, cursor, person_type: str, person_code: str, 
                                 person_name: str, image_data: bytes):
        """Update person information in Student or Teacher table using SQL Server syntax."""
        if person_type == "student":
            # Check if student exists
            cursor.execute("SELECT Id FROM Student WHERE StudentCode = ?", (person_code,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute(
                    """UPDATE Student SET StudentName = ?, StudentImage = ? 
                       WHERE StudentCode = ?""",
                    (person_name, image_data, person_code)
                )
            else:
                cursor.execute(
                    """INSERT INTO Student (StudentCode, StudentName, StudentImage, CreatedDate) 
                       VALUES (?, ?, ?, GETDATE())""",
                    (person_code, person_name, image_data)
                )
        
        elif person_type == "teacher":
            # Check if teacher exists
            cursor.execute("SELECT Id FROM Teacher WHERE TeacherCode = ?", (person_code,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute(
                    """UPDATE Teacher SET TeacherName = ?, TeacherImage = ? 
                       WHERE TeacherCode = ?""",
                    (person_name, image_data, person_code)
                )
            else:
                cursor.execute(
                    """INSERT INTO Teacher (TeacherCode, TeacherName, TeacherImage, CreatedDate) 
                       VALUES (?, ?, ?, GETDATE())""",
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
            cursor = connection.cursor()
            
            if person_type:
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.FaceDescriptor,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName
                       FROM FaceData fd
                       LEFT JOIN Student s ON RTRIM(fd.PersonCode) = RTRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON RTRIM(fd.PersonCode) = RTRIM(t.TeacherCode) AND fd.PersonType = 'teacher'
                       WHERE fd.PersonType = ?""",
                    (person_type,)
                )
            else:
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.FaceDescriptor,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName
                       FROM FaceData fd
                       LEFT JOIN Student s ON RTRIM(fd.PersonCode) = RTRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON RTRIM(fd.PersonCode) = RTRIM(t.TeacherCode) AND fd.PersonType = 'teacher'"""
                )
            
            # Fetch results and convert to list of dictionaries
            columns = [column[0] for column in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse face descriptors from JSON
                if result['FaceDescriptor']:
                    try:
                        result['FaceDescriptor'] = json.loads(result['FaceDescriptor'])
                    except:
                        result['FaceDescriptor'] = None
                results.append(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error getting face descriptors: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_enrolled_faces(self, person_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of enrolled faces with metadata from SQL Server."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            if person_type and person_type != "all":
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.CreatedDate,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName,
                       CASE WHEN fd.ImageData IS NOT NULL THEN 1 ELSE 0 END as HasImage
                       FROM FaceData fd
                       LEFT JOIN Student s ON RTRIM(fd.PersonCode) = RTRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON RTRIM(fd.PersonCode) = RTRIM(t.TeacherCode) AND fd.PersonType = 'teacher'
                       WHERE fd.PersonType = ?
                       ORDER BY fd.CreatedDate DESC""",
                    (person_type,)
                )
            else:
                cursor.execute(
                    """SELECT fd.Id, fd.PersonType, fd.PersonCode, fd.CreatedDate,
                       COALESCE(s.StudentName, t.TeacherName) as PersonName,
                       CASE WHEN fd.ImageData IS NOT NULL THEN 1 ELSE 0 END as HasImage
                       FROM FaceData fd
                       LEFT JOIN Student s ON RTRIM(fd.PersonCode) = RTRIM(s.StudentCode) AND fd.PersonType = 'student'
                       LEFT JOIN Teacher t ON RTRIM(fd.PersonCode) = RTRIM(t.TeacherCode) AND fd.PersonType = 'teacher'
                       ORDER BY fd.CreatedDate DESC"""
                )
            
            # Convert results to list of dictionaries
            columns = [column[0] for column in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Convert HasImage to boolean
                result['HasImage'] = bool(result['HasImage'])
                results.append(result)
            
            return results
            
        except Exception as e:
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
                "DELETE FROM FaceData WHERE PersonType = ? AND PersonCode = ?",
                (person_type, person_code)
            )
            
            deleted_rows = cursor.rowcount
            connection.commit()
            
            return deleted_rows > 0
            
        except Exception as e:
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
                    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='FaceRecognitionLogs' AND xtype='U')
                    CREATE TABLE FaceRecognitionLogs (
                        Id INT IDENTITY(1,1) PRIMARY KEY,
                        PersonCode NCHAR(10),
                        PersonType VARCHAR(20),
                        Confidence FLOAT,
                        ProcessingTimeMs INT,
                        SourceImageName NVARCHAR(255),
                        RecognitionDate DATETIME DEFAULT GETDATE(),
                        Success BIT
                    )
                """)
                connection.commit()
            except:
                pass  # Table may already exist
            
            cursor.execute(
                """INSERT INTO FaceRecognitionLogs 
                   (PersonCode, PersonType, Confidence, ProcessingTimeMs, SourceImageName, RecognitionDate, Success)
                   VALUES (?, ?, ?, ?, ?, GETDATE(), ?)""",
                (person_code, person_type, confidence, processing_time_ms, 
                 source_image_name, success)
            )
            
            connection.commit()
            
        except Exception as e:
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
            
        except Exception as e:
            logger.error(f"Error getting face count: {e}")
            return 0
        finally:
            if connection:
                connection.close()
    
    def base64_to_bytes(self, base64_string: str) -> bytes:
        """Convert base64 string to bytes for database storage."""
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)