"""
Database service for MySQL operations.
"""

import json
import asyncio
import secrets
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import base64
from io import BytesIO
import bcrypt

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
    
    # ============ USER MANAGEMENT METHODS ============
    
    async def create_user(self, username: str, password: str, full_name: str, 
                         email: Optional[str], role: str, is_active: bool = True) -> int:
        """Create a new user with hashed password."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT Id FROM users WHERE Username = %s", (username,))
            if cursor.fetchone():
                raise Exception(f"Username '{username}' already exists")
            
            # Hash password with bcrypt
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Insert new user
            cursor.execute(
                """INSERT INTO users (Username, PasswordHash, FullName, Email, Role, IsActive)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (username, password_hash, full_name, email, role, is_active)
            )
            
            user_id = cursor.lastrowid
            connection.commit()
            
            return int(user_id)
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error creating user: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users (excluding password hashes)."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute(
                """SELECT Id, Username, FullName, Email, Role, IsActive, 
                   LastLoginDate, CreatedDate, UpdatedDate 
                   FROM users ORDER BY CreatedDate DESC"""
            )
            
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error getting users: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get a single user by ID (excluding password hash)."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute(
                """SELECT Id, Username, FullName, Email, Role, IsActive,
                   LastLoginDate, CreatedDate, UpdatedDate
                   FROM users WHERE Id = %s""",
                (user_id,)
            )
            
            return cursor.fetchone()
            
        except Error as e:
            logger.error(f"Error getting user by ID: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username including password hash (for authentication)."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute(
                """SELECT Id, Username, PasswordHash, FullName, Email, Role, IsActive,
                   LastLoginDate, CreatedDate, UpdatedDate
                   FROM users WHERE Username = %s""",
                (username,)
            )
            
            return cursor.fetchone()
            
        except Error as e:
            logger.error(f"Error getting user by username: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def update_user(self, user_id: int, username: Optional[str] = None,
                         password: Optional[str] = None, full_name: Optional[str] = None,
                         email: Optional[str] = None, role: Optional[str] = None,
                         is_active: Optional[bool] = None) -> bool:
        """Update user information."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Build dynamic update query
            updates = []
            params = []
            
            if username is not None:
                # Check if new username already exists
                cursor.execute("SELECT Id FROM users WHERE Username = %s AND Id != %s", 
                             (username, user_id))
                if cursor.fetchone():
                    raise Exception(f"Username '{username}' already exists")
                updates.append("Username = %s")
                params.append(username)
            
            if password is not None:
                # Hash password with bcrypt
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                updates.append("PasswordHash = %s")
                params.append(password_hash)
            
            if full_name is not None:
                updates.append("FullName = %s")
                params.append(full_name)
            
            if email is not None:
                updates.append("Email = %s")
                params.append(email)
            
            if role is not None:
                updates.append("Role = %s")
                params.append(role)
            
            if is_active is not None:
                updates.append("IsActive = %s")
                params.append(is_active)
            
            if not updates:
                return True  # Nothing to update
            
            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE Id = %s"
            
            cursor.execute(query, params)
            connection.commit()
            
            return cursor.rowcount > 0
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error updating user: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("DELETE FROM users WHERE Id = %s", (user_id,))
            deleted = cursor.rowcount > 0
            connection.commit()
            
            return deleted
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error deleting user: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    async def update_last_login(self, user_id: int) -> None:
        """Update user's last login date."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute(
                "UPDATE users SET LastLoginDate = NOW() WHERE Id = %s",
                (user_id,)
            )
            connection.commit()
            
        except Error as e:
            logger.error(f"Error updating last login: {e}")
        finally:
            if connection:
                connection.close()
    
    # ============ USER SESSION METHODS ============
    
    async def create_session(self, user_id: int, ip_address: Optional[str] = None,
                            user_agent: Optional[str] = None) -> str:
        """Create a new user session and return session token."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Generate unique session token
            session_token = secrets.token_urlsafe(32)
            
            cursor.execute(
                """INSERT INTO user_sessions (UserId, SessionToken, IpAddress, UserAgent, IsActive)
                   VALUES (%s, %s, %s, %s, TRUE)""",
                (user_id, session_token, ip_address, user_agent)
            )
            
            connection.commit()
            return session_token
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error creating session: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_all_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all user sessions with user info."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute(
                """SELECT s.*, u.Username, u.FullName, u.Role
                   FROM user_sessions s
                   LEFT JOIN users u ON s.UserId = u.Id
                   ORDER BY s.LoginTime DESC
                   LIMIT %s""",
                (limit,)
            )
            
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error getting sessions: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get only active user sessions."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute(
                """SELECT s.*, u.Username, u.FullName, u.Role
                   FROM user_sessions s
                   LEFT JOIN users u ON s.UserId = u.Id
                   WHERE s.IsActive = TRUE
                   ORDER BY s.LoginTime DESC"""
            )
            
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error getting active sessions: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def terminate_session(self, session_id: int) -> bool:
        """Terminate a user session."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute(
                """UPDATE user_sessions 
                   SET IsActive = FALSE, LogoutTime = NOW()
                   WHERE Id = %s""",
                (session_id,)
            )
            
            terminated = cursor.rowcount > 0
            connection.commit()
            
            return terminated
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error terminating session: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user info if valid."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute(
                """SELECT s.*, u.Id as UserId, u.Username, u.FullName, u.Role, u.IsActive as UserIsActive
                   FROM user_sessions s
                   JOIN users u ON s.UserId = u.Id
                   WHERE s.SessionToken = %s AND s.IsActive = TRUE AND u.IsActive = TRUE""",
                (session_token,)
            )
            
            return cursor.fetchone()
            
        except Error as e:
            logger.error(f"Error validating session: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    # ============ AUDIT LOG METHODS ============
    
    async def log_user_action(self, user_id: Optional[int], username: Optional[str],
                              action: str, table_name: Optional[str] = None,
                              record_id: Optional[int] = None, old_value: Optional[str] = None,
                              new_value: Optional[str] = None, description: Optional[str] = None,
                              ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> int:
        """Log a user action for audit trail."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            cursor.execute(
                """INSERT INTO logs (UserId, Username, Action, TableName, RecordId,
                   OldValue, NewValue, Description, IpAddress, UserAgent)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (user_id, username, action, table_name, record_id, old_value,
                 new_value, description, ip_address, user_agent)
            )
            
            log_id = cursor.lastrowid
            connection.commit()
            
            return int(log_id)
            
        except Error as e:
            if connection:
                connection.rollback()
            logger.error(f"Error logging user action: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_logs(self, user_id: Optional[int] = None, action: Optional[str] = None,
                      table_name: Optional[str] = None, start_date: Optional[str] = None,
                      end_date: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get user logs with optional filtering."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Build dynamic query
            where_clauses = []
            params = []
            
            if user_id is not None:
                where_clauses.append("UserId = %s")
                params.append(user_id)
            
            if action is not None:
                where_clauses.append("Action = %s")
                params.append(action)
            
            if table_name is not None:
                where_clauses.append("TableName = %s")
                params.append(table_name)
            
            if start_date is not None:
                where_clauses.append("DATE(CreatedDate) >= %s")
                params.append(start_date)
            
            if end_date is not None:
                where_clauses.append("DATE(CreatedDate) <= %s")
                params.append(end_date)
            
            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            params.append(limit)
            
            query = f"""SELECT * FROM logs{where_sql}
                       ORDER BY CreatedDate DESC LIMIT %s"""
            
            cursor.execute(query, params)
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"Error getting logs: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    async def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logged actions."""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Top actions
            cursor.execute(
                """SELECT Action, COUNT(*) as count
                   FROM logs
                   GROUP BY Action
                   ORDER BY count DESC
                   LIMIT 10"""
            )
            top_actions = cursor.fetchall()
            
            # Top users
            cursor.execute(
                """SELECT Username, COUNT(*) as count
                   FROM logs
                   WHERE Username IS NOT NULL
                   GROUP BY Username
                   ORDER BY count DESC
                   LIMIT 10"""
            )
            top_users = cursor.fetchall()
            
            # Today's count
            cursor.execute(
                """SELECT COUNT(*) as count
                   FROM logs
                   WHERE DATE(CreatedDate) = CURDATE()"""
            )
            today_result = cursor.fetchone()
            today_count = today_result['count'] if today_result else 0
            
            return {
                "topActions": top_actions,
                "topUsers": top_users,
                "todayCount": today_count
            }
            
        except Error as e:
            logger.error(f"Error getting log stats: {e}")
            raise
        finally:
            if connection:
                connection.close()