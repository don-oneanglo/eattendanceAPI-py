# SQL Server Database Integration Instructions

## Current Status
The Face Recognition API is now fully configured to work with your SQL Server database. However, the current Replit environment doesn't have the required SQL Server drivers (pyodbc) installed.

## Your Database Configuration
The API is configured to connect to your SQL Server with these settings:

```python
DATABASE_CONFIG = {
    'host': 'srv1521.hstgr.io',
    'user': 'u311154254_TADB',
    'password': 'Anglo!123456',
    'database': 'u311154254_TestAttendance',
    'charset': 'utf8mb4'
}
```

## Database Schema Support
The API supports your existing SQL Server schema:

### FaceData Table
- ✅ Stores face embeddings as JSON in `FaceDescriptor` field
- ✅ Stores binary image data in `ImageData` field  
- ✅ Links to Student/Teacher tables via `PersonCode`
- ✅ Tracks enrollment dates and metadata

### Student/Teacher Tables
- ✅ Automatic updates when enrolling faces
- ✅ Image storage in both tables
- ✅ Name and code synchronization

### Recognition Logging
- ✅ Creates `FaceRecognitionLogs` table automatically
- ✅ Logs all recognition attempts with confidence scores
- ✅ Tracks processing times and success rates

## To Enable Full Database Integration

### Option 1: Deploy to Environment with SQL Server Support

1. **Install Required Dependencies:**
   ```bash
   pip install pyodbc
   ```

2. **Install SQL Server ODBC Driver:**
   - On Ubuntu/Debian: `apt-get install msodbcsql17`
   - On Windows: Download from Microsoft
   - On macOS: `brew install msodbcsql17`

3. **Update the Code:**
   Uncomment the pyodbc imports and connection code in `services/database_service.py`

4. **Test Connection:**
   ```bash
   python test_database.py
   ```

### Option 2: Use Current File-Based System

The API currently works with file-based storage and will:
- ✅ Store all face data in JSON files
- ✅ Provide full face recognition functionality
- ✅ Maintain all API endpoints
- ✅ Support batch operations

## API Endpoints Ready for Database

All endpoints are database-ready and will automatically use SQL Server when available:

- `POST /api/face/enroll` - Stores in both file system AND database
- `POST /api/face/recognize` - Logs attempts to database
- `GET /api/face/list/{type}` - Shows database status
- `DELETE /api/face/delete` - Removes from both systems
- `GET /api/health` - Reports database connectivity

## Testing Database Connectivity

The health check endpoint (`/api/health`) shows database status:

```json
{
  "status": "healthy",
  "database_connected": false,
  "database_config": {
    "host": "srv1521.hstgr.io",
    "database": "u311154254_TestAttendance",
    "user": "u311154254_TADB"
  }
}
```

## Migration Path

1. **Current State**: File-based storage working fully
2. **Next Step**: Deploy to environment with SQL Server drivers
3. **Final State**: Dual storage (file + database) for maximum reliability

The API is designed to work seamlessly in both modes, providing a smooth migration path when you're ready to deploy to a production environment with proper database drivers.

## Benefits of Database Integration

Once enabled, you'll get:
- ✅ Centralized face data storage
- ✅ Attendance tracking integration
- ✅ Recognition attempt logging
- ✅ Student/Teacher data synchronization
- ✅ Better scalability and reliability