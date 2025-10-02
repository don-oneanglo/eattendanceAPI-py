# API Updates for User Management System

This document contains the complete API endpoint specifications for the user management, user sessions, and user logs features.

## Overview

Three new database tables have been added to track administrators and their actions:
- **users** - Stores administrator and elevated user accounts
- **user_sessions** - Tracks login sessions  
- **logs** - Complete audit trail for all user actions

## Prerequisites

1. Database tables `users`, `user_sessions`, and `logs` must be created (see `attached_assets/database_schema.txt`)
2. Install bcrypt package: `npm install bcrypt`
3. Import bcrypt in server.js: `const bcrypt = require('bcrypt');`

## API Endpoints

### User Management Endpoints

#### 1. Get All Users
```
GET /api/admin/users
```
**Response:**
```json
{
  "success": true,
  "users": [
    {
      "Id": 1,
      "Username": "admin",
      "FullName": "Administrator",
      "Email": "admin@example.com",
      "Role": "admin",
      "IsActive": true,
      "LastLoginDate": "2025-10-02T10:30:00.000Z",
      "CreatedDate": "2025-10-01T08:00:00.000Z",
      "UpdatedDate": "2025-10-01T08:00:00.000Z"
    }
  ]
}
```

#### 2. Get Single User
```
GET /api/admin/users/:id
```
**Response:**
```json
{
  "success": true,
  "user": {
    "Id": 1,
    "Username": "admin",
    "FullName": "Administrator",
    "Email": "admin@example.com",
    "Role": "admin",
    "IsActive": true,
    "LastLoginDate": "2025-10-02T10:30:00.000Z",
    "CreatedDate": "2025-10-01T08:00:00.000Z",
    "UpdatedDate": "2025-10-01T08:00:00.000Z"
  }
}
```

#### 3. Create User
```
POST /api/admin/users
Content-Type: application/json
```
**Request Body:**
```json
{
  "username": "newuser",
  "password": "SecurePassword123!",
  "fullName": "New User",
  "email": "newuser@example.com",
  "role": "admin",
  "isActive": true
}
```
**Response:**
```json
{
  "success": true,
  "message": "User created successfully",
  "userId": 5
}
```

**Role Options:**
- `admin` - Standard administrator
- `super_admin` - Super administrator with elevated privileges
- `teacher` - Teacher with limited admin access
- `staff` - Staff member with basic access

#### 4. Update User
```
PUT /api/admin/users/:id
Content-Type: application/json
```
**Request Body:**
```json
{
  "username": "updateduser",
  "password": "NewPassword123!",
  "fullName": "Updated Name",
  "email": "updated@example.com",
  "role": "super_admin",
  "isActive": false
}
```
**Note:** All fields are optional. Password field can be omitted to keep existing password.

**Response:**
```json
{
  "success": true,
  "message": "User updated successfully"
}
```

#### 5. Delete User
```
DELETE /api/admin/users/:id
```
**Response:**
```json
{
  "success": true,
  "message": "User deleted successfully"
}
```

### User Sessions Endpoints

#### 6. Get All User Sessions
```
GET /api/admin/user-sessions
```
**Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "Id": 1,
      "UserId": 5,
      "SessionToken": "abc123def456...",
      "LoginTime": "2025-10-02T09:00:00.000Z",
      "LogoutTime": "2025-10-02T12:00:00.000Z",
      "IpAddress": "192.168.1.100",
      "UserAgent": "Mozilla/5.0...",
      "IsActive": false,
      "Username": "admin",
      "FullName": "Administrator",
      "Role": "admin"
    }
  ]
}
```
**Note:** Returns last 100 sessions ordered by LoginTime DESC

#### 7. Get Active User Sessions
```
GET /api/admin/user-sessions/active
```
**Response:** Same format as Get All User Sessions, but filtered to only active sessions (IsActive = 1)

#### 8. Terminate User Session
```
POST /api/admin/user-sessions/:id/terminate
```
**Response:**
```json
{
  "success": true,
  "message": "Session terminated successfully"
}
```
**Effect:** Sets `IsActive = 0` and `LogoutTime = NOW()` for the specified session

### User Logs Endpoints

#### 9. Get User Logs (with filtering)
```
GET /api/admin/logs?userId=5&action=CREATE_USER&tableName=users&startDate=2025-10-01&endDate=2025-10-02&limit=100
```
**Query Parameters:**
- `userId` (optional) - Filter by user ID
- `action` (optional) - Filter by action type
- `tableName` (optional) - Filter by affected table
- `startDate` (optional) - Filter logs from this date (YYYY-MM-DD)
- `endDate` (optional) - Filter logs until this date (YYYY-MM-DD)
- `limit` (optional) - Maximum number of records to return (default: 100)

**Response:**
```json
{
  "success": true,
  "logs": [
    {
      "Id": 1,
      "UserId": 5,
      "Username": "admin",
      "Action": "CREATE_USER",
      "TableName": "users",
      "RecordId": 10,
      "OldValue": null,
      "NewValue": "{\"username\":\"newuser\",\"fullName\":\"New User\",\"role\":\"admin\"}",
      "Description": "Created new user: newuser",
      "IpAddress": "192.168.1.100",
      "UserAgent": "Mozilla/5.0...",
      "CreatedDate": "2025-10-02T10:30:00.000Z"
    }
  ]
}
```

**Common Action Types:**
- `CREATE_USER` - User created
- `UPDATE_USER` - User updated
- `DELETE_USER` - User deleted
- `TERMINATE_SESSION` - Session terminated
- `LOGIN` - User logged in
- `LOGOUT` - User logged out

#### 10. Get Log Statistics
```
GET /api/admin/logs/stats
```
**Response:**
```json
{
  "success": true,
  "stats": {
    "topActions": [
      { "Action": "CREATE_USER", "count": 15 },
      { "Action": "UPDATE_USER", "count": 10 }
    ],
    "topUsers": [
      { "Username": "admin", "count": 25 },
      { "Username": "staff1", "count": 10 }
    ],
    "todayCount": 42
  }
}
```

## Helper Function: logUserAction

A helper function has been added to server.js to automatically log user actions:

```javascript
async function logUserAction(userId, username, action, tableName = null, recordId = null, oldValue = null, newValue = null, description = null, ipAddress = null, userAgent = null)
```

**Usage Example:**
```javascript
await logUserAction(
    req.userId,
    req.username,
    'DELETE_STUDENT',
    'Student',
    studentId,
    oldStudentData,
    null,
    `Deleted student: ${studentCode}`,
    req.ip,
    req.get('user-agent')
);
```

## Security Considerations

1. **Password Hashing:** All passwords are hashed using bcrypt with 10 salt rounds before storage
2. **Password Updates:** When updating a user, omit the password field to keep the existing password
3. **Session Management:** Sessions can be terminated remotely for security purposes
4. **Audit Trail:** All administrative actions are logged with IP address and user agent
5. **Sensitive Data:** Password hashes are never returned in API responses

## Integration Steps

To integrate these endpoints into your existing project:

1. **Install bcrypt:**
   ```bash
   npm install bcrypt
   ```

2. **Add bcrypt import to server.js:**
   ```javascript
   const bcrypt = require('bcrypt');
   ```

3. **Add the logUserAction helper function** (see server.js lines 1803-1817)

4. **Add all endpoint handlers** to server.js before the "Initialize database and start server" comment (see server.js lines 1819-2151)

5. **Update your admin UI** to include the three new sections:
   - User Management
   - User Sessions  
   - User Logs

6. **Add JavaScript functions** to admin-script.js for handling the UI interactions (see admin-script.js lines 763-1067)

## Testing the API

### Test User Creation:
```bash
curl -X POST http://localhost:5000/api/admin/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "TestPass123!",
    "fullName": "Test User",
    "email": "test@example.com",
    "role": "admin",
    "isActive": true
  }'
```

### Test Get Users:
```bash
curl http://localhost:5000/api/admin/users
```

### Test Get Logs:
```bash
curl "http://localhost:5000/api/admin/logs?limit=10"
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200 OK` - Success
- `400 Bad Request` - Invalid input (e.g., missing required fields, duplicate username)
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error responses follow this format:
```json
{
  "success": false,
  "message": "Error description here"
}
```

## Database Schema Reference

See `attached_assets/database_schema.txt` for complete SQL schema including:
- Table definitions
- Foreign key relationships
- Indexes for performance
- Field constraints

---

**Note:** This documentation assumes the database tables have been created and the database connection is properly configured in server.js.
