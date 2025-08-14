-- MySQL Database Schema for Face Recognition API
-- For Hostinger MySQL Database: u311154254_TestAttendance

-- Create FaceData table
CREATE TABLE IF NOT EXISTS FaceData (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    PersonType VARCHAR(20) CHECK (PersonType IN ('student', 'teacher')),
    PersonCode CHAR(10) NOT NULL,
    ImageData LONGBLOB,
    FaceDescriptor TEXT, -- Store InsightFace embeddings as JSON
    OriginalName VARCHAR(255),
    ContentType VARCHAR(100),
    CreatedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_person_type_code (PersonType, PersonCode)
);

-- Create Student table
CREATE TABLE IF NOT EXISTS Student (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    StudentCode CHAR(10),
    StudentNickname VARCHAR(100),
    StudentName VARCHAR(200),
    StudentImage LONGBLOB,
    EmailAddress VARCHAR(100),
    Campus VARCHAR(50),
    Form VARCHAR(100),
    CreatedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_student_code (StudentCode)
);

-- Create Teacher table
CREATE TABLE IF NOT EXISTS Teacher (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    TeacherCode CHAR(10),
    TeacherNickname VARCHAR(100),
    TeacherName VARCHAR(200),
    TeacherImage LONGBLOB,
    EmailAddress VARCHAR(100),
    Campus VARCHAR(50),
    Department VARCHAR(100),
    CreatedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE INDEX idx_teacher_code (TeacherCode)
);

-- Create Sessions table (if needed for attendance)
CREATE TABLE IF NOT EXISTS Sessions (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    SessionName VARCHAR(200),
    SessionDate DATETIME,
    Campus VARCHAR(50),
    CreatedDate DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create AttendanceRecords table
CREATE TABLE IF NOT EXISTS AttendanceRecords (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    SessionId INT NOT NULL,
    StudentCode CHAR(10) NOT NULL,
    Status VARCHAR(20) CHECK (Status IN ('Present', 'Absent', 'Late')),
    AttendanceDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    CreatedDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SessionId) REFERENCES Sessions(Id)
);

-- Create FaceRecognitionLogs table for tracking recognition attempts
CREATE TABLE IF NOT EXISTS FaceRecognitionLogs (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    PersonCode CHAR(10),
    PersonType VARCHAR(20),
    Confidence FLOAT,
    ProcessingTimeMs INT,
    SourceImageName VARCHAR(255),
    RecognitionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    Success BOOLEAN,
    INDEX idx_recognition_date (RecognitionDate),
    INDEX idx_person_code (PersonCode)
);

-- Insert sample data (optional)
-- INSERT INTO Student (StudentCode, StudentName, Campus, Form) VALUES
-- ('ST001', 'John Doe', 'Main Campus', 'Form 5A'),
-- ('ST002', 'Jane Smith', 'Main Campus', 'Form 5B');

-- INSERT INTO Teacher (TeacherCode, TeacherName, Campus, Department) VALUES
-- ('TC001', 'Dr. Smith', 'Main Campus', 'Mathematics'),
-- ('TC002', 'Ms. Johnson', 'Main Campus', 'English');