"""
Configuration management for the InsightFace API.
"""

import os
from functools import lru_cache
from typing import Optional


class Settings:
    """Application settings."""
    
    # Database Configuration
    DB_HOST: str = "srv1521.hstgr.io"
    DB_USER: str = "u311154254_TADB"
    DB_PASSWORD: str = "Anglo!123456"
    DB_NAME: str = "u311154254_TestAttendance"
    DB_CHARSET: str = "utf8mb4"
    DB_POOL_SIZE: int = 5
    
    # InsightFace Configuration
    INSIGHTFACE_MODEL: str = "buffalo_l"
    SIMILARITY_THRESHOLD: float = 0.6
    FACE_CONFIDENCE_THRESHOLD: float = 0.7
    AUTO_DOWNLOAD_MODELS: bool = True
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 10
    MAX_IMAGE_SIZE_MB: int = 10
    PROCESSING_TIMEOUT_SECONDS: int = 30
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    ENABLE_FACE_QUALITY_CHECK: bool = True
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_MINUTES: int = 1
    
    # File Paths
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    
    def __init__(self):
        """Initialize settings with environment variables."""
        self.DB_HOST = os.getenv("DB_HOST", "srv1521.hstgr.io")
        self.DB_USER = os.getenv("DB_USER", "u311154254_TADB")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "Anglo!123456")
        self.DB_NAME = os.getenv("DB_NAME", "u311154254_TestAttendance")
        self.DB_CHARSET = os.getenv("DB_CHARSET", "utf8mb4")
        self.DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
        
        self.INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
        self.FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.7"))
        self.AUTO_DOWNLOAD_MODELS = os.getenv("AUTO_DOWNLOAD_MODELS", "True").lower() == "true"
        
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
        self.PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "30"))
        
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.ENABLE_FACE_QUALITY_CHECK = os.getenv("ENABLE_FACE_QUALITY_CHECK", "True").lower() == "true"
        
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.RATE_LIMIT_WINDOW_MINUTES = int(os.getenv("RATE_LIMIT_WINDOW_MINUTES", "1"))
        
        self.MODELS_DIR = os.getenv("MODELS_DIR", "models")
        self.LOGS_DIR = os.getenv("LOGS_DIR", "logs")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Database configuration for MySQL connection
def get_database_config() -> dict:
    """Get database connection configuration."""
    settings = get_settings()
    return {
        'host': settings.DB_HOST,
        'user': settings.DB_USER,
        'password': settings.DB_PASSWORD,
        'database': settings.DB_NAME,
        'charset': settings.DB_CHARSET,
        'autocommit': True
    }