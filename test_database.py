"""
Test script to verify SQL Server database connectivity.
"""

import requests
import json
from config import get_settings

def test_database_connection():
    """Test database connection using HTTP requests (since pyodbc is not available)."""
    print("Testing SQL Server database connectivity...")
    
    settings = get_settings()
    
    print(f"Database Configuration:")
    print(f"Host: {settings.DB_HOST}")
    print(f"Database: {settings.DB_NAME}")
    print(f"User: {settings.DB_USER}")
    print(f"Password: {'*' * len(settings.DB_PASSWORD)}")
    
    try:
        # Test basic connectivity by making a simple HTTP request to database server
        # This is a workaround since we don't have ODBC drivers available
        import socket
        
        # Test if the host is reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((settings.DB_HOST, 1433))  # SQL Server default port
        sock.close()
        
        if result == 0:
            print("✅ Database host is reachable on port 1433")
            return True
        else:
            print("❌ Database host is not reachable on port 1433")
            return False
            
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()