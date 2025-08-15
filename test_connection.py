
import mysql.connector
from config import get_database_config

def test_database_connection():
    """Test MySQL database connection independently."""
    try:
        print("Testing MySQL database connection...")
        config = get_database_config()
        
        print(f"Connecting to: {config['host']}")
        print(f"Database: {config['database']}")
        print(f"User: {config['user']}")
        
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✅ Successfully connected to MySQL Server version: {version[0]}")
            
            # Test if our tables exist
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"✅ Found {len(tables)} tables in database")
            
            # Check for required tables
            table_names = [table[0] for table in tables]
            required_tables = ['FaceData', 'Student', 'Teacher']
            
            for table in required_tables:
                if table in table_names:
                    print(f"✅ Table '{table}' exists")
                else:
                    print(f"❌ Table '{table}' missing")
            
            cursor.close()
            connection.close()
            return True
            
    except mysql.connector.Error as e:
        print(f"❌ Database connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
