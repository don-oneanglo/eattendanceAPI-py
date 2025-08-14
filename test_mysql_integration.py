"""
Test script to verify MySQL database integration with the Face Recognition API.
"""

import requests
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import time

API_BASE = "http://0.0.0.0:5000"

def create_test_image(size=(200, 200), color=(255, 255, 255)):
    """Create a simple test image with a face-like shape."""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face-like shape
    # Head (circle)
    draw.ellipse([50, 50, 150, 150], fill=(255, 220, 177), outline=(0, 0, 0))
    
    # Eyes
    draw.ellipse([70, 80, 85, 95], fill=(0, 0, 0))
    draw.ellipse([115, 80, 130, 95], fill=(0, 0, 0))
    
    # Nose
    draw.polygon([(95, 100), (105, 100), (100, 115)], fill=(255, 200, 150))
    
    # Mouth
    draw.arc([80, 120, 120, 140], 0, 180, fill=(255, 0, 0), width=2)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    img_data = buffer.getvalue()
    base64_string = base64.b64encode(img_data).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_string}"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{API_BASE}/api/health")
    data = response.json()
    
    print(f"Status: {data['status']}")
    print(f"Database Connected: {data['database_connected']}")
    print(f"Database Face Count: {data['database_face_count']}")
    print(f"Model Loaded: {data['model_loaded']}")
    print()
    
    return data['database_connected']

def test_face_enrollment():
    """Test face enrollment with database integration."""
    print("👤 Testing face enrollment...")
    
    # Create test image
    test_image = create_test_image()
    
    enrollment_data = {
        "person_type": "student",
        "person_code": "TEST001",
        "person_name": "Test Student",
        "image_base64": test_image,
        "original_name": "test_student.jpg"
    }
    
    response = requests.post(f"{API_BASE}/api/face/enroll", json=enrollment_data)
    data = response.json()
    
    print(f"Success: {data['success']}")
    print(f"Message: {data['message']}")
    if 'embedding_id' in data:
        print(f"Database ID: {data['embedding_id']}")
    print()
    
    return data['success']

def test_face_recognition():
    """Test face recognition with database logging."""
    print("🔍 Testing face recognition...")
    
    # Create test image (similar to enrolled one)
    test_image = create_test_image(color=(250, 250, 250))  # Slightly different
    
    recognition_data = {
        "image_base64": test_image,
        "person_type": "student",
        "similarity_threshold": 0.5,
        "source_image_name": "test_recognition.jpg"
    }
    
    response = requests.post(f"{API_BASE}/api/face/recognize", json=recognition_data)
    data = response.json()
    
    print(f"Success: {data['success']}")
    if data.get('match'):
        match = data['match']
        print(f"Match Found: {match['person_name']} ({match['person_code']})")
        print(f"Confidence: {match['confidence']:.3f}")
    else:
        print("No match found")
    print(f"Processing Time: {data['processing_time_ms']}ms")
    print()
    
    return data['success']

def test_list_faces():
    """Test listing enrolled faces."""
    print("📋 Testing face listing...")
    
    response = requests.get(f"{API_BASE}/api/face/list/student")
    data = response.json()
    
    print(f"Success: {data['success']}")
    print(f"Total Faces: {data['total_count']}")
    for face in data['faces']:
        print(f"  - {face['person_name']} ({face['person_code']}) - Has Image: {face['has_image']}")
    print()
    
    return data['success']

def test_face_deletion():
    """Test face deletion."""
    print("🗑️ Testing face deletion...")
    
    deletion_data = {
        "person_type": "student",
        "person_code": "TEST001"
    }
    
    response = requests.delete(f"{API_BASE}/api/face/delete", json=deletion_data)
    data = response.json()
    
    print(f"Success: {data['success']}")
    print(f"Message: {data['message']}")
    print()
    
    return data['success']

def main():
    """Run all tests."""
    print("🚀 MySQL Database Integration Test Suite")
    print("=" * 50)
    
    # Test health check
    db_connected = test_health_check()
    if not db_connected:
        print("❌ Database not connected - stopping tests")
        return
    
    # Test enrollment
    if not test_face_enrollment():
        print("❌ Face enrollment failed")
        return
    
    # Test listing faces
    if not test_list_faces():
        print("❌ Face listing failed")
        return
    
    # Test recognition
    if not test_face_recognition():
        print("❌ Face recognition failed")
        return
    
    # Test deletion
    if not test_face_deletion():
        print("❌ Face deletion failed")
        return
    
    # Final health check
    print("🏁 Final health check...")
    test_health_check()
    
    print("✅ All tests completed successfully!")
    print("🎉 MySQL database integration is working perfectly!")

if __name__ == "__main__":
    main()