"""
Image processing utilities.
"""

import base64
from typing import Tuple
from io import BytesIO
from PIL import Image


def validate_base64_image(base64_string: str) -> bool:
    """Validate base64 image format."""
    try:
        if not base64_string.startswith('data:image/'):
            return False
        
        # Extract the actual base64 data
        base64_data = base64_string.split(',')[1]
        
        # Try to decode
        image_bytes = base64.b64decode(base64_data)
        
        # Try to open as image
        Image.open(BytesIO(image_bytes))
        
        return True
    except Exception:
        return False


def get_image_info(base64_string: str) -> dict:
    """Get basic image information."""
    try:
        if base64_string.startswith('data:image/'):
            base64_data = base64_string.split(',')[1]
        else:
            base64_data = base64_string
        
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))
        
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "byte_size": len(image_bytes)
        }
    except Exception as e:
        return {"error": str(e)}


def resize_image_if_needed(base64_string: str, max_width: int = 1920, max_height: int = 1080) -> str:
    """Resize image if it's too large."""
    try:
        # Extract format and data
        if base64_string.startswith('data:image/'):
            format_part, base64_data = base64_string.split(',', 1)
        else:
            format_part = 'data:image/jpeg;base64'
            base64_data = base64_string
        
        # Decode image
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Check if resizing is needed
        if image.width <= max_width and image.height <= max_height:
            return base64_string
        
        # Calculate new size maintaining aspect ratio
        ratio = min(max_width / image.width, max_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        
        # Resize image
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert back to base64
        output_buffer = BytesIO()
        resized_image.save(output_buffer, format='JPEG', quality=85)
        output_buffer.seek(0)
        
        new_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        return f"{format_part},{new_base64}"
        
    except Exception:
        # Return original if resizing fails
        return base64_string