# Replace face-api.js with FastAPI Face Recognition Backend

## Context
I have an existing web application that uses face-api.js for face recognition functionality. I want to replace it with a new FastAPI backend that provides more accurate face recognition using the InsightFace library.

## Current FastAPI Backend
The FastAPI backend is already running and provides these endpoints:

### Available Endpoints

1. **POST /enroll** - Register a new face
   - Form data: `name` (string), `file` (image file)
   - Returns: face_id, bounding_box, confidence
   - Use for: Initial face registration

2. **POST /verify** - Compare two images
   - Form data: `file1`, `file2` (image files)
   - Returns: is_match (boolean), similarity_score, bounding boxes
   - Use for: 1:1 face verification

3. **POST /identify** - Find matching face from database
   - Form data: `file` (image file)
   - Returns: top 3 matches with scores, or "Unknown"
   - Use for: 1:N face identification

4. **GET /faces** - List all enrolled faces
   - Returns: list of enrolled people with metadata
   - Use for: Managing face database

5. **DELETE /faces/{name}** - Remove a person
   - URL parameter: name
   - Use for: Deleting face records

6. **GET /health** - Check system status
   - Returns: system health and enrolled face count

## Migration Tasks

### 1. Replace face-api.js Client Code
Replace your existing face-api.js JavaScript code with fetch API calls to the FastAPI backend:

```javascript
// OLD: face-api.js detection
// const detections = await faceapi.detectAllFaces(video)

// NEW: FastAPI enrollment
async function enrollFace(name, imageFile) {
  const formData = new FormData();
  formData.append('name', name);
  formData.append('file', imageFile);
  
  const response = await fetch('/enroll', {
    method: 'POST',
    body: formData
  });
  return await response.json();
}

// NEW: FastAPI identification
async function identifyFace(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('/identify', {
    method: 'POST',
    body: formData
  });
  return await response.json();
}

// NEW: FastAPI verification
async function verifyFaces(imageFile1, imageFile2) {
  const formData = new FormData();
  formData.append('file1', imageFile1);
  formData.append('file2', imageFile2);
  
  const response = await fetch('/verify', {
    method: 'POST',
    body: formData
  });
  return await response.json();
}
```

### 2. Image Capture and Processing
Update your image capture logic to work with the new API:

```javascript
// Capture image from video/canvas
function captureImageFromVideo(video) {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  
  return new Promise(resolve => {
    canvas.toBlob(resolve, 'image/jpeg', 0.95);
  });
}

// Use captured image with API
async function processVideoFrame(video) {
  const imageBlob = await captureImageFromVideo(video);
  const result = await identifyFace(imageBlob);
  
  if (result.matches.length > 0 && result.matches[0].is_match) {
    console.log('Recognized:', result.matches[0].name);
  } else {
    console.log('Unknown person');
  }
}
```

### 3. Remove face-api.js Dependencies
Remove these from your HTML/package.json:
- `face-api.js` library imports
- Model loading code
- face-api.js initialization

### 4. Error Handling
Update error handling for the new API responses:

```javascript
async function handleApiCall(apiFunction, ...args) {
  try {
    const result = await apiFunction(...args);
    if (result.detail) {
      // API returned an error
      console.error('API Error:', result.detail);
      return null;
    }
    return result;
  } catch (error) {
    console.error('Network Error:', error);
    return null;
  }
}
```

### 5. UI Updates
Update your user interface to work with the new API:

- Replace face-api.js loading indicators with API call loading states
- Update face recognition results display to use the new response format
- Add controls for enrollment (name input + image capture)
- Add face management (list/delete enrolled faces)

## Key Differences from face-api.js

1. **Server-side Processing**: All face recognition now happens on the server
2. **Higher Accuracy**: Uses InsightFace buffalo_l model instead of browser models
3. **Persistent Storage**: Face embeddings are stored in a database
4. **Better Performance**: No need to load large models in the browser
5. **Image Upload Required**: Must send images to server instead of processing locally

## Implementation Steps

1. **Set up API calls**: Create JavaScript functions for each endpoint
2. **Update image capture**: Modify video/camera capture to create image blobs
3. **Replace detection logic**: Remove face-api.js calls, add API calls
4. **Update UI**: Modify interface to show API responses
5. **Add enrollment flow**: Create UI for registering new faces
6. **Test thoroughly**: Verify all face recognition features work with new backend

## Sample Complete Integration

```javascript
class FaceRecognitionAPI {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  async enroll(name, imageBlob) {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', imageBlob);
    return this.makeRequest('/enroll', 'POST', formData);
  }

  async identify(imageBlob) {
    const formData = new FormData();
    formData.append('file', imageBlob);
    return this.makeRequest('/identify', 'POST', formData);
  }

  async verify(imageBlob1, imageBlob2) {
    const formData = new FormData();
    formData.append('file1', imageBlob1);
    formData.append('file2', imageBlob2);
    return this.makeRequest('/verify', 'POST', formData);
  }

  async listFaces() {
    return this.makeRequest('/faces', 'GET');
  }

  async deleteFace(name) {
    return this.makeRequest(`/faces/${encodeURIComponent(name)}`, 'DELETE');
  }

  async makeRequest(endpoint, method, body = null) {
    const options = { method };
    if (body) options.body = body;

    const response = await fetch(this.baseUrl + endpoint, options);
    return await response.json();
  }
}

// Usage
const faceAPI = new FaceRecognitionAPI();

// Replace your face-api.js code with:
const result = await faceAPI.identify(imageBlob);
if (result.matches.length > 0 && result.matches[0].is_match) {
  console.log('Person identified:', result.matches[0].name);
}
```

This migration will give you more accurate face recognition with persistent storage and better performance.