# Fix Git Large File Issues

## Problem
GitHub rejected your push because the InsightFace model files are too large (over 100MB limit).

## Solution Steps

### 1. Remove large files from Git tracking
```bash
# Remove the models directory from Git tracking (but keep files locally)
git rm -r --cached models/

# Also remove any large files that might be tracked
git rm --cached storage/faces_db.json 2>/dev/null || true
```

### 2. Commit the removal
```bash
git add .gitignore
git commit -m "Remove large model files from Git tracking, add .gitignore"
```

### 3. Push your changes
```bash
git push origin main
```

## What's Excluded Now

The `.gitignore` file I created excludes:
- `models/` directory (contains large ONNX model files)
- `*.onnx` files (individual model files)
- `*.zip` files (compressed models)
- `storage/faces_db.json` (face database)
- `storage/*.jpg` (face images)

## Manual Setup for Other Developers

After cloning the repository, developers need to:

1. **Install dependencies:**
   ```bash
   pip install insightface==0.7.3 onnxruntime==1.17.1 fastapi==0.111.0 uvicorn==0.29.0 pillow==10.2.0 numpy==1.26.4 python-multipart==0.0.9
   ```

2. **Create directories:**
   ```bash
   mkdir -p models storage
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

The models will download automatically on first run (about 275MB download).

## Alternative: Use Git LFS (Optional)

If you want to track model files with Git LFS instead:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "models/**/*.onnx"
git lfs track "models/**/*.zip"

# Add .gitattributes
git add .gitattributes

# Remove from .gitignore and re-add files
# Edit .gitignore to remove models/ line
git add models/
git commit -m "Add models with Git LFS"
```

## Current Status

✅ `.gitignore` created to exclude large files
✅ Face recognition API still running normally
✅ Models will auto-download when needed

The application will continue working normally - the models are only excluded from Git, not deleted from your local system.