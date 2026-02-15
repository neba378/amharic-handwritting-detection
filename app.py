
"""Amharic Handwriting Recognition Backend"""

import io
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─── Configuration ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 28
MODEL_PATH = "amharic_model_scripted.pt"
CLASS_NAMES_PATH = "class_names.json"

# ─── Initialize FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Amharic Character Recognition API",
    description="Deep learning API for recognizing handwritten Amharic (Ge'ez) characters",
    version="1.0.0"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model & Class Names ────────────────────────────────────────────────
print(f"Loading model on {DEVICE}...")

try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Make sure 'amharic_model_scripted.pt' is in the same directory")
    model = None

try:
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
    print(f"✓ Loaded {len(CLASS_NAMES)} class names")
except Exception as e:
    print(f"✗ Error loading class names: {e}")
    print("Make sure 'class_names.json' is in the same directory")
    CLASS_NAMES = []


# ─── Helper Functions ────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to model-ready tensor."""
    try:
        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Cannot decode image")
        
        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor (1, 1, 28, 28)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        return tensor
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


# ─── API Routes ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES),
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict Amharic character from uploaded image.
    
    Returns:
        - character: Predicted character
        - confidence: Confidence percentage
        - top5: Top 5 predictions with confidence scores
    """
    if model is None or not CLASS_NAMES:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Read and validate image
    contents = await image.read()
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")
    
    # Preprocess
    try:
        tensor = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Predict
    try:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_idx = probs.topk(5, dim=1)
        
        # Format response
        response = {
            "character": CLASS_NAMES[top5_idx[0, 0].item()],
            "confidence": round(top5_probs[0, 0].item() * 100, 2),
            "top5": [
                {
                    "character": CLASS_NAMES[top5_idx[0, i].item()],
                    "confidence": round(top5_probs[0, i].item() * 100, 2),
                    "rank": i + 1
                }
                for i in range(min(5, len(CLASS_NAMES)))
            ]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/classes")
async def get_classes():
    """Get list of all supported classes."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Amharic Character Recognition API")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {'✓ Loaded' if model else '✗ Not loaded'}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
