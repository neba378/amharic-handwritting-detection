# Amharic Handwritten Character Recognition Web App

A web application for recognizing handwritten Amharic (Ge'ez script) characters.

## Features

- Draw directly on canvas with mouse or touch
- Upload images of handwritten characters
- Real-time predictions with confidence scores
- Top 5 predictions displayed
- Responsive UI for desktop and mobile
- Fast inference
- 231 Amharic characters supported

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Files

Make sure you have these files in the same directory:

- `amharic_model_scripted.pt` - TorchScript model
- `class_names.json` - Character mappings (231 Ge'ez characters)
- `app.py` - Backend
- `index.html` - Frontend interface

### 3. Run the Server

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

Navigate to: **http://localhost:8000**

## Usage

1. **Draw** a character on the canvas using your mouse or finger (on touch devices)
2. Or **upload** an image by clicking "Upload Image"
3. Click **"Recognize"** to get the prediction
4. View the predicted character with confidence score and top 5 alternatives

## Supported Characters

The model recognizes 231 Amharic (Ge'ez script) characters including:

- 33 base consonants (ሀ-ፐ)
- 7 vowel forms for each consonant (fidel system)
- All standard Amharic writing characters

## API Endpoints

### `GET /`

Serves the main web application interface.

### `POST /predict`

Upload an image and get character prediction.

**Request:** Form-data with `image` field  
**Response:**

```json
{
  "character": "ሀ",
  "confidence": 95.43,
  "top5": [
    { "character": "ሀ", "confidence": 95.43, "rank": 1 },
    { "character": "ሁ", "confidence": 2.31, "rank": 2 },
    { "character": "ሂ", "confidence": 1.15, "rank": 3 },
    { "character": "ሃ", "confidence": 0.67, "rank": 4 },
    { "character": "ሄ", "confidence": 0.44, "rank": 5 }
  ]
}
```

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 231,
  "device": "cpu"
}
```

### `GET /classes`

Get list of all supported character classes.

**Response:**

```json
{
  "classes": ["ሀ", "ሁ", "ሂ", ...],
  "count": 231
}
```

## Project Structure

```
handwritten/
├── app.py                          # FastAPI backend API
├── index.html                      # Frontend UI with canvas
├── requirements.txt                # Python dependencies
├── amharic_model_scripted.pt       # Trained PyTorch model
├── amharic_model.onnx              # ONNX export (optional)
├── amharic_model.onnx.data         # ONNX model data
├── class_names.json                # 231 Amharic character mappings
└── README.md                       # This file
```

## Technical Details

- **Framework:** FastAPI for backend, vanilla JavaScript for frontend
- **Model:** PyTorch TorchScript for fast inference
- **Image Processing:** OpenCV for preprocessing
- **Input Size:** 28x28 grayscale images
- **Characters:** 231 Ge'ez script characters (33 × 7 fidel system)

## Troubleshooting

**Error: "Model not loaded"**

- Make sure `amharic_model_scripted.pt` is in the same directory as `app.py`

**Error: "Cannot decode image"**

- Ensure uploaded images are valid image files (PNG, JPG, etc.)
- Try drawing on the canvas instead

**Error: Port 8000 already in use**

```bash
# Use a different port
uvicorn app:app --host 0.0.0.0 --port 8080
```

**Blank predictions or low confidence**

- Make sure the drawn character is clear and centered
- Try increasing line thickness by drawing slower
- Ensure good contrast (dark on light background)

## Development

To run in development mode with auto-reload:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Model Information

The model was trained to recognize 231 Amharic characters organized in the traditional fidel system:

- Each base consonant has 7 forms (ሀ ሁ ሂ ሃ ሄ ህ ሆ)
- Includes all 33 base characters from ሀ to ፐ

## License

MIT License - Feel free to use for educational and commercial purposes.

## Credits

Built with ❤️ for Amharic language preservation and digital literacy.

---

**Status:** ✅ Fully Functional  
**Backend:** http://localhost:8000  
**Model:** 231 Classes Loaded  
**Device:** CPU (GPU compatible)

**Error: "Cannot connect to backend"**

- Ensure the backend server is running on port 8000
- Check that no firewall is blocking the connection

**Poor predictions**

- Ensure characters are drawn clearly and large enough
- Try different pen thickness or upload a clearer image

## Tech Stack

- **Backend:** FastAPI, PyTorch, OpenCV
- **Frontend:** HTML, CSS, JavaScript (Vanilla)
- **Model:** CNN trained on Amharic handwriting dataset

## License

MIT License - Feel free to use and modify!
