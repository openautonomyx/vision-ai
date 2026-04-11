# AutonomyX Vision AI

Unified Vision AI API — single container, all services.

## Services
- **YOLO26** — Object detection
- **Tesseract** — OCR (English + Hindi)
- **Whisper** — Speech-to-text
- **CLIP** — Image classification & embeddings
- **rembg** — Background removal
- **OpenCV** — Image processing (faces, edges, resize, analyze)

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /models | List available models |
| POST | /detect | YOLO object detection |
| POST | /ocr | Extract text from image |
| POST | /ocr/preprocess | Clean image for OCR |
| POST | /transcribe | Speech-to-text |
| POST | /detect-language | Detect spoken language |
| POST | /remove-bg | Remove image background |
| POST | /clip/classify | Zero-shot image classification |
| POST | /clip/embed | Generate image embedding |
| POST | /faces | Face detection |
| POST | /edges | Edge detection |
| POST | /resize | Resize image |
| POST | /analyze | Image analysis |

## Run
```bash
docker compose up -d --build
```

## API
```bash
# Object detection
curl -X POST http://localhost:8120/detect -F file=@photo.jpg

# OCR
curl -X POST http://localhost:8120/ocr -F file=@document.png

# Transcribe audio
curl -X POST http://localhost:8120/transcribe -F file=@audio.mp3

# Remove background
curl -X POST http://localhost:8120/remove-bg -F file=@photo.jpg -o output.png

# Classify image
curl -X POST "http://localhost:8120/clip/classify?labels=cat,dog,bird" -F file=@photo.jpg
```
