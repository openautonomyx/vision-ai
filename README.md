# AutonomyX Vision AI

Unified Vision AI API — single container, production-ready with authentication, metrics, and MCP agent integration.

## Production Features

- **API Key Authentication** — Create and manage API keys with role-based access control
- **Rate Limiting** — Per-key rate limits to prevent abuse
- **Metrics & Observability** — Prometheus metrics endpoint for monitoring
- **Request Validation** — File size limits and MIME type validation
- **Structured Errors** — Consistent error responses with error codes
- **Request ID Tracking** — All requests tracked with unique IDs
- **MCP Agent Integration** — Exposes all services as MCP tools for AI agents

## Services

| Service | Description |
|--------|-------------|
| YOLO26 | Object detection with YOLO |
| Tesseract | OCR (English + Hindi) |
| Whisper | Speech-to-text transcription |
| CLIP | Image classification & embeddings |
| rembg | Background removal |
| OpenCV | Image processing (faces, edges, resize, analyze) |

## Quick Start

### Development
```bash
# Clone and run
docker compose --profile dev up -d --build
```

### Production (CPU)
```bash
docker compose --profile prod up -d
```

### Production (GPU)
```bash
docker compose --profile gpu up -d
```

## API Documentation

### Authentication

First, create an API key:

```bash
curl -X POST http://localhost:8000/auth/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "role": "developer", "rate_limit": 100}'
```

Then use the key in requests:

```bash
curl -X POST http://localhost:8000/detect \
  -H "X-API-Key: ax_your_api_key_here" \
  -F file=@photo.jpg
```

### Core Endpoints

| Method | Path | Description | Auth Required |
|--------|------|-------------|-------------|
| GET | `/health` | Health check | No |
| GET | `/models` | List models | No |
| GET | `/metrics` | Prometheus metrics | No |
| GET | `/usage` | Usage statistics | Yes |
| POST | `/auth/api-keys` | Create API key | No |
| GET | `/auth/api-keys` | List API keys | Yes |
| DELETE | `/auth/api-keys/{name}` | Revoke API key | Yes |
| POST | `/detect` | Object detection | Yes |
| POST | `/ocr` | Extract text from image | Yes |
| POST | `/transcribe` | Speech-to-text | Yes |
| POST | `/remove-bg` | Remove background | Yes |
| POST | `/clip/classify` | Image classification | Yes |
| POST | `/clip/embed` | Image embedding | Yes |
| POST | `/faces` | Face detection | Yes |
| POST | `/resize` | Resize image | Yes |
| POST | `/analyze` | Image analysis | Yes |

### Example Usage

```bash
# Basic API calls
curl -X POST http://localhost:8000/detect -F file=@photo.jpg

# OCR
curl -X POST http://localhost:8000/ocr -F file=@document.png

# Transcribe audio
curl -X POST http://localhost:8000/transcribe -F file=@audio.mp3

# Remove background
curl -X POST http://localhost:8000/remove-bg -F file=@photo.jpg -o output.png

# Classify image
curl -X POST "http://localhost:8000/clip/classify?labels=cat,dog,bird" -F file=@photo.jpg
```

### MCP Server

The MCP server exposes Vision AI as tools for AI agents:

```bash
# Run MCP server
python mcp_server.py

# Or with API key
VISION_API_KEY=ax_your_key python mcp_server.py
```

Available MCP tools:
- `detect_objects` — Detect objects with YOLO
- `extract_text` — OCR text extraction
- `transcribe_audio` — Speech-to-text
- `remove_background` — Background removal
- `classify_image` — CLIP classification
- `generate_embedding` — CLIP embeddings
- `detect_faces` — Face detection
- `analyze_image` — Image analysis
- `comprehensive_analyze` — Multi-tool analysis
- `compare_images` — Image similarity
- `extract_document_fields` — Document parsing
- `resize_image` — Resize images

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `YOLO_MODEL` | YOLO model | `yolo26n.pt` |
| `WHISPER_MODEL` | Whisper model | `base` |
| `VISION_API_URL` | API URL | `http://localhost:8000` |
| `VISION_API_KEY` | API key for MCP | - |
| `ENVIRONMENT` | dev/prod | `development` |

### Docker Profiles

- `dev` — Development with port 8120
- `prod` — Production with port 8000
- `gpu` — GPU-accelerated with CUDA

## Documentation

OpenAPI docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Metrics

Prometheus-format metrics available at `/metrics`:

```
# TYPE vision_api_requests_total counter
vision_api_requests_total{endpoint="/detect",status="200"} 1234
vision_api_request_duration_seconds{endpoint="/detect"} 0.234
```
