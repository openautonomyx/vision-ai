"""
AutonomyX Vision AI — Unified API
YOLO26 + OpenCV + Tesseract OCR + CLIP + Whisper + Background Remover

Production-ready with:
- API Key authentication
- Rate limiting
- Structured errors
- Request validation
- Metrics and observability
- Request ID tracking
"""
import os
import uuid
import time
import tempfile
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Query, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import pytesseract
import cv2

from auth import (
    APIKeyCreate, APIKeyResponse, verify_api_key, create_api_key, 
    revoke_api_key, list_api_keys as _list_api_keys, api_key_header, api_keys_store
)
from rate_limit import rate_limiter
from metrics import metrics, generate_request_id

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif",
    "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg",
    "video/mp4", "video/webm"
}

# ══════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="AutonomyX Vision AI",
    version="1.0.0",
    description=(
        "Unified Vision AI API: object detection, OCR, speech-to-text, "
        "image classification, background removal. Production-ready with auth and metrics."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID and metrics middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.time() - start
        # Record metrics
        metrics.increment(f"vision_api_requests_total", {"endpoint": request.url.path, "status": getattr(response, 'status_code', 0)})
        metrics.observe(f"vision_api_request_duration_seconds", duration, {"endpoint": request.url.path})
        response.headers["X-Request-ID"] = request_id

# ══════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════
from ultralytics import YOLO

model_name = os.environ.get("YOLO_MODEL", "yolo26n.pt")
yolo = YOLO(model_name)

# Lazy-load heavy models
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_whisper_model = None

def get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        _clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_tokenizer

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(os.environ.get("WHISPER_MODEL", "base"))
    return _whisper_model

# ══════════════════════════════════════════════════════════════════════
# Request Validation & Error Handling
# ══════════════════════════════════════════════════════════════════════
class APIError(Exception):
    """Structured API error."""
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )

async def validate_file(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> bytes:
    """Validate and read uploaded file."""
    # Check content type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise APIError(
            "INVALID_FILE_TYPE",
            f"File type {file.content_type} not allowed",
            status_code=400
        )
    
    # Read content
    content = await file.read()
    
    # Check size
    if len(content) > max_size:
        raise APIError(
            "FILE_TOO_LARGE",
            f"File size {len(content)} exceeds maximum {max_size}",
            status_code=413
        )
    
    return content

async def get_authenticated_key(api_key: str = Depends(api_key_header)) -> dict:
    """Dependency to verify API key with rate limiting."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    key_data = verify_api_key(api_key)
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    # Check rate limit
    if not rate_limiter.check(api_key, key_data["rate_limit"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return key_data

async def require_admin(key_data: dict = Depends(get_authenticated_key)) -> dict:
    """Dependency to require admin role."""
    if key_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return key_data

# Bootstrap admin key (set via environment for first-time setup)
ADMIN_BOOTSTRAP_TOKEN = os.environ.get("ADMIN_BOOTSTRAP_TOKEN", "")

async def require_bootstrap_or_admin():
    """Require bootstrap token or admin role."""
    def _check(request: Request):
        # Check for bootstrap token
        auth_header = request.headers.get("Authorization", "")
        if auth_header == f"Bearer {ADMIN_BOOTSTRAP_TOKEN}":
            return True
        # Otherwise require admin
        try:
            key_data = verify_api_key(
                request.headers.get("X-API-Key", "").replace("Bearer ", "")
            )
            if key_data and key_data.get("role") == "admin":
                return True
        except:
            pass
        raise HTTPException(status_code=403, detail="Admin or bootstrap required")
    return _check

# ══════════════════════════════════════════════════════════════════════
# Authentication Endpoints
# ══════════════════════════════════════════════════════════════════════
@app.post("/auth/api-keys", response_model=APIKeyResponse, tags=["Auth"])
async def create_key(request: Request, data: APIKeyCreate):
    """Create a new API key. Requires bootstrap token or admin role."""
    # Check for bootstrap token or admin
    auth_header = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-API-Key", "")
    
    is_authorized = False
    if ADMIN_BOOTSTRAP_TOKEN and auth_header == f"Bearer {ADMIN_BOOTSTRAP_TOKEN}":
        is_authorized = True
    elif api_key:
        key_data = verify_api_key(api_key)
        if key_data and key_data.get("role") == "admin":
            is_authorized = True
    
    if not is_authorized:
        raise HTTPException(
            status_code=403, 
            detail="Admin role or bootstrap token required. Set ADMIN_BOOTSTRAP_TOKEN env var for first-time setup."
        )
    
    key, response = create_api_key(data)
    response.key = key
    return response

@app.get("/auth/api-keys", tags=["Auth"])
async def list_keys(_: dict = Depends(require_admin)):
    """List all API keys (admin only)."""
    return {"keys": _list_api_keys()}

@app.delete("/auth/api-keys/{key_name}", tags=["Auth"])
async def delete_key(key_name: str, _: dict = Depends(require_admin)):
    """Revoke an API key (admin only)."""
    for key_hash, data in list(api_keys_store.items()):
        if data["name"] == key_name:
            api_keys_store[key_hash]["is_active"] = False
            return {"status": "revoked", "name": key_name}
    raise HTTPException(status_code=404, detail="Key not found")

# ══════════════════════════════════════════════════════════════════════════════
# Metrics & Observability
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(content=metrics.get_metrics())

@app.get("/usage", tags=["Observability"])
async def get_usage(_: dict = Depends(get_authenticated_key)):
    """Usage statistics."""
    return metrics.get_stats()

# ═════════════════════════════════════════════════════════════════════════=======
# Health & Info
# ══════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["Health"])
def health(request: Request):
    """Health check endpoint."""
    request_id = request.headers.get("X-Request-ID", "N/A")
    return {
        "status": "ok",
        "version": "1.0.0",
        "request_id": request_id,
        "services": {
            "yolo": model_name,
            "opencv": cv2.__version__,
            "tesseract": pytesseract.get_tesseract_version().public,
            "clip": "ViT-B-32 (lazy)",
            "whisper": os.environ.get("WHISPER_MODEL", "base") + " (lazy)",
            "rembg": "u2net (lazy)"
        }
    }

@app.get("/models", tags=["Models"])
def list_models():
    """List available models."""
    return {
        "yolo": ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        "opencv": ["face_detection", "edge_detection", "resize", "analyze"],
        "ocr": {"engine": "tesseract", "languages": ["eng", "hin"]},
        "clip": {"model": "ViT-B-32", "tasks": ["classify", "embed"]},
        "whisper": {"models": ["tiny", "base", "small", "medium"], "tasks": ["transcribe", "detect_language"]},
        "rembg": {"task": "background_removal"},
        "supported_endpoints": ["/detect", "/ocr", "/transcribe", "/remove-bg", "/clip/classify", "/clip/embed", "/faces", "/edges", "/resize", "/analyze"]
    }

# ══════════════════════════════════════════════════════════════
# YOLO26 — Object Detection
# ══════════════════════════════════════════════════════════════
@app.post("/detect")
async def detect(
    file: UploadFile = File(...), 
    conf: float = Query(0.25),
    key_data: dict = Depends(get_authenticated_key),
):
    content = await validate_file(file)
    img = Image.open(io.BytesIO(content))
    results = yolo(img, conf=conf)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(x), 1) for x in box.xyxy[0]]
            })
    return JSONResponse({"detections": detections, "count": len(detections)})

# ══════════════════════════════════════════════════════════════
# Tesseract — OCR
# ══════════════════════════════════════════════════════════════
@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...), 
    lang: str = Query("eng"), 
    psm: int = Query(3),
    key_data: dict = Depends(get_authenticated_key),
):
    content = await validate_file(file)
    img = Image.open(io.BytesIO(content))
    cfg = f'--oem 3 --psm {psm}'
    text = pytesseract.image_to_string(img, lang=lang, config=cfg)
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    words = [{"text": w, "confidence": data['conf'][i],
              "bbox": [data['left'][i], data['top'][i],
                       data['left'][i]+data['width'][i], data['top'][i]+data['height'][i]]}
             for i, w in enumerate(data['text']) if w.strip()]
    return JSONResponse({"text": text.strip(), "words": words, "lang": lang})

@app.post("/ocr/preprocess")
async def ocr_preprocess(
    file: UploadFile = File(...),
    key_data: dict = Depends(get_authenticated_key),
):
    nparr = np.frombuffer(await validate_file(file), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, buf = cv2.imencode('.png', thresh)
    return Response(content=buf.tobytes(), media_type="image/png")

# ══════════════════════════════════════════════════════════════
# Whisper — Speech-to-Text
# ══════════════════════════════════════════════════════════════
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...), 
    language: str = Query(None),
    key_data: dict = Depends(get_authenticated_key),
):
    content = await validate_file(file)
    model = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or ".wav")[1], delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        opts = {"fp16": False}
        if language:
            opts["language"] = language
        result = model.transcribe(tmp.name, **opts)
    return JSONResponse({
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "segments": [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                     for s in result.get("segments", [])]
    })

@app.post("/detect-language")
async def detect_language(
    file: UploadFile = File(...),
    key_data: dict = Depends(get_authenticated_key),
):
    import whisper
    content = await validate_file(file)
    model = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        audio = whisper.load_audio(tmp.name)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    return JSONResponse({"languages": [{"code": k, "probability": round(v, 4)} for k, v in top5]})

# ══════════════════════════════════════════════════════════════
# rembg — Background Removal
# ══════════════════════════════════════════════════════════════
@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    key_data: dict = Depends(get_authenticated_key),
):
    from rembg import remove
    content = await validate_file(file)
    output = remove(content)
    return Response(content=output, media_type="image/png")

# ══════════════════════════════════════════════════════════════
# CLIP — Image Classification & Embeddings
# ══════════════════════════════════════════════════════════════
@app.post("/clip/classify")
async def clip_classify(
    file: UploadFile = File(...), 
    labels: str = Query("cat,dog,car,person,building"),
    key_data: dict = Depends(get_authenticated_key),
):
    import torch
    content = await validate_file(file)
    clip_model, preprocess, tokenizer = get_clip()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    image_input = preprocess(img).unsqueeze(0)
    label_list = [l.strip() for l in labels.split(",")]
    text_input = tokenizer(label_list)
    with torch.no_grad():
        img_feat = clip_model.encode_image(image_input)
        txt_feat = clip_model.encode_text(text_input)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).squeeze(0).tolist()
    results = sorted(zip(label_list, sim), key=lambda x: x[1], reverse=True)
    return JSONResponse({"classifications": [{"label": l, "score": round(s, 4)} for l, s in results]})

@app.post("/clip/embed")
async def clip_embed(
    file: UploadFile = File(...),
    key_data: dict = Depends(get_authenticated_key),
):
    import torch
    content = await validate_file(file)
    clip_model, preprocess, _ = get_clip()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    image_input = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = clip_model.encode_image(image_input)
        feat /= feat.norm(dim=-1, keepdim=True)
    return JSONResponse({"embedding": feat.squeeze(0).tolist(), "dimensions": feat.shape[1]})

# ══════════════════════════════════════════════════════════════
# OpenCV — Image Processing
# ══════════════════════════════════════════════════════════════
@app.post("/edges")
async def detect_edges(
    file: UploadFile = File(...),
    low: int = Query(50),
    high: int = Query(150),
    key_data: dict = Depends(get_authenticated_key),
):
    nparr = np.frombuffer(await validate_file(file), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, low, high)
    _, buf = cv2.imencode('.png', edges)
    return Response(content=buf.tobytes(), media_type="image/png")

@app.post("/faces")
async def detect_faces(
    file: UploadFile = File(...),
    key_data: dict = Depends(get_authenticated_key),
):
    nparr = np.frombuffer(await validate_file(file), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    results = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
    return JSONResponse({"faces": results, "count": len(results)})

@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Query(640),
    height: int = Query(480),
    key_data: dict = Depends(get_authenticated_key),
):
    nparr = np.frombuffer(await validate_file(file), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (width, height))
    _, buf = cv2.imencode('.png', resized)
    return Response(content=buf.tobytes(), media_type="image/png")

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    key_data: dict = Depends(get_authenticated_key),
):
    nparr = np.frombuffer(await validate_file(file), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return JSONResponse({
        "dimensions": {"width": w, "height": h, "channels": c},
        "brightness": round(float(np.mean(hsv[:,:,2])), 1),
        "saturation": round(float(np.mean(hsv[:,:,1])), 1),
        "mean_color_bgr": [round(float(x), 1) for x in cv2.mean(img)[:3]],
        "is_grayscale": bool(np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]))
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
