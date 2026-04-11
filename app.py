"""
AutonomyX Vision AI — Unified API
YOLO26 + OpenCV + Tesseract OCR + CLIP + Whisper + Background Remover
"""
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse, Response
import uvicorn, io, os, cv2, tempfile
import numpy as np
from PIL import Image
import pytesseract

app = FastAPI(
    title="AutonomyX Vision AI",
    version="1.0",
    description="Unified Vision AI: object detection, OCR, speech-to-text, image classification, background removal"
)

# ── Eager-load lightweight models ─────────────────────────────
model_name = os.environ.get("YOLO_MODEL", "yolo26n.pt")
yolo = YOLO(model_name)

# ── Lazy-load heavy models ────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════
# Health & Info
# ══════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status": "ok",
        "services": {
            "yolo": model_name,
            "opencv": cv2.__version__,
            "tesseract": pytesseract.get_tesseract_version().public,
            "clip": "ViT-B-32 (lazy)",
            "whisper": os.environ.get("WHISPER_MODEL", "base") + " (lazy)",
            "rembg": "u2net (lazy)"
        }
    }

@app.get("/models")
def list_models():
    return {
        "yolo": ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        "opencv": ["face_detection", "edge_detection", "resize", "analyze"],
        "ocr": {"engine": "tesseract", "languages": ["eng", "hin"]},
        "clip": {"model": "ViT-B-32", "tasks": ["classify", "embed"]},
        "whisper": {"models": ["tiny", "base", "small", "medium"], "tasks": ["transcribe", "detect_language"]},
        "rembg": {"task": "background_removal"}
    }

# ══════════════════════════════════════════════════════════════
# YOLO26 — Object Detection
# ══════════════════════════════════════════════════════════════
@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.25)):
    img = Image.open(io.BytesIO(await file.read()))
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
async def ocr(file: UploadFile = File(...), lang: str = Query("eng"), psm: int = Query(3)):
    img = Image.open(io.BytesIO(await file.read()))
    cfg = f'--oem 3 --psm {psm}'
    text = pytesseract.image_to_string(img, lang=lang, config=cfg)
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    words = [{"text": w, "confidence": data['conf'][i],
              "bbox": [data['left'][i], data['top'][i],
                       data['left'][i]+data['width'][i], data['top'][i]+data['height'][i]]}
             for i, w in enumerate(data['text']) if w.strip()]
    return JSONResponse({"text": text.strip(), "words": words, "lang": lang})

@app.post("/ocr/preprocess")
async def ocr_preprocess(file: UploadFile = File(...)):
    nparr = np.frombuffer(await file.read(), np.uint8)
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
async def transcribe(file: UploadFile = File(...), language: str = Query(None)):
    model = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or ".wav")[1], delete=True) as tmp:
        tmp.write(await file.read())
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
async def detect_language(file: UploadFile = File(...)):
    import whisper
    model = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(await file.read())
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
async def remove_background(file: UploadFile = File(...)):
    from rembg import remove
    output = remove(await file.read())
    return Response(content=output, media_type="image/png")

# ══════════════════════════════════════════════════════════════
# CLIP — Image Classification & Embeddings
# ══════════════════════════════════════════════════════════════
@app.post("/clip/classify")
async def clip_classify(file: UploadFile = File(...), labels: str = Query("cat,dog,car,person,building")):
    import torch
    clip_model, preprocess, tokenizer = get_clip()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
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
async def clip_embed(file: UploadFile = File(...)):
    import torch
    clip_model, preprocess, _ = get_clip()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_input = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = clip_model.encode_image(image_input)
        feat /= feat.norm(dim=-1, keepdim=True)
    return JSONResponse({"embedding": feat.squeeze(0).tolist(), "dimensions": feat.shape[1]})

# ══════════════════════════════════════════════════════════════
# OpenCV — Image Processing
# ══════════════════════════════════════════════════════════════
@app.post("/edges")
async def detect_edges(file: UploadFile = File(...), low: int = Query(50), high: int = Query(150)):
    nparr = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, low, high)
    _, buf = cv2.imencode('.png', edges)
    return Response(content=buf.tobytes(), media_type="image/png")

@app.post("/faces")
async def detect_faces(file: UploadFile = File(...)):
    nparr = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    results = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
    return JSONResponse({"faces": results, "count": len(results)})

@app.post("/resize")
async def resize_image(file: UploadFile = File(...), width: int = Query(640), height: int = Query(480)):
    nparr = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (width, height))
    _, buf = cv2.imencode('.png', resized)
    return Response(content=buf.tobytes(), media_type="image/png")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    nparr = np.frombuffer(await file.read(), np.uint8)
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
