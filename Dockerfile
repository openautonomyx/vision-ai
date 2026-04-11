FROM ultralytics/ultralytics:latest-cpu

RUN apt-get update &&     apt-get install -y --no-install-recommends       tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin       ffmpeg curl &&     rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir     fastapi uvicorn python-multipart pillow     opencv-python-headless pytesseract     rembg[cpu]     open-clip-torch     openai-whisper     mcp httpx

COPY app.py /app/app.py
COPY mcp_server.py /app/mcp_server.py
WORKDIR /app
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python3", "app.py"]
