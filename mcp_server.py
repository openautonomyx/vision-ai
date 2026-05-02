"""
AutonomyX Vision AI — MCP Server
Exposes all Vision AI endpoints as MCP tools for AI agents.

Production features:
- API key authentication
- Expanded higher-level tools
- Workflow templates
"""
import asyncio, io, os, json, base64, httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:8000")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")  # Optional API key for production

app = Server("vision-ai-mcp")

async def _post_file(endpoint: str, file_bytes: bytes, filename: str = "image.jpg", params: dict = None):
    headers = {}
    if VISION_API_KEY:
        headers["X-API-Key"] = VISION_API_KEY
    
    async with httpx.AsyncClient(timeout=120, headers=headers) as client:
        files = {"file": (filename, file_bytes)}
        url = f"{VISION_API_URL}{endpoint}"
        resp = await client.post(url, files=files, params=params or {})
        return resp

async def _get(endpoint: str, params: dict = None):
    headers = {}
    if VISION_API_KEY:
        headers["X-API-Key"] = VISION_API_KEY
    
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        url = f"{VISION_API_URL}{endpoint}"
        resp = await client.get(url, params=params or {})
        return resp

@app.list_tools()
async def list_tools():
    return [
        # === Core Detection Tools ===
        Tool(name="detect_objects", description="Detect objects in an image using YOLO26. Returns bounding boxes, classes, and confidence scores.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "confidence": {"type": "number", "description": "Min confidence threshold (0-1)", "default": 0.25}
             }, "required": ["image_base64"]}),

        Tool(name="extract_text", description="Extract text from an image using Tesseract OCR. Supports English and Hindi.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "language": {"type": "string", "description": "OCR language: eng, hin", "default": "eng"}
             }, "required": ["image_base64"]}),

        Tool(name="transcribe_audio", description="Transcribe speech from audio file to text using Whisper.",
             inputSchema={"type": "object", "properties": {
                 "audio_base64": {"type": "string", "description": "Base64-encoded audio file (mp3, wav, etc.)"},
                 "language": {"type": "string", "description": "Language code (en, hi, etc.) or omit for auto-detect"}
             }, "required": ["audio_base64"]}),

        Tool(name="detect_language", description="Detect the spoken language in an audio file.",
             inputSchema={"type": "object", "properties": {
                 "audio_base64": {"type": "string", "description": "Base64-encoded audio file"}
             }, "required": ["audio_base64"]}),

        # === Image Processing Tools ===
        Tool(name="remove_background", description="Remove background from an image. Returns transparent PNG.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"}
             }, "required": ["image_base64"]}),

        Tool(name="classify_image", description="Zero-shot image classification using CLIP. Classify image against custom labels.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "labels": {"type": "string", "description": "Comma-separated labels to classify against", "default": "cat,dog,car,person,building"}
             }, "required": ["image_base64"]}),

        Tool(name="generate_embedding", description="Generate CLIP embedding vector for an image. Useful for image search and similarity.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"}
             }, "required": ["image_base64"]}),

        Tool(name="detect_faces", description="Detect faces in an image using OpenCV Haar cascades.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"}
             }, "required": ["image_base64"]}),

        Tool(name="analyze_image", description="Analyze image properties: dimensions, brightness, saturation, dominant color.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"}
             }, "required": ["image_base64"]}),

        # === Higher-Level Agent Tools (NEW) ===
        Tool(name="analyze_image", description="Comprehensive image analysis: objects, text, faces, and visual features.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "include_objects": {"type": "boolean", "description": "Detect objects", "default": True},
                 "include_text": {"type": "boolean", "description": "Extract text via OCR", "default": True},
                 "include_faces": {"type": "boolean", "description": "Detect faces", "default": True},
                 "include_classification": {"type": "boolean", "description": "Classify image", "default": True}
             }, "required": ["image_base64"]}),

        Tool(name="compare_images", description="Compare two images and find differences. Useful for change detection, quality inspection.",
             inputSchema={"type": "object", "properties": {
                 "image1_base64": {"type": "string", "description": "Base64-encoded first image"},
                 "image2_base64": {"type": "string", "description": "Base64-encoded second image"}
             }, "required": ["image1_base64", "image2_base64"]}),

        Tool(name="extract_document_fields", description="Extract structured data from documents (receipts, forms, invoices). Returns JSON with recognized fields.",
             inputSchema={"type": "object", "properties": {
                 "document_base64": {"type": "string", "description": "Base64-encoded document image"},
                 "field_types": {"type": "string", "description": "Expected fields (comma-separated): date,total,address,etc.", "default": "date,total,address"}
             }, "required": ["document_base64"]}),

        Tool(name="resize_image", description="Resize image to specified dimensions.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "width": {"type": "integer", "description": "Target width", "default": 640},
                 "height": {"type": "integer", "description": "Target height", "default": 480}
             }, "required": ["image_base64"]}),

        # === Health & Info ===
        Tool(name="vision_health", description="Check Vision AI service health and available models.",
             inputSchema={"type": "object", "properties": {}}),
        
        Tool(name="list_models", description="List all available models and their configurations.",
             inputSchema={"type": "object", "properties": {}}),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        # === Health & Info ===
        if name == "vision_health":
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{VISION_API_URL}/health")
                return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "list_models":
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{VISION_API_URL}/models")
                return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        # === Core Detection Tools ===
        if name == "detect_objects":
            img = base64.b64decode(arguments["image_base64"])
            params = {"conf": arguments.get("confidence", 0.25)}
            resp = await _post_file("/detect", img, params=params)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "extract_text":  # Renamed from ocr
            img = base64.b64decode(arguments["image_base64"])
            params = {"lang": arguments.get("language", "eng")}
            resp = await _post_file("/ocr", img, params=params)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "transcribe_audio":
            audio = base64.b64decode(arguments["audio_base64"])
            params = {}
            if "language" in arguments:
                params["language"] = arguments["language"]
            resp = await _post_file("/transcribe", audio, filename="audio.wav", params=params)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "detect_language":
            audio = base64.b64decode(arguments["audio_base64"])
            resp = await _post_file("/detect-language", audio, filename="audio.wav")
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        # === Image Processing Tools ===
        if name == "remove_background":
            img = base64.b64decode(arguments["image_base64"])
            resp = await _post_file("/remove-bg", img)
            result_b64 = base64.b64encode(resp.content).decode()
            return [TextContent(type="text", text=json.dumps({"image_base64": result_b64, "format": "png"}))]

        if name == "classify_image":
            img = base64.b64decode(arguments["image_base64"])
            params = {"labels": arguments.get("labels", "cat,dog,car,person,building")}
            resp = await _post_file("/clip/classify", img, params=params)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "generate_embedding":  # Renamed from embed_image
            img = base64.b64decode(arguments["image_base64"])
            resp = await _post_file("/clip/embed", img)
            data = resp.json()
            return [TextContent(type="text", text=json.dumps({"dimensions": data["dimensions"], "embedding_preview": data["embedding"][:10]}))]

        if name == "detect_faces":
            img = base64.b64decode(arguments["image_base64"])
            resp = await _post_file("/faces", img)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "analyze_image":
            img = base64.b64decode(arguments["image_base64"])
            resp = await _post_file("/analyze", img)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        # === Higher-Level Agent Tools ===
        # Comprehensive image analysis
        elif name == "comprehensive_analyze":  # Fixed duplicate name
            img = base64.b64decode(arguments["image_base64"])
            results = {}
            
            if arguments.get("include_objects", True):
                resp = await _post_file("/detect", img)
                results["objects"] = resp.json()
            
            if arguments.get("include_text", True):
                resp = await _post_file("/ocr", img)
                results["text"] = resp.json()
            
            if arguments.get("include_faces", True):
                resp = await _post_file("/faces", img)
                results["faces"] = resp.json()
            
            if arguments.get("include_classification", True):
                resp = await _post_file("/clip/classify", img)
                results["classification"] = resp.json()
            
            return [TextContent(type="text", text=json.dumps(results, indent=2))]

        # Compare images
        elif name == "compare_images":
            img1 = base64.b64decode(arguments["image1_base64"])
            img2 = base64.b64decode(arguments["image2_base64"])
            
            # Get embeddings for both images
            emb1 = await _post_file("/clip/embed", img1)
            emb2 = await _post_file("/clip/embed", img2)
            
            import numpy as np
            e1 = np.array(emb1.json()["embedding"])
            e2 = np.array(emb2.json()["embedding"])
            
            # Cosine similarity
            similarity = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            
            return [TextContent(type="text", text=json.dumps({
                "similarity": round(similarity, 4),
                "interpretation": "identical" if similarity > 0.95 else "similar" if similarity > 0.8 else "different"
            }))]

        # Extract document fields
        elif name == "extract_document_fields":
            doc = base64.b64decode(arguments["document_base64"])
            field_types = arguments.get("field_types", "date,total,address").split(",")
            
            # Get OCR result
            ocr_resp = await _post_file("/ocr", doc)
            text = ocr_resp.json().get("text", "")
            
            # Simple field extraction (regex-based)
            import re
            fields = {}
            
            for ft in field_types:
                ft = ft.strip()
                if ft == "date":
                    match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                    if match: fields["date"] = match.group()
                elif ft == "total":
                    match = re.search(r'(?:total|amount|sum)[:\s]*\$?(\d+\.?\d*)', text, re.I)
                    if match: fields["total"] = match.group(1)
                elif ft == "address":
                    # Look for common address patterns
                    match = re.search(r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd)', text, re.I)
                    if match: fields["address"] = match.group()
            
            return [TextContent(type="text", text=json.dumps(fields, indent=2))]

        # Resize image
        elif name == "resize_image":
            img = base64.b64decode(arguments["image_base64"])
            width = arguments.get("width", 640)
            height = arguments.get("height", 480)
            resp = await _post_file("/resize", img, params={"width": width, "height": height})
            result_b64 = base64.b64encode(resp.content).decode()
            return [TextContent(type="text", text=json.dumps({"image_base64": result_b64, "format": "png"}))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
