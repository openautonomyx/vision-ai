"""
AutonomyX Vision AI — MCP Server
Exposes all Vision AI endpoints as MCP tools for AI agents.
"""
import asyncio, io, os, json, base64, httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

VISION_API_URL = os.environ.get("VISION_API_URL", "http://localhost:8000")

app = Server("vision-ai-mcp")

async def _post_file(endpoint: str, file_bytes: bytes, filename: str = "image.jpg", params: dict = None):
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (filename, file_bytes)}
        url = f"{VISION_API_URL}{endpoint}"
        resp = await client.post(url, files=files, params=params or {})
        return resp

@app.list_tools()
async def list_tools():
    return [
        Tool(name="detect_objects", description="Detect objects in an image using YOLO26. Returns bounding boxes, classes, and confidence scores.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "confidence": {"type": "number", "description": "Min confidence threshold (0-1)", "default": 0.25}
             }, "required": ["image_base64"]}),

        Tool(name="ocr", description="Extract text from an image using Tesseract OCR. Supports English and Hindi.",
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

        Tool(name="remove_background", description="Remove background from an image. Returns transparent PNG.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"}
             }, "required": ["image_base64"]}),

        Tool(name="classify_image", description="Zero-shot image classification using CLIP. Classify image against custom labels.",
             inputSchema={"type": "object", "properties": {
                 "image_base64": {"type": "string", "description": "Base64-encoded image"},
                 "labels": {"type": "string", "description": "Comma-separated labels to classify against", "default": "cat,dog,car,person,building"}
             }, "required": ["image_base64"]}),

        Tool(name="embed_image", description="Generate CLIP embedding vector for an image. Useful for image search and similarity.",
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

        Tool(name="vision_health", description="Check Vision AI service health and available models.",
             inputSchema={"type": "object", "properties": {}}),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "vision_health":
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{VISION_API_URL}/health")
                return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "detect_objects":
            img = base64.b64decode(arguments["image_base64"])
            params = {"conf": arguments.get("confidence", 0.25)}
            resp = await _post_file("/detect", img, params=params)
            return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        if name == "ocr":
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

        if name == "embed_image":
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

        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
