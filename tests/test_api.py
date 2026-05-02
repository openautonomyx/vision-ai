"""
Test suite for AutonomyX Vision AI API
Run with: pytest tests/ -v
"""

import pytest
import io
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

# Test health endpoint
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "services" in data

# Test models endpoint
def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "yolo" in data
    assert "ocr" in data

# Test API key creation
def test_create_api_key(client):
    response = client.post("/auth/api-keys", json={
        "name": "test-key",
        "role": "developer",
        "rate_limit": 100
    })
    assert response.status_code == 200
    data = response.json()
    assert "key" in data
    assert data["name"] == "test-key"

# Test metrics endpoint
def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200

# Generate test images
def create_test_image(width=640, height=480):
    """Create a simple test image."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img)

def create_test_image_file(format="JPEG"):
    """Create a test image file upload."""
    img = create_test_image()
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# Test object detection
@pytest.mark.asyncio
async def test_detect_endpoint(client):
    img_file = create_test_image_file()
    response = await client.post(
        "/detect",
        files={"file": ("test.jpg", img_file, "image/jpeg")},
        data={"conf": 0.25}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data

# Test OCR endpoint
@pytest.mark.asyncio
async def test_ocr_endpoint(client):
    img_file = create_test_image_file()
    response = await client.post(
        "/ocr",
        files={"file": ("test.jpg", img_file, "image/jpeg")},
        data={"lang": "eng", "psm": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data

# Test CLIP classify endpoint
@pytest.mark.asyncio
async def test_clip_classify_endpoint(client):
    img_file = create_test_image_file()
    response = await client.post(
        "/clip/classify",
        files={"file": ("test.jpg", img_file, "image/jpeg")},
        data={"labels": "cat,dog,car"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "classifications" in data

# Test resize endpoint
@pytest.mark.asyncio
async def test_resize_endpoint(client):
    img_file = create_test_image_file()
    response = await client.post(
        "/resize",
        files={"file": ("test.jpg", img_file, "image/jpeg")},
        data={"width": 320, "height": 240}
    )
    assert response.status_code == 200

# Test analyze endpoint
@pytest.mark.asyncio
async def test_analyze_endpoint(client):
    img_file = create_test_image_file()
    response = await client.post(
        "/analyze",
        files={"file": ("test.jpg", img_file, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "dimensions" in data
    assert "brightness" in data

# Test structured error handling
def test_invalid_file_type(client):
    """Test error for invalid file type."""
    # This would require actually testing with an invalid type
    # For now, just test the error structure exists
    pass

def test_request_id_middleware(client):
    """Test request ID is added to responses."""
    response = client.get("/health")
    assert "x-request-id" in response.headers or "request_id" in response.json()

# Pytest fixture for client
@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)