"""
Test suite for AutonomyX Vision AI API
Run with: pytest tests/ -v
"""

import pytest
import io
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock


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


# Pytest fixture for client
@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


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


# Test metrics endpoint (no auth required)
def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200


# Test auth without key returns 401
def test_detect_without_auth(client):
    img_file = create_test_image_file()
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img_file, "image/jpeg")}
    )
    assert response.status_code == 401


# Test auth with invalid key returns 401
def test_detect_with_invalid_key(client):
    img_file = create_test_image_file()
    response = client.post(
        "/detect",
        files={"file": ("test.jpg", img_file, "image/jpeg")},
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == 401


# Test list_keys requires admin
def test_list_keys_requires_admin(client):
    response = client.get("/auth/api-keys")
    # Should require auth now
    assert response.status_code in [401, 403]


# Test delete_key requires admin
def test_delete_key_requires_admin(client):
    response = client.delete("/auth/api-keys/test-key")
    assert response.status_code in [401, 403]


# Test structured error format
def test_structured_error(client):
    response = client.post("/detect")
    assert response.status_code == 422  # Missing file param
    data = response.json()
    assert "detail" in data


# Test request_id is added to responses
def test_request_id_header(client):
    response = client.get("/health")
    # Should have request ID in header or body
    assert "x-request-id" in response.headers or "request_id" in response.json()