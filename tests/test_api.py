import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_jpeg_bytes():
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_single_shot_success(client, dummy_jpeg_bytes):
    response = client.post(
        "/predict-image",
        files={"file": ("test_apple.jpg", dummy_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_apple.jpg"
    assert data["status"] in ["FRESH", "ROTTEN"]
    assert isinstance(data["confidence"], float)


def test_predict_single_shot_invalid_filetype(client):
    response = client.post(
        "/predict-image",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_single_shot_payload_too_large(client):
    huge_bytes = b"0" * (2 * 1024 * 1024 + 1)
    response = client.post(
        "/predict-image",
        files={"file": ("huge.jpg", huge_bytes, "image/jpeg")},
    )
    assert response.status_code == 413


def test_websocket_predict_stream(client, dummy_jpeg_bytes):
    with client.websocket_connect("/ws/predict-stream") as websocket:
        websocket.send_bytes(dummy_jpeg_bytes)
        data = websocket.receive_json()
        assert data["status"] in ["FRESH", "ROTTEN"]
        assert isinstance(data["confidence"], float)
