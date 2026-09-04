import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Produce Freshness Scanner API"


def test_predict_fresh():
    response = client.post("/predict", json={"color": 8.2, "texture": 7.1})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FRESH"
    assert 0.0 <= data["confidence"] <= 100.0


def test_predict_rotten():
    response = client.post("/predict", json={"color": 2.1, "texture": 1.4})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ROTTEN"
    assert 0.0 <= data["confidence"] <= 100.0


def test_predict_returns_valid_status():
    response = client.post("/predict", json={"color": 5.0, "texture": 4.5})
    assert response.status_code == 200
    assert response.json()["status"] in ["FRESH", "ROTTEN"]


def test_predict_missing_color():
    response = client.post("/predict", json={"texture": 7.1})
    assert response.status_code == 422


def test_predict_missing_texture():
    response = client.post("/predict", json={"color": 8.2})
    assert response.status_code == 422


def test_predict_missing_both_fields():
    response = client.post("/predict", json={})
    assert response.status_code == 422
