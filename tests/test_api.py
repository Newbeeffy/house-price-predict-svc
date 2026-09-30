"""API tests for the three endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_HOUSE = {
    "square_footage": 1850,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1998,
    "lot_size": 7500,
    "distance_to_city_center": 5.6,
    "school_rating": 8.2,
}

FEATURE_NAMES = {
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_single():
    resp = client.post("/predict", json=SAMPLE_HOUSE)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    # The house above is ~$265k; allow generous tolerance.
    assert abs(data["prediction"] - 265000) < 50000


def test_predict_batch():
    payload = [SAMPLE_HOUSE, SAMPLE_HOUSE]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2
    assert all(isinstance(p, float) for p in data["predictions"])


def test_predict_rejects_bad_fields():
    resp = client.post("/predict", json={"square_footage": 1850})
    assert resp.status_code == 422


def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_type"] == "LinearRegression"
    assert set(data["feature_names"]) == FEATURE_NAMES
    assert set(data["coefficients"].keys()) == FEATURE_NAMES
    assert isinstance(data["intercept"], float)
    assert set(data["metrics"].keys()) == {"r2", "rmse", "mae"}
    assert data["n_samples"] > 0
    assert data["n_features"] == len(FEATURE_NAMES)
