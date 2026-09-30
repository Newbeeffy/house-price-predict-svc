"""Model serving layer: loads the trained scaler + model once at import time."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_scaler = joblib.load(MODELS_DIR / "scaler.joblib")
_model = joblib.load(MODELS_DIR / "model.joblib")

with open(MODELS_DIR / "metrics.json", encoding="utf-8") as f:
    _info = json.load(f)


def predict(features: list[dict[str, float]]) -> list[float]:
    """Predict prices for one or more houses.

    ``features`` is a list of dicts keyed by feature name. The feature order is
    taken from ``metrics.json`` (not the dict insertion order) so the columns
    line up with what the scaler/model were fitted on.
    """
    names = _info["feature_names"]
    X = np.array([[f[name] for name in names] for f in features], dtype=float)
    X_scaled = _scaler.transform(X)
    return _model.predict(X_scaled).tolist()


def get_info() -> dict:
    return _info
