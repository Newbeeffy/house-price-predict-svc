"""Pydantic request/response contracts for the API."""
from __future__ import annotations

from pydantic import BaseModel


class HouseFeatures(BaseModel):
    """A single house's features. All fields are float so integers or decimals
    (e.g. ``"bedrooms": 3`` and ``"bedrooms": 3.0``) both work."""

    square_footage: float
    bedrooms: float
    bathrooms: float
    year_built: float
    lot_size: float
    distance_to_city_center: float
    school_rating: float


class Metrics(BaseModel):
    r2: float
    rmse: float
    mae: float


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_names: list[str]
    coefficients: dict[str, float]
    intercept: float
    metrics: Metrics
    n_samples: int
    n_features: int
