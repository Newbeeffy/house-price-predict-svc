"""FastAPI application for the housing price prediction model service."""
from __future__ import annotations

from fastapi import FastAPI

from app import model
from app.schemas import HouseFeatures, ModelInfoResponse

app = FastAPI(title="Housing Price Prediction Model Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: HouseFeatures | list[HouseFeatures]) -> dict:
    """Predict housing prices.

    Accepts a single feature object or a list of them; returns a single
    ``prediction`` or a list ``predictions`` respectively.
    """
    if isinstance(payload, list):
        predictions = model.predict([h.model_dump() for h in payload])
        return {"predictions": predictions}

    prediction = model.predict([payload.model_dump()])[0]
    return {"prediction": prediction}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model coefficients (unstandardised scale) and performance metrics."""
    return model.get_info()
