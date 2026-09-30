"""Offline training script. Run once to produce the model artifacts packaged
into the serving image: scaler.joblib, model.joblib, metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house-price-dataset.csv"
MODELS_DIR = BASE_DIR / "models"

# Feature columns (id is dropped; price is the target).
FEATURE_COLUMNS = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]
TARGET_COLUMN = "price"


def load_data() -> pd.DataFrame:
    # utf-8-sig strips the BOM from the header (first column is "id").
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def main() -> None:
    df = load_data().drop(columns=["id"])
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the scaler on training data only, to avoid data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }

    # Convert standardised coefficients back to raw-feature scale:
    #   y = w_std · ((X - μ) / σ) + b = (w_std / σ) · X + (b - Σ w_std·μ/σ)
    coef_unstd = model.coef_ / scaler.scale_
    intercept_unstd = model.intercept_ - float(
        np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    )

    model_info = {
        "model_type": "LinearRegression",
        "feature_names": FEATURE_COLUMNS,
        "coefficients": {
            name: float(c) for name, c in zip(FEATURE_COLUMNS, coef_unstd)
        },
        "intercept": float(intercept_unstd),
        "metrics": metrics,
        "n_samples": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(model, MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("Training complete.")
    print(f"  n_samples  = {model_info['n_samples']}")
    print(f"  r2         = {metrics['r2']:.4f}")
    print(f"  rmse       = {metrics['rmse']:.4f}")
    print(f"  mae        = {metrics['mae']:.4f}")
    print(f"  intercept  = {intercept_unstd:.4f}")
    print(f"  artifacts  -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
