"""REST API for the trained CWRU bearing-condition classifier."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_classifier, predict_signal

MODEL_PATH = Path(os.getenv("MODEL_PATH", "reports/classifier/bearing_random_forest.joblib"))
app = FastAPI(title="CNC Predictive Maintenance API", description="Bearing-condition inference from vibration signal windows.", version="1.0.0")


class PredictionRequest(BaseModel):
    signal: list[float] = Field(min_length=8, description="Raw vibration samples from one recording")
    sampling_rate_hz: float = Field(gt=0, description="Verified signal sampling frequency in Hz")
    window_size: int = Field(default=2048, ge=8)
    overlap: float = Field(default=0.5, ge=0, lt=1)


@lru_cache(maxsize=1)
def get_model_bundle():
    """Load the trained model once per API process."""
    return load_classifier(MODEL_PATH)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"name": "CNC Predictive Maintenance API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["health"])
def health() -> dict[str, object]:
    return {"status": "ready" if MODEL_PATH.is_file() else "model_not_trained", "model_path": str(MODEL_PATH), "model_exists": MODEL_PATH.is_file()}


@app.post("/predict", tags=["inference"])
def predict(request: PredictionRequest) -> dict[str, object]:
    """Predict bearing condition for each complete window in a raw signal."""
    try:
        bundle = get_model_bundle()
        predictions = predict_signal(np.asarray(request.signal, dtype=np.float64), request.sampling_rate_hz, bundle, window_size=request.window_size, overlap=request.overlap)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if predictions.empty:
        raise HTTPException(status_code=422, detail="Signal did not produce any complete analysis windows.")

    counts = predictions["predicted_condition"].value_counts()
    response: dict[str, object] = {
        "predicted_condition": str(counts.index[0]),
        "window_count": int(len(predictions)),
        "condition_counts": {str(key): int(value) for key, value in counts.items()},
        "windows": predictions.to_dict(orient="records"),
    }
    if "confidence" in predictions:
        response["mean_confidence"] = float(predictions["confidence"].mean())
    return response
