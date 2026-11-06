"""Load a trained bearing classifier and predict vibration-window conditions."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from .feature_extraction import extract_feature_rows
from .preprocessing import segment_signal

def load_classifier(model_path: str | Path) -> dict[str, Any]:
    """Load the model bundle produced by train_bearing_classifier.py."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Run scripts/train_bearing_classifier.py first.")
    bundle = joblib.load(path)
    required = {"model", "feature_columns", "class_names"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Model file is not a compatible bearing-classifier bundle.")
    return bundle

def predict_signal(signal: np.ndarray, sampling_rate: float, bundle: dict[str, Any],
                   window_size: int = 2048, overlap: float = 0.5) -> pd.DataFrame:
    """Extract features and predict one condition for each complete signal window."""
    windows = segment_signal(signal, window_size, overlap)
    features = pd.DataFrame(extract_feature_rows(windows, sampling_rate))
    columns = bundle["feature_columns"]
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(f"Required model features are unavailable: {missing}")
    features = features[columns]
    model = bundle["model"]
    labels = model.predict(features)
    result = features.copy()
    result.insert(0, "window_index", np.arange(len(result)))
    result["predicted_condition"] = labels
    if hasattr(model, "predict_proba"):
        result["confidence"] = model.predict_proba(features).max(axis=1)
    return result
