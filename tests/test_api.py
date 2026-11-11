from fastapi.testclient import TestClient
import pandas as pd

from api import main

client = TestClient(main.app)


def test_health_reports_missing_model_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.joblib")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "model_not_trained"


def test_predict_returns_window_predictions(monkeypatch):
    monkeypatch.setattr(main, "get_model_bundle", lambda: {"model": object()})

    def fake_predict_signal(signal, sampling_rate, bundle, window_size, overlap):
        return pd.DataFrame({
            "window_index": [0, 1],
            "rms": [0.4, 0.5],
            "predicted_condition": ["normal", "inner_race"],
            "confidence": [0.91, 0.83],
        })

    monkeypatch.setattr(main, "predict_signal", fake_predict_signal)
    response = client.post("/predict", json={
        "signal": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        "sampling_rate_hz": 12000,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["window_count"] == 2
    assert body["predicted_condition"] == "normal"
    assert abs(body["mean_confidence"] - 0.87) < 1e-9


def test_predict_rejects_invalid_sampling_rate():
    response = client.post("/predict", json={
        "signal": [0.1] * 16,
        "sampling_rate_hz": 0,
    })
    assert response.status_code == 422
