import numpy as np
import pandas as pd
from src.inference import predict_signal

class DummyClassifier:
    def predict(self, frame):
        assert isinstance(frame, pd.DataFrame)
        return np.array(["normal"] * len(frame))
    def predict_proba(self, frame):
        return np.tile([0.9, 0.1], (len(frame), 1))

def test_predict_signal_returns_one_row_per_window():
    bundle = {
        "model": DummyClassifier(),
        "feature_columns": [
            "mean", "std", "rms", "peak_amplitude", "peak_to_peak",
            "skewness", "kurtosis_pearson", "crest_factor", "shape_factor",
            "dominant_frequency_hz", "spectral_centroid_hz",
        ],
        "class_names": ["normal", "fault"],
    }
    signal = np.sin(np.linspace(0, 40 * np.pi, 4096))
    result = predict_signal(signal, 1000, bundle, window_size=2048, overlap=0)
    assert len(result) == 2
    assert result["predicted_condition"].tolist() == ["normal", "normal"]
    assert np.allclose(result["confidence"], 0.9)
