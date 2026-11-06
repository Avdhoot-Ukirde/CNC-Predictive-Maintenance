import numpy as np
import pytest

from src.feature_extraction import extract_features, extract_feature_rows
from src.preprocessing import segment_signal, validate_signal


def test_segment_signal_without_overlap():
    windows = segment_signal(np.arange(10, dtype=float), window_size=4)
    assert windows.shape == (2, 4)
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[1], [4, 5, 6, 7])


def test_segment_signal_with_overlap():
    windows = segment_signal(np.arange(10, dtype=float), window_size=4, overlap=0.5)
    assert windows.shape == (4, 4)
    np.testing.assert_array_equal(windows[1], [2, 3, 4, 5])


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.1])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        segment_signal(np.arange(10), 4, overlap=overlap)


def test_validate_signal_rejects_non_finite_values():
    with pytest.raises(ValueError):
        validate_signal(np.array([1.0, np.nan]))


def test_feature_extraction_known_sine_frequency():
    sampling_rate = 1000.0
    time = np.arange(1000) / sampling_rate
    signal = 2.0 * np.sin(2 * np.pi * 50.0 * time)
    features = extract_features(signal, sampling_rate)
    assert features["rms"] == pytest.approx(np.sqrt(2), rel=1e-2)
    assert features["peak_amplitude"] == pytest.approx(2.0, rel=1e-2)
    assert features["dominant_frequency_hz"] == pytest.approx(50.0, abs=1.0)


def test_constant_signal_has_finite_features():
    features = extract_features(np.ones(128), sampling_rate=1000)
    assert all(np.isfinite(value) for value in features.values())
    assert features["rms"] == pytest.approx(1.0)


def test_feature_rows_count():
    rows = extract_feature_rows(np.ones((3, 128)), sampling_rate=12000)
    assert len(rows) == 3
