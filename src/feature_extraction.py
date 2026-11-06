"""Time-domain and frequency-domain features for vibration windows."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .preprocessing import validate_signal


def extract_features(signal: np.ndarray, sampling_rate: float) -> dict[str, float]:
    """Extract interpretable features from one vibration window.

    Kurtosis uses Pearson's definition (normal distribution is approximately 3).
    FFT magnitudes use a Hann-windowed, mean-centred signal.
    Frequency outputs are in Hz and require the true sampling rate.
    """
    x = validate_signal(signal)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be a positive finite number.")

    centred = x - np.mean(x)
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    mean_abs = float(np.mean(np.abs(x)))

    if std == 0.0:
        skewness, kurt = 0.0, 0.0
    else:
        skewness = float(skew(x, bias=False)) if x.size > 2 else 0.0
        kurt = float(kurtosis(x, fisher=False, bias=False)) if x.size > 3 else 0.0

    crest_factor = peak / rms if rms > 0 else 0.0
    shape_factor = rms / mean_abs if mean_abs > 0 else 0.0

    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft(centred * window))
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sampling_rate)
    if spectrum.size > 1:
        dominant_index = int(np.argmax(spectrum[1:]) + 1)
        dominant_frequency = float(frequencies[dominant_index])
    else:
        dominant_frequency = 0.0

    spectral_sum = float(np.sum(spectrum))
    spectral_centroid = (
        float(np.sum(frequencies * spectrum) / spectral_sum)
        if spectral_sum > 0 else 0.0
    )

    return {
        "mean": float(np.mean(x)),
        "std": std,
        "rms": rms,
        "peak_amplitude": peak,
        "peak_to_peak": float(np.ptp(x)),
        "skewness": skewness,
        "kurtosis_pearson": kurt,
        "crest_factor": float(crest_factor),
        "shape_factor": float(shape_factor),
        "dominant_frequency_hz": dominant_frequency,
        "spectral_centroid_hz": spectral_centroid,
    }


def extract_feature_rows(
    windows: np.ndarray,
    sampling_rate: float,
) -> list[dict[str, float]]:
    """Extract one feature dictionary per row in a two-dimensional windows array."""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("windows must be a non-empty 2-D array.")
    return [extract_features(row, sampling_rate) for row in values]
