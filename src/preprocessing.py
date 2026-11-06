"""Signal validation and fixed-length window segmentation."""
from __future__ import annotations

import numpy as np


def validate_signal(signal: np.ndarray) -> np.ndarray:
    """Return a validated float64 one-dimensional signal."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if values.size == 0:
        raise ValueError("Signal must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Signal contains NaN or infinite values.")
    return values


def segment_signal(
    signal: np.ndarray,
    window_size: int,
    overlap: float = 0.0,
) -> np.ndarray:
    """Split a signal into complete fixed-size windows.

    overlap is the fraction shared by adjacent windows and must be in [0, 1).
    Incomplete trailing samples are intentionally discarded.
    """
    values = validate_signal(signal)
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0, 1).")
    if values.size < window_size:
        raise ValueError(
            f"Signal length ({values.size}) is shorter than window_size ({window_size})."
        )

    step = max(int(round(window_size * (1.0 - overlap))), 1)
    starts = range(0, values.size - window_size + 1, step)
    return np.stack([values[start:start + window_size] for start in starts])
