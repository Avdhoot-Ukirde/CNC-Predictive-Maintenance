"""Load vibration signals from CWRU MATLAB files or simple CSV files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

_CWRU_SUFFIXES = ("_DE_time", "_FE_time", "_BA_time")


def discover_mat_signals(file_path: str | Path) -> list[str]:
    """Return likely vibration-channel keys in a CWRU-style .mat file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Expected a .mat file, got: {path.suffix}")
    contents = loadmat(path, squeeze_me=True)
    keys = [
        key for key in contents
        if not key.startswith("__")
        and any(key.endswith(suffix) for suffix in _CWRU_SUFFIXES)
    ]
    return sorted(keys)


def load_cwru_signal(
    file_path: str | Path,
    signal_key: str | None = None,
) -> np.ndarray:
    """Load one one-dimensional vibration channel from a CWRU .mat file.

    If signal_key is omitted, the file must contain exactly one recognizable
    DE/FE/BA channel. Ambiguous files require explicit selection.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Expected a .mat file, got: {path.suffix}")
    contents = loadmat(path, squeeze_me=True)
    available = [
        key for key in contents
        if not key.startswith("__")
        and any(key.endswith(suffix) for suffix in _CWRU_SUFFIXES)
    ]

    if signal_key is None:
        if len(available) != 1:
            raise ValueError(
                "Could not select a unique vibration channel. "
                f"Available channels: {sorted(available)}. Pass signal_key explicitly."
            )
        signal_key = available[0]
    elif signal_key not in available:
        raise KeyError(
            f"Signal key {signal_key!r} not found. Available channels: {sorted(available)}"
        )

    signal = np.asarray(contents[signal_key], dtype=np.float64).squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"Selected channel {signal_key!r} is not a non-empty 1-D signal.")
    if not np.isfinite(signal).all():
        raise ValueError(f"Selected channel {signal_key!r} contains NaN or infinite values.")
    return signal


def load_csv_signal(file_path: str | Path, column: str | None = None) -> np.ndarray:
    """Load a numeric vibration column from a CSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    frame = pd.read_csv(path)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()

    if column is None:
        if len(numeric_columns) != 1:
            raise ValueError(
                "CSV must have exactly one numeric column when column is omitted. "
                f"Numeric columns: {numeric_columns}"
            )
        column = numeric_columns[0]
    if column not in frame.columns:
        raise KeyError(f"Column {column!r} not found. Available columns: {list(frame.columns)}")

    signal = frame[column].to_numpy(dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Selected CSV column is empty or not one-dimensional.")
    if not np.isfinite(signal).all():
        raise ValueError(f"Column {column!r} contains NaN or infinite values.")
    return signal


def load_signal(
    file_path: str | Path,
    signal_key: str | None = None,
    csv_column: str | None = None,
) -> np.ndarray:
    """Dispatch to the supported loader based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".mat":
        return load_cwru_signal(file_path, signal_key=signal_key)
    if suffix == ".csv":
        return load_csv_signal(file_path, column=csv_column)
    raise ValueError(f"Unsupported file type {suffix!r}. Use .mat or .csv.")
