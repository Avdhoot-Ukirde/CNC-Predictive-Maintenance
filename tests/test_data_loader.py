import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.data_loader import discover_mat_signals, load_csv_signal, load_cwru_signal, load_signal


def test_load_csv_signal(tmp_path):
    path = tmp_path / "vibration.csv"
    pd.DataFrame({"acceleration": [0.1, 0.2, 0.3]}).to_csv(path, index=False)
    np.testing.assert_allclose(load_csv_signal(path), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(load_signal(path), [0.1, 0.2, 0.3])


def test_load_mat_signal_and_discover_channel(tmp_path):
    path = tmp_path / "sample.mat"
    expected = np.array([1.0, -1.0, 0.5])
    savemat(path, {"X097_DE_time": expected, "metadata": np.array([1])})
    assert discover_mat_signals(path) == ["X097_DE_time"]
    np.testing.assert_allclose(load_cwru_signal(path), expected)
    np.testing.assert_allclose(load_signal(path), expected)


def test_mat_loader_requires_channel_when_ambiguous(tmp_path):
    path = tmp_path / "multi.mat"
    savemat(path, {
        "X097_DE_time": np.array([1.0, 2.0]),
        "X097_FE_time": np.array([3.0, 4.0]),
    })
    with pytest.raises(ValueError, match="unique vibration channel"):
        load_cwru_signal(path)
    np.testing.assert_allclose(
        load_cwru_signal(path, signal_key="X097_FE_time"),
        [3.0, 4.0],
    )


def test_csv_loader_rejects_ambiguous_numeric_columns(tmp_path):
    path = tmp_path / "multi.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one numeric column"):
        load_csv_signal(path)


def test_loader_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "signal.txt"
    path.write_text("1,2,3")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_signal(path)
