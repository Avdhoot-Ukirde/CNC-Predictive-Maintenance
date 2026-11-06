"""Create a first-pass vibration analysis report for a CWRU recording.

Example:
    python scripts/analyze_cwru.py data/raw/cwru/100.mat \
        --signal-key X100_DE_time --sampling-rate 12000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Allow running this file directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_signal
from src.feature_extraction import extract_features, extract_feature_rows
from src.preprocessing import segment_signal


def read_rpm(file_path: Path) -> float | None:
    """Read the RPM scalar when the input is a CWRU MATLAB file."""
    if file_path.suffix.lower() != ".mat":
        return None
    contents = loadmat(file_path, squeeze_me=True)
    rpm_keys = [key for key in contents if key.endswith("RPM")]
    if not rpm_keys:
        return None
    return float(np.asarray(contents[rpm_keys[0]]).squeeze())


def save_time_plot(signal: np.ndarray, sampling_rate: float, output: Path) -> None:
    """Save a readable overview of the first 0.1 seconds of the signal."""
    sample_count = min(signal.size, max(1, int(0.1 * sampling_rate)))
    time = np.arange(sample_count) / sampling_rate
    plt.figure(figsize=(11, 4))
    plt.plot(time, signal[:sample_count], linewidth=0.75)
    plt.title("Drive-end vibration signal (first 0.1 seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dataset units)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_fft_plot(signal: np.ndarray, sampling_rate: float, output: Path) -> None:
    """Save a Hann-windowed, single-sided amplitude spectrum."""
    centred = signal - np.mean(signal)
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(centred * window))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    amplitude = 2.0 * spectrum / np.sum(window)

    plt.figure(figsize=(11, 4))
    plt.plot(frequencies, amplitude, linewidth=0.6)
    plt.xlim(0, sampling_rate / 2)
    plt.title("Drive-end vibration FFT amplitude spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (approx. dataset units)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Path to a CWRU .mat or numeric .csv file")
    parser.add_argument("--signal-key", help="MATLAB vibration key, e.g. X100_DE_time")
    parser.add_argument("--csv-column", help="Numeric CSV column name")
    parser.add_argument("--sampling-rate", type=float, required=True,
                        help="Verified sampling frequency in samples/second (Hz)")
    parser.add_argument("--window-size", type=int, default=2048,
                        help="Samples per feature window (default: 2048)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Window overlap fraction from 0 to less than 1 (default: 0.5)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/cwru_analysis"),
                        help="Directory for plots, JSON summary, and window features")
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"Input file does not exist: {args.file}")
    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive")

    signal = load_signal(
        args.file,
        signal_key=args.signal_key,
        csv_column=args.csv_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall = extract_features(signal, args.sampling_rate)
    windows = segment_signal(signal, args.window_size, args.overlap)
    window_features = extract_feature_rows(windows, args.sampling_rate)

    rpm = read_rpm(args.file)
    summary = {
        "source_file": str(args.file),
        "signal_key": args.signal_key,
        "sampling_rate_hz": args.sampling_rate,
        "rpm": rpm,
        "sample_count": int(signal.size),
        "duration_seconds": float(signal.size / args.sampling_rate),
        "window_size_samples": args.window_size,
        "window_overlap_fraction": args.overlap,
        "window_count": len(window_features),
        "overall_features": overall,
    }

    save_time_plot(signal, args.sampling_rate, args.output_dir / "time_signal.png")
    save_fft_plot(signal, args.sampling_rate, args.output_dir / "fft_spectrum.png")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    import pandas as pd
    pd.DataFrame(window_features).to_csv(
        args.output_dir / "window_features.csv", index=False
    )

    print(json.dumps(summary, indent=2))
    print(f"\nAnalysis outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
