"""Fit an Isolation Forest baseline using only normal CWRU recordings.

Expected files in data/raw/cwru/: 97.mat, 98.mat, 99.mat, 100.mat.
This creates an unsupervised novelty detector; it does not identify fault type.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loader import load_cwru_signal
from src.feature_extraction import extract_feature_rows
from src.preprocessing import segment_signal

NORMAL_FILES = ["97.mat", "98.mat", "99.mat", "100.mat"]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument(
        "--normal-files",
        nargs="+",
        default=NORMAL_FILES,
        choices=NORMAL_FILES,
        help="Normal recordings used to fit the detector.",
)
    parser.add_argument("--sampling-rate", type=float, default=12000.0)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly/normal_isolation_forest.joblib"))
    args = parser.parse_args()
    if args.sampling_rate <= 0 or args.window_size < 8:
        parser.error("sampling rate must be positive and window size at least 8")
    if not 0 < args.contamination <= 0.5:
        parser.error("contamination must be in (0, 0.5]")

    if not args.normal_files:
        parser.error("At least one normal recording is required.")

    missing = [
        name
        for name in args.normal_files
        if not (args.data_dir / name).is_file()
]
    if missing:
        parser.error("Missing normal baseline files: " + ", ".join(missing))
    rows = []
    for name in args.normal_files:
        signal_key = f"X{Path(name).stem.zfill(3)}_DE_time"
        signal = load_cwru_signal(args.data_dir / name, signal_key=signal_key)
        windows = segment_signal(signal, args.window_size, args.overlap)
        for index, features in enumerate(extract_feature_rows(windows, args.sampling_rate)):
            rows.append({**features, "source_file": name, "window_index": index})
    table = pd.DataFrame(rows)
    columns = [c for c in table.columns if c not in {"source_file", "window_index"}]
    detector = IsolationForest(
        n_estimators=300, contamination=args.contamination,
        random_state=42, n_jobs=-1
    )
    detector.fit(table[columns])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": detector,
        "feature_columns": columns,
        "normal_source_files": args.normal_files,
        "sampling_rate_hz": args.sampling_rate,
        "window_size_samples": args.window_size,
        "overlap_fraction": args.overlap,
        "contamination": args.contamination,
    }, args.output)
    print(f"Fitted normal-only Isolation Forest on {len(table)} windows from {len(args.normal_files)} recordings: {args.normal_files}")
    print(f"Saved model: {args.output.resolve()}")
    print("This is a baseline novelty detector. Validate false alarms and missed detections on held-out machine data.")

if __name__ == "__main__":
    main()
