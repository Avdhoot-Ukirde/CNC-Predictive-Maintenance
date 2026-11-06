"""Apply a trained normal-baseline Isolation Forest to a vibration recording."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loader import load_signal
from src.feature_extraction import extract_feature_rows
from src.preprocessing import segment_signal

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--model", type=Path, default=Path("reports/anomaly/normal_isolation_forest.joblib"))
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--signal-key")
    parser.add_argument("--csv-column")
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly/anomaly_predictions.csv"))
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"Input file does not exist: {args.file}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. Train it with scripts/train_anomaly_detector.py first.")
    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive")
    bundle = joblib.load(args.model)
    signal = load_signal(args.file, args.signal_key, args.csv_column)
    window_size = int(bundle["window_size_samples"])
    overlap = float(bundle["overlap_fraction"])
    windows = segment_signal(signal, window_size, overlap)
    features = pd.DataFrame(extract_feature_rows(windows, args.sampling_rate))
    columns = bundle["feature_columns"]
    if set(columns) - set(features.columns):
        parser.error("The model requires features that could not be extracted.")
    features = features[columns]
    model = bundle["model"]
    result = features.copy()
    result.insert(0, "window_index", range(len(result)))
    result["anomaly_score"] = model.decision_function(features)
    result["is_anomaly"] = model.predict(features) == -1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result["is_anomaly"].value_counts().rename(index={False: "normal", True: "anomaly"}).to_string())
    print(f"Saved window anomaly results to: {args.output.resolve()}")
    print("Anomaly means unusual relative to the training baseline; it does not diagnose a specific fault.")

if __name__ == "__main__":
    main()
