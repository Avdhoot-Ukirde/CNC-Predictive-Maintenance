"""Predict bearing condition for a CWRU .mat or numeric .csv signal."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data_loader import load_signal
from src.inference import load_classifier, predict_signal

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--model", type=Path, default=Path("reports/classifier/bearing_random_forest.joblib"))
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--signal-key")
    parser.add_argument("--csv-column")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/predictions/predictions.csv"))
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"Input file does not exist: {args.file}")
    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive")
    signal = load_signal(args.file, args.signal_key, args.csv_column)
    bundle = load_classifier(args.model)
    predictions = predict_signal(signal, args.sampling_rate, bundle, args.window_size, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(predictions["predicted_condition"].value_counts().to_string())
    print(f"\\nSaved window predictions to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
