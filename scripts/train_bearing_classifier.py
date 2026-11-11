"""Train and evaluate a CWRU bearing-condition Random Forest classifier.

Expected input files in data/raw/cwru/:
  Normal:      97.mat, 98.mat, 99.mat, 100.mat
  Inner race: 105.mat, 106.mat, 107.mat, 108.mat
  Ball:       118.mat, 119.mat, 120.mat, 121.mat
  Outer race: 130.mat, 131.mat, 132.mat, 133.mat

All are the classic 12 kHz drive-end subset. The split is grouped by
original recording so windows from one recording cannot appear in both
training and test sets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cwru_signal
from src.feature_extraction import extract_feature_rows
from src.preprocessing import segment_signal

# One recording from each load condition per class. Filenames follow the
# official CWRU 12 kHz drive-end file table.
FILE_LABELS = {
    **{f"{number}.mat": "normal" for number in (97, 98, 99, 100)},
    **{f"{number}.mat": "inner_race" for number in (105, 106, 107, 108)},
    **{f"{number}.mat": "ball" for number in (118, 119, 120, 121)},
    **{f"{number}.mat": "outer_race" for number in (130, 131, 132, 133)},
}
CLASS_NAMES = ["normal", "inner_race", "ball", "outer_race"]


def build_feature_table(
    data_dir: Path,
    sampling_rate: float,
    window_size: int,
    overlap: float,
) -> pd.DataFrame:
    """Extract features from every available manifest recording."""
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for filename, label in FILE_LABELS.items():
        path = data_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue

        # Files in this manifest contain a single DE vibration channel.
        signal = load_cwru_signal(path)
        windows = segment_signal(signal, window_size, overlap)
        features = extract_feature_rows(windows, sampling_rate)

        for window_index, feature_row in enumerate(features):
            rows.append({
                **feature_row,
                "label": label,
                "source_file": filename,
                "window_index": window_index,
            })

    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            "Dataset is incomplete. Download the following files from the "
            "official CWRU 12 kHz Drive End dataset and place them in "
            f"{data_dir}:\n  {formatted}"
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No feature rows were created.")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument("--sampling-rate", type=float, default=12000.0)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/classifier"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive")
    if args.window_size < 8:
        parser.error("--window-size must be at least 8")
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1)")

    table = build_feature_table(
        args.data_dir, args.sampling_rate, args.window_size, args.overlap
    )

    feature_columns = [
        column for column in table.columns
        if column not in {"label", "source_file", "window_index"}
    ]
    X = table[feature_columns]
    y = table["label"]
    groups = table["source_file"]

    # Each original recording is one group. StratifiedGroupKFold ensures
    # windows from a recording stay together and attempts to preserve classes.
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=args.random_state
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_files = sorted(groups.iloc[train_indices].unique().tolist())
    test_files = sorted(groups.iloc[test_indices].unique().tolist())

    if set(y_train.unique()) != set(CLASS_NAMES):
        raise RuntimeError("Training split does not contain all four classes.")
    if set(y_test.unique()) != set(CLASS_NAMES):
        raise RuntimeError("Test split does not contain all four classes.")
    if set(train_files) & set(test_files):
        raise RuntimeError("Data leakage: source recordings overlap across splits.")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "feature_dataset.csv", index=False)
    joblib.dump(
        {"model": model, "feature_columns": feature_columns, "class_names": CLASS_NAMES},
        args.output_dir / "bearing_random_forest.joblib",
    )

    report = classification_report(
        y_test, predictions, labels=CLASS_NAMES, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_NAMES)
    summary = {
        "model": "RandomForestClassifier",
        "sampling_rate_hz": args.sampling_rate,
        "window_size_samples": args.window_size,
        "overlap_fraction": args.overlap,
        "recording_count": int(groups.nunique()),
        "window_count": int(len(table)),
        "train_recordings": train_files,
        "test_recordings": test_files,
        "train_window_count": int(len(train_indices)),
        "test_window_count": int(len(test_indices)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "feature_importance": dict(sorted(
            zip(feature_columns, model.feature_importances_),
            key=lambda item: item[1], reverse=True
        )),
        "evaluation_note": (
            "Holdout is grouped by source file, preventing overlapping windows "
            "from one recording crossing train/test. This is still a benchmark "
            "on CWRU test-rig data, not validation on an industrial CNC spindle."
        ),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=CLASS_NAMES
    )
    display.plot(cmap="Blues", values_format="d", xticks_rotation=25)
    plt.title("CWRU bearing condition — held-out recording test")
    plt.tight_layout()
    confusion_path = args.output_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=160)
    plt.close()

    # Track the real holdout evaluation and fitted model; no placeholder metrics.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cnc_bearing_fault_detection")
    with mlflow.start_run(run_name="cwru_random_forest"):
        mlflow.set_tags({
            "dataset": "CWRU 12 kHz drive-end",
            "validation": "recording_grouped_holdout",
            "task": "four_class_bearing_condition_classification",
        })
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": args.random_state,
            "sampling_rate_hz": args.sampling_rate,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "recording_count": int(groups.nunique()),
            "train_recording_count": len(train_files),
            "test_recording_count": len(test_files),
        })
        mlflow.log_metric("accuracy", summary["accuracy"])
        for class_name in CLASS_NAMES:
            class_metrics = report.get(class_name, {})
            for metric_name in ("precision", "recall", "f1-score"):
                if metric_name in class_metrics:
                    mlflow.log_metric(f"{class_name}_{metric_name.replace('-', '_')}", float(class_metrics[metric_name]))
        mlflow.log_dict(summary, "evaluation/metrics.json")
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        mlflow.log_artifact(str(args.output_dir / "bearing_random_forest.joblib"), artifact_path="model_bundle")
        mlflow.sklearn.log_model(model, artifact_path="sklearn_model")

    print(json.dumps({
        "accuracy": summary["accuracy"],
        "train_recordings": train_files,
        "test_recordings": test_files,
        "classification_report": report,
        "outputs": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
