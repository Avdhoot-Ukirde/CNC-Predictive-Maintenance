# CWRU four-class bearing classifier

## Dataset selection

The first recording, `100.mat`, is the normal 3 HP baseline (about 1,730 RPM). To compare operating conditions consistently, the first fault recordings at 3 HP and 0.007-inch seeded fault diameter are:

| File | Condition | Load | Approx. speed |
|---|---|---:|---:|
| 100.mat | Normal | 3 HP | 1,730 RPM |
| 108.mat | Inner-race fault | 3 HP | 1,730 RPM |
| 121.mat | Ball fault | 3 HP | 1,730 RPM |
| 133.mat | Outer-race fault, 6 o'clock | 3 HP | 1,730 RPM |

**Correction to an earlier suggestion:** files 105, 118, and 130 are 0 HP recordings, not 3 HP. Use 108, 121, and 133 when matching your 3 HP baseline.

The script is configured for the classic 12 kHz drive-end (DE) subset. The full manifest contains four load conditions per class:

- Normal: 97, 98, 99, 100
- Inner race: 105, 106, 107, 108
- Ball: 118, 119, 120, 121
- Outer race: 130, 131, 132, 133

This broader 16-file starter set gives multiple independent source recordings per class, enabling a grouped holdout split. It also varies load/speed across files, so results should be interpreted as a CWRU benchmark rather than a controlled same-load comparison.

Official source: [CWRU 12k Drive End Bearing Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data). The CWRU table documents the 0.007-inch fault diameter, four motor loads, approximate speeds, and outer-race positions.

## Download and organize the data

Download these 16 MATLAB files from the official CWRU data page and place them directly in:

```text
data/raw/cwru/
  97.mat  98.mat  99.mat  100.mat
  105.mat 106.mat 107.mat 108.mat
  118.mat 119.mat 120.mat 121.mat
  130.mat 131.mat 132.mat 133.mat
```

The repository does not redistribute the dataset. Keep the downloaded files local; `.gitignore` excludes raw data.

## Train and evaluate

From the repository root, install dependencies and run:

```bash
pip install -r requirements.txt
python scripts/train_bearing_classifier.py
```

The script:

1. Loads the drive-end channel from each recording.
2. Segments each signal into 2,048-sample windows with 50% overlap.
3. Extracts time-domain and frequency-domain features.
4. Uses `StratifiedGroupKFold` with the original recording as the group.
5. Trains a Random Forest and evaluates on held-out recordings.
6. Saves the feature table, model, metrics, and confusion matrix.

Outputs are written to `reports/classifier/`:

- `feature_dataset.csv`
- `bearing_random_forest.joblib`
- `metrics.json`
- `confusion_matrix.png`

## Why group splitting matters

Overlapping windows from the same source recording are highly related. A random row-wise train/test split could place near-duplicate windows in both sets and produce an overly optimistic score. Grouping by source file prevents this direct leakage.

Even with recording-level separation, this remains a small benchmark from one laboratory test rig. It does not prove generalization to a CNC spindle, other bearing types, sensor mountings, speeds, or loads. Do not advertise the accuracy as industrial fault-detection performance.

## Evaluation result and current status

The repository README records a completed grouped-holdout evaluation with **100% accuracy on 825 held-out windows** for the current CWRU benchmark. This result is produced by the training workflow above, which uses `StratifiedGroupKFold` and keeps each source recording entirely within either the training or test split.

The reported score is specific to this dataset subset, feature pipeline, model configuration, and selected grouped split. The 825 windows are not 825 independent machine recordings: windows are segmented from a small number of CWRU recordings, with 50% overlap between adjacent windows. This result must not be interpreted as industrial CNC accuracy or evidence of generalization to other machines, bearing types, sensor installations, speeds, or loads.

The training script writes the actual evaluation details to `reports/classifier/metrics.json`, including the accuracy, train/test recording lists, window counts, and per-class metrics. It also writes the fitted model, feature dataset, and confusion matrix. These generated files are local/DVC-managed artifacts rather than a committed metrics report in this Git documentation. To reproduce the evaluation, obtain all 16 recordings listed above and run:

```bash
python scripts/train_bearing_classifier.py
```

The script will fail explicitly if any required recording is missing. Do not substitute an expected score for a newly generated result, and retain the limitations above whenever quoting the reported 100% benchmark score.
